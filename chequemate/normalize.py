"""Turn raw OCR text into typed values.

Each normaliser returns a Field so the caller can distinguish "absent",
"unparseable" and "parsed". Nothing here talks to Azure.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

CENTS = Decimal("0.01")

from .models import Field, ParseStatus, TokenMatchResult

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _clean(text: str | None) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return _WS.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# payee
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]")
_LEADING_ARTICLE = re.compile(r"^(the)\s+", re.IGNORECASE)


def canonical_payee(text: str | None) -> str:
    """Case/punctuation/article-insensitive form used for comparison."""
    s = _clean(text).lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    s = _LEADING_ARTICLE.sub("", s)
    return s


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize_payee(raw: str | None, **kw) -> Field:
    text = _clean(raw)
    if not text:
        return Field("payee", parse_status=ParseStatus.ABSENT, raw_text=raw, **kw)
    return Field(
        "payee",
        value=canonical_payee(text),
        raw_text=text,
        parse_status=ParseStatus.OK,
        **kw,
    )


# ---------------------------------------------------------------------------
# numeric amount
# ---------------------------------------------------------------------------

_AMOUNT_RE = re.compile(r"\d[\d,.\s]*\d|\d")
# a '.' or ',' with 1-2 trailing digits is the decimal point; anything else
# is a grouping separator ('1,000' vs '2 000,50' vs '125.50')
_DECIMAL_SEP = re.compile(r"[.,](\d{1,2})$")
# the candidate must be a well-formed amount end to end. Without this,
# '125.501' silently becomes 125501.00 instead of being rejected.
_VALID_NUMBER = re.compile(r"(?:\d{1,3}(?:[,\s]\d{3})+|\d+)(?:[.,]\d{1,2})?")


def normalize_amount_numeric(raw: str | None, **kw) -> Field:
    """'$\\n125.50' -> Decimal('125.50');  '$ 300' -> Decimal('300.00')."""
    text = _clean(raw)
    if not text:
        return Field("amount_numeric", parse_status=ParseStatus.ABSENT,
                     raw_text=raw, **kw)

    body = text.replace("$", " ").strip()
    m = _AMOUNT_RE.search(body)
    if not m:
        return Field("amount_numeric", raw_text=text,
                     parse_status=ParseStatus.UNPARSEABLE,
                     note="no digits found", **kw)

    number = m.group(0)
    if not _VALID_NUMBER.fullmatch(number):
        return Field("amount_numeric", raw_text=text,
                     parse_status=ParseStatus.UNPARSEABLE,
                     note=f"malformed amount {number!r}", **kw)

    sep = _DECIMAL_SEP.search(number)
    if sep:
        cents = sep.group(1).ljust(2, "0")
        whole = number[: sep.start()]
    else:
        cents, whole = "00", number
    whole = re.sub(r"[,.\s]", "", whole) or "0"
    try:
        value = Decimal(f"{whole}.{cents}")
    except InvalidOperation:
        return Field("amount_numeric", raw_text=text,
                     parse_status=ParseStatus.UNPARSEABLE,
                     note="not a decimal", **kw)
    return Field("amount_numeric", value=value, raw_text=text,
                 parse_status=ParseStatus.OK, **kw)


# ---------------------------------------------------------------------------
# word amount
# ---------------------------------------------------------------------------

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_NOISE = {"and", "dollars", "dollar", "dollers", "only", "of", "xx", "no",
          "none"}

# Deterministic repair of misspellings seen on real cheques. A fixed table,
# NOT fuzzy matching: on a money field, snapping an unrecognised word to its
# nearest neighbour can turn 'sixty' into 'sixteen' without anyone noticing.
_MISSPELLINGS = {
    "fivety": "fifty", "fifthy": "fifty", "ninty": "ninety",
    "eigthy": "eighty", "eightteen": "eighteen", "thirtheen": "thirteen",
    "fourty": "forty", "hundered": "hundred", "thousend": "thousand",
    "twelth": "twelve", "fourteeen": "fourteen",
}

# '50/100', '/100', 'no/100', 'xx/100', 'none/100'
_CENTS_FRACTION = re.compile(
    r"(?P<num>\d{1,2}|none|no|xx|zero)?\s*/\s*(?P<den>100|00|xx)",
    re.IGNORECASE,
)
# OCR very often drops the '/' in the printed '.../100 DOLLARS' cents suffix,
# leaving a bare '100' (optionally preceded by its handwritten numerator)
# floating in the text - e.g. '36 100 DOLLARS' for '36/100', or '- 100
# DOLLARS' when the numerator itself was lost too. A bare '100' token is an
# unambiguous denominator - cheques always print that exact suffix - so
# recovering it is safe, unlike a bare small number with no '100' nearby
# (still rejected below: that really is ambiguous).
_TRAILING_CENTS_100 = re.compile(r"(?P<num>\d{1,2})?[^\w]{0,4}\b100\b")
_CENT_WORD = re.compile(r"\bcents?\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# cents-suffix construct stripping (ruleset 1.7.0)
#
# Every Canadian cheque prints the '.../100 DOLLARS' cents construct as fixed
# form boilerplate (CPA Standard 006) - the payer's handwritten numerator
# there is redundant with, and far less reliably OCR'd than, the numeric
# amount box a few centimetres away. Rather than enumerate every OCR
# mangling of that numerator as a recognised pattern (an unbounded task -
# '460 100', '4/kg 100', a bare stray digit before 'DOLLARS' with the '/'
# lost entirely - each new mangling was a fresh UNPARSEABLE), find the one
# unambiguous anchor (a literal '100', 'xx', 'no' or 'none' denominator - the
# fixed part of the printed construct) and discard the ENTIRE construct
# outward from it, without ever trying to read what the numerator said. Its
# value plays no further role - amount_words's caller supplies the numeric
# amount's cents instead (see normalize_amount_words).
# ---------------------------------------------------------------------------

_ANCHOR_100 = re.compile(
    r"^(?:\d{0,3}|xx|no|none|zero)?[/\\.\-]*100$", re.IGNORECASE)


def _is_dollar_word(tok: str) -> bool:
    """True for a token that is (or, once any surrounding/embedded
    non-letter noise - digits, '/', '-', OCR garbage like '->>' - is split
    off, entirely consists of) recognised dollar-amount vocabulary: a real
    number word or a connector ('and', 'dollars', ...). Extracting every
    letters-only run (rather than just stripping the token's edges) is what
    correctly separates 'hundred->>' -> ['hundred'] (real word, OCR-glued
    to unrelated punctuation - keep it) from '4/kg' -> ['kg'] (numerator
    noise - discard it), and handles 'eighty-one' the same way it always
    did (split into two number words)."""
    pieces = re.findall(r"[a-z]+", tok)
    return bool(pieces) and all(_is_number_word(p) or p in _NOISE for p in pieces)


_HAS_DIGIT_OR_SLASH = re.compile(r"[\d/\\]")


def _looks_like_numerator_noise(tok: str) -> bool:
    """True only for a token that is structurally part of the printed
    numerator/fraction (contains a digit or a slash) AND isn't itself
    recognisable dollar-amount text once that noise is stripped off. This
    is deliberately narrower than 'not a recognised word': a token with NO
    digit or slash at all (e.g. a completely garbled dollar word like
    'sihatred') is never numerator noise, however unrecognisable it is -
    the printed cents construct is inherently digit/slash-shaped, so
    anything without either character was never part of it. Without this
    distinction, a genuinely unparseable dollar-words field (garbage in
    the DOLLARS portion, unrelated to the cents suffix) would have its
    entire content silently swallowed by the backward walk, turning an
    honest 'could not read the written amount' UNABLE into a confidently
    wrong parsed value - exactly the failure mode this codebase has
    fixed elsewhere (see rules.py's _BANK_KEYWORDS-removal note and the
    SIGNATURE_ZONE_FRAC history in CLAUDE.md)."""
    return bool(_HAS_DIGIT_OR_SLASH.search(tok)) and not _is_dollar_word(tok)


def _strip_cents_suffix_construct(lowered: str) -> tuple[str, bool]:
    """Remove the trailing '.../100'-style construct, walking back from the
    denominator anchor through however much numerator noise OCR produced,
    stopping the instant a token is no longer numerator-shaped noise.
    Returns (text_with_construct_removed, whether_anything_was_found)."""
    tokens = lowered.split()
    anchor = next((i for i in range(len(tokens) - 1, -1, -1)
                  if _ANCHOR_100.match(tokens[i])), None)
    if anchor is None:
        return lowered, False

    start = anchor
    j = anchor - 1
    while j >= 0 and _looks_like_numerator_noise(tokens[j]):
        start = j
        j -= 1
    return " ".join(tokens[:start] + tokens[anchor + 1:]), True


def _words_to_int(tokens: list[str]) -> int | None:
    total = current = 0
    seen = False
    for tok in tokens:
        tok = _MISSPELLINGS.get(tok, tok)
        if tok in _ONES:
            current += _ONES[tok]
            seen = True
        elif tok in _TENS:
            current += _TENS[tok]
            seen = True
        elif tok == "hundred":
            current = (current or 1) * 100
            seen = True
        elif tok in _SCALES:
            total += (current or 1) * _SCALES[tok]
            current = 0
            seen = True
        elif tok in _NOISE:
            continue
        else:
            return None  # unknown token -> refuse to guess
    return total + current if seen else None


def _is_number_word(tok: str) -> bool:
    tok = _MISSPELLINGS.get(tok, tok)
    return tok in _ONES or tok in _TENS or tok == "hundred" or tok in _SCALES


def _split_spelled_cents(tokens: list[str]) -> tuple[list[str], int | None]:
    """Pull 'and sixty cents' off the end. Returns (dollar_tokens, cents).

    Without this, 'Three Hundred dollars and sixty cents' parses as $360 —
    a wrong value rather than a refusal, which is the dangerous failure.
    """
    try:
        idx = next(i for i, t in enumerate(tokens) if t in ("cent", "cents"))
    except StopIteration:
        return tokens, None

    j = idx
    while j > 0 and _is_number_word(tokens[j - 1]):
        j -= 1
    if j == idx:  # 'cents' with no number in front of it
        return tokens[:idx] + tokens[idx + 1:], None

    cents = _words_to_int(tokens[j:idx])
    if cents is None or not 0 <= cents <= 99:
        return tokens, None  # let the caller fail loudly
    return tokens[:j] + tokens[idx + 1:], cents


def normalize_amount_words(raw: str | None, numeric_cents: int | None = None,
                           **kw) -> Field:
    """'One Hundred and twenty five dollars and 50/100 DOLLARS' -> 125.50.

    `numeric_cents` (ruleset 1.7.0) is the already-normalised numeric
    amount's cents component, when the caller has one (extract.to_normalized
    always supplies it when amount_numeric parsed). When given, the printed
    '.../100' cents-suffix construct is discarded outright rather than
    parsed - see `_strip_cents_suffix_construct`'s module-level docstring -
    and its cents value is taken from `numeric_cents` instead, which OCR
    reads far more reliably than a handwritten numerator squeezed onto the
    legal-amount line. Genuinely spelled-out cents ('and sixty cents') are
    NOT the printed construct - they're an independent written statement -
    and still take priority when present. With no `numeric_cents` (e.g. a
    caller with only the words text, or the numeric field itself didn't
    parse), this falls back to the pre-1.7.0 behaviour of trying to read the
    numerator from the words text itself.
    """
    text = _clean(raw)
    if not text:
        return Field("amount_words", parse_status=ParseStatus.ABSENT,
                     raw_text=raw, **kw)

    lowered = text.lower()
    cents: int | None = None
    cents_from_numeric_fallback = False

    if numeric_cents is not None:
        lowered, found = _strip_cents_suffix_construct(lowered)
        cents_from_numeric_fallback = found
    else:
        m = _CENTS_FRACTION.search(lowered)
        if m:
            num = (m.group("num") or "").lower()
            cents = int(num) if num.isdigit() else 0
            lowered = lowered[: m.start()] + " " + lowered[m.end():]
        else:
            m2 = _TRAILING_CENTS_100.search(lowered)
            if m2:
                cents = int(m2.group("num")) if m2.group("num") else 0
                lowered = lowered[: m2.start()] + " " + lowered[m2.end():]

    # Any digit left after the construct is removed is contamination
    # elsewhere in the line: the OCR merged the courtesy amount into the
    # legal line, or misread a word. Unlike the trailing '.../100' construct,
    # this is genuinely ambiguous - it might be real corruption of the
    # dollar amount itself - so it is still rejected outright, never stripped.
    if re.search(r"\d", lowered):
        stray = "".join(re.findall(r"\d+", lowered))
        return Field("amount_words", raw_text=text,
                     parse_status=ParseStatus.UNPARSEABLE,
                     note=f"stray digits in word amount: {stray!r}", **kw)

    tokens = re.findall(r"[a-z]+", lowered.replace("-", " "))
    if cents is None:
        tokens, cents = _split_spelled_cents(tokens)
    used_numeric_fallback = False
    if cents is None and cents_from_numeric_fallback:
        cents = int(numeric_cents)
        used_numeric_fallback = True
    cents = cents or 0

    # 'no dollar words at all' (e.g. 'sixty cents only') means zero dollars.
    # 'dollar words present but unrecognised' must fail — otherwise
    # 'Seventty five and 10/100' silently becomes $0.10.
    meaningful = [t for t in tokens if t not in _NOISE]
    dollars = _words_to_int(tokens) if meaningful else (0 if cents else None)
    if dollars is None:
        return Field("amount_words", raw_text=text,
                     parse_status=ParseStatus.UNPARSEABLE,
                     note="unrecognised number words", **kw)

    value = (Decimal(dollars) + Decimal(cents) / 100).quantize(CENTS)
    note = None
    if used_numeric_fallback:
        note = ("cents suffix in the written amount was unreadable as "
                f"printed ({text[-32:]!r}); cents taken from the numeric "
                f"amount field instead — cents figure is still from the "
                f"written amount as usual")
    return Field("amount_words", value=value,
                 raw_text=text, parse_status=ParseStatus.OK, note=note,
                 degraded=used_numeric_fallback, **kw)


# ---------------------------------------------------------------------------
# token-match amount verification (ruleset 1.9.0)
#
# When the exact word-amount parse fails (or produces a value that doesn't
# match the numeral), this is a fallback: instead of independently reading
# the written amount cold, verify it against the numeral we ALREADY have.
# Derive the number words the numeral implies ($933 -> nine, hundred,
# thirty, three), search the extracted text for those tokens, and ignore
# everything else - OCR garbage, punctuation, the printed '.../100
# DOLLARS' construct (none of it is alphabetic, so the tokenizer already
# drops it). This resolves cases a strict, order-sensitive parser can't:
# real cheques get garbled in ways ("sixhatred", "eight-one" for
# "eighty-one") that scramble structure while leaving the actual digit
# words present and findable.
#
# THE SCALE TRAP, and why it must not be skipped: "nine", "hundred",
# "thirty", "three" are all present in "nine thousand three hundred
# thirty" too - a presence-only check can't tell a genuine $933 from a
# cheque altered from $9,330. So after confirming presence, this also
# confirms no (scale word, multiplier) claim is stated in the text that
# no valid reading of the numeral permits - not just "is 'thousand'
# present", but "is 'nine hundred' or is it 'three hundred' - only the
# multiplier(s) actually implied by the numeral are allowed to appear.
#
# FUZZY MATCHING, and the collision risk it creates: OCR corrupts short,
# common words too ("thirty" -> "thirtty"/"thurty" - both distance 1 from
# "thirty"). But number words are semantically loaded and often close to
# EACH OTHER: nine/five and two/ten are both edit-distance 2, eight/eighty
# and million/billion are edit-distance 1 - a naive fuzzy match at any of
# those distances would let the scale trap back in through the side door
# (a genuine "eighty" misread as "eight" is indistinguishable, by edit
# distance alone, from a genuine "eight" that a fuzzy matcher wrongly
# stretches to satisfy an expected "eighty"). The fix isn't a smaller
# distance - "million"/"billion" and "eight"/"eighty" already collide at
# distance 1, the smallest distance that still catches real OCR noise -
# it's a categorical rule: a text token that is ITSELF a real, recognised
# number/scale word (after the existing misspelling table) is taken at
# face value and is NEVER fuzzy-substituted for a different expected
# token, no matter how close. Only a token that isn't a real word at all
# (genuine garbage) is eligible for fuzzy rescue. This is what lets
# "thirtty" satisfy "thirty" while keeping "eight" from ever satisfying
# "eighty" - verified against the full number-word vocabulary, not just
# spot-checked (see tests/test_cheque.py's
# test_no_real_number_word_fuzzy_collides_with_a_different_one). Bound:
# distance <= 1 for tokens up to 5 characters, <= 2 for longer ones -
# generous now that real-word collisions are excluded categorically
# rather than by distance alone.
#
# ALTERNATE PHRASINGS: $1,150 is genuinely written either as "one thousand
# one hundred fifty" or "eleven hundred fifty" - two different token sets
# for the same amount. expected_amount_token_forms() generates both the
# standard (thousand + hundred) and, for 1000-9999, the "compact hundreds"
# reading (a real convention seen on this project's own batch - record
# CHQ-20260824-0008 is genuinely written "Twenty-two hundred eighteen" for
# $2,218) - a match against EITHER form is accepted. Hyphenation ("thirty-
# three" vs "thirty three") and "and" being present or absent need no
# special handling: the existing tokenizer already splits on hyphens, and
# "and" is neither an expected token nor a scale word, so its presence or
# absence never affects a token-set search either way.
# ---------------------------------------------------------------------------

_SCALE_WORDS = frozenset(_SCALES) | {"hundred"}
_CANONICAL_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven",
                   "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                   "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                   "nineteen")
_CANONICAL_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
                   60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}

# Short tokens get the tighter bound - see the module-level note above for
# why the real-word exclusion (not the distance) is what actually prevents
# collisions like nine/five and eight/eighty; this bound just controls how
# much genuine OCR noise on non-dictionary garbage is tolerated.
_FUZZY_SHORT_MAX_LEN = 5
_FUZZY_SHORT_DISTANCE = 1
_FUZZY_LONG_DISTANCE = 2


def _fuzzy_bound(token: str) -> int:
    return _FUZZY_SHORT_DISTANCE if len(token) <= _FUZZY_SHORT_MAX_LEN else _FUZZY_LONG_DISTANCE


def _real_number_word(tok: str) -> str | None:
    """`tok`, after the existing misspelling table, if it is a real
    recognised number/scale word - else None. Deliberately not fuzzy: this
    is what keeps a real, different word (an OCR-clean "eight" facing an
    expected "eighty") from ever being fuzzy-collapsed into something else,
    and what the scale-word guard uses to decide what the text confidently
    states, as opposed to what it might, fuzzily, be noise for."""
    corrected = _MISSPELLINGS.get(tok, tok)
    if corrected in _ONES or corrected in _TENS or corrected == "hundred" or corrected in _SCALES:
        return corrected
    return None


def _amount_word_tokens(n: int) -> list[str]:
    """0-999 -> its ones/tens/hundred token sequence, in canonical spelling.
    Empty list for 0 (no token needed for a zero group)."""
    if n <= 0:
        return []
    if n < 20:
        return [_CANONICAL_ONES[n]]
    if n < 100:
        tens, ones = divmod(n, 10)
        tokens = [_CANONICAL_TENS[tens * 10]]
        if ones:
            tokens.append(_CANONICAL_ONES[ones])
        return tokens
    hundreds, rest = divmod(n, 100)
    tokens = [_CANONICAL_ONES[hundreds], "hundred"]
    tokens.extend(_amount_word_tokens(rest))
    return tokens


def _standard_form(dollars: int) -> list[str]:
    tokens: list[str] = []
    remaining = dollars
    for scale_name, scale_val in (("billion", 10**9), ("million", 10**6),
                                  ("thousand", 1000)):
        group, remaining = divmod(remaining, scale_val)
        if group:
            tokens.extend(_amount_word_tokens(group))
            tokens.append(scale_name)
    tokens.extend(_amount_word_tokens(remaining))
    return tokens


def expected_amount_token_forms(dollars: int) -> list[list[str]]:
    """Every plausible token form for a dollar integer: the standard
    thousand/hundred grouping always, plus (for 1000-9999 only, where the
    alternate genuinely differs) the 'compact hundreds' reading - see the
    module note above for why both are real, not speculative."""
    if dollars <= 0:
        return [["zero"]]
    forms = [_standard_form(dollars)]
    if 1000 <= dollars <= 9999:
        hundred_group, remainder = divmod(dollars, 100)
        compact = _amount_word_tokens(hundred_group) + ["hundred"] + _amount_word_tokens(remainder)
        if compact != forms[0]:
            forms.append(compact)
    return forms


def _confident_scale_claims(tokens: list[str]) -> dict[str, int]:
    """(scale_word -> multiplier) claims the token stream CONFIDENTLY
    states - strict recognition only (misspelling-normalized exact
    matches), never fuzzy. A scale word is claimed by the ones/tens word,
    or ones+tens compound ('twenty two hundred'), immediately before it.
    Conflicting claims for the same scale are dropped as ambiguous, not
    guessed at."""
    claims: dict[str, int | None] = {}
    normalized = [_MISSPELLINGS.get(t, t) for t in tokens]
    for i, tok in enumerate(normalized):
        if tok not in _SCALE_WORDS or i == 0:
            continue
        prev = normalized[i - 1]
        mult = _ONES.get(prev)
        if mult is None:
            mult = _TENS.get(prev)
        if mult is None and i >= 2:
            tens_val, ones_val = _TENS.get(normalized[i - 2]), _ONES.get(prev)
            if tens_val is not None and ones_val is not None:
                mult = tens_val + ones_val
        if not mult:
            continue
        if tok in claims and claims[tok] != mult:
            claims[tok] = None
        else:
            claims[tok] = mult
    return {k: v for k, v in claims.items() if v is not None}


def _match_form(expected_tokens: list[str], text_tokens: list[str]
                ) -> tuple[bool, list[dict], list[str]]:
    """Whether every token in `expected_tokens` can be found in
    `text_tokens` (each text token consumed at most once, so a single
    'one' can't satisfy two expected 'one' slots). Exact/misspelling
    match tried first; fuzzy match against non-real-word tokens only,
    second - see this module's token-match section docstring."""
    available = list(text_tokens)
    found: list[dict] = []
    missing: list[str] = []
    for expected in expected_tokens:
        match_pos = None
        for pos, tok in enumerate(available):
            if _MISSPELLINGS.get(tok, tok) == expected:
                match_pos = pos
                found.append({"expected": expected, "matched_text": tok, "distance": 0})
                break
        if match_pos is None:
            best = None
            for pos, tok in enumerate(available):
                if _real_number_word(tok) is not None:
                    continue
                d = levenshtein(tok, expected)
                if d <= _fuzzy_bound(expected) and (best is None or d < best[2]):
                    best = (pos, tok, d)
            if best is not None:
                match_pos, tok, d = best
                found.append({"expected": expected, "matched_text": tok, "distance": d})
        if match_pos is None:
            missing.append(expected)
        else:
            del available[match_pos]
    return (len(missing) == 0), found, missing


def verify_amount_by_tokens(raw_text: str | None, dollars: int,
                            numeral_cents: int | None = None) -> TokenMatchResult:
    """Verify `raw_text` against an already-known dollar integer by fuzzy
    token presence plus the scale-word guard. Fallback only - see this
    module's token-match section docstring and check_amounts_match's
    decision table for how the result is used.

    `numeral_cents` (0-99), when given, is compared against any stray digit
    left in the text after the recognised cents-suffix construct is
    stripped - see the stray-digit guard below for why this can't just be
    "any stray digit blocks a match" (record CHQ-20260824-0013's genuine
    '42' cents numerator, sitting outside any '/100' the parser recognised,
    would be wrongly blocked by that)."""
    if not raw_text or not raw_text.strip():
        return TokenMatchResult(outcome="tokens_missing",
                                missing=expected_amount_token_forms(dollars)[0])

    text_tokens = re.findall(r"[a-z]+", raw_text.lower().replace("-", " "))
    forms = expected_amount_token_forms(dollars)

    best_match: tuple[list[str], list[dict]] | None = None
    worst_case: tuple[list[str], list[str], list[dict]] | None = None
    for form in forms:
        ok, found, missing = _match_form(form, text_tokens)
        if ok:
            best_match = (form, found)
            break
        if worst_case is None or len(missing) < len(worst_case[1]):
            worst_case = (form, missing, found)

    if best_match is None:
        form, missing, found = worst_case
        return TokenMatchResult(outcome="tokens_missing", form_used=form,
                                found=found, missing=missing)

    matched_form, found = best_match

    # A stray digit the recognised cents-suffix construct doesn't account
    # for, and that DISAGREES with the numeral's own cents, is exactly the
    # case that surfaced this guard: CHQ-20260824-0020's words read 'Five
    # Hundred Ninety Dollars-6 100 DOLLARS' against a $590.00 numeral -
    # "five", "hundred", "ninety" are all genuinely present, so a word-only
    # check calls that a match, but the un-placed '6' right next to '/100'
    # is legible evidence of a real cents figure (visually confirmed
    # against the source image: the cheque appears to actually say $590.06)
    # that the numeral disagrees with. Token-match only ever verifies
    # WORDS - it has no way to weigh a bare digit either way - so the
    # honest answer when one is present, unaccounted for, AND inconsistent
    # with the numeral's cents is "can't confirm", never a confident match.
    # A stray digit that MATCHES the numeral's cents (CHQ-20260824-0013's
    # genuine '42' numerator, unstripped only because the parser never
    # associated it with the printed '/100' line) is not a discrepancy and
    # must not block anything - this check is deliberately about
    # disagreement, not mere presence. Runs before the scale guard so a
    # genuine stray-digit disagreement blocks a PASS even when the scale
    # guard alone would have found nothing wrong.
    stripped, _ = _strip_cents_suffix_construct(raw_text.lower())
    stray = re.search(r"\d{1,2}", stripped)
    if stray and (numeral_cents is None or int(stray.group(0)) != numeral_cents):
        return TokenMatchResult(outcome="tokens_missing", form_used=matched_form,
                                found=found, stray_digit=stray.group(0))

    permitted: dict[str, set[int]] = {}
    for form in forms:
        for scale, mult in _confident_scale_claims(form).items():
            permitted.setdefault(scale, set()).add(mult)

    found_claims = _confident_scale_claims(text_tokens)
    unexpected = {s: m for s, m in found_claims.items()
                 if s not in permitted or m not in permitted[s]}
    if unexpected:
        return TokenMatchResult(outcome="contradiction", form_used=matched_form,
                                found=found, unexpected_claims=unexpected)
    return TokenMatchResult(outcome="matched", form_used=matched_form, found=found)


# ---------------------------------------------------------------------------
# date
# ---------------------------------------------------------------------------

_GUIDE_TOKEN = re.compile(r"\b[DMY]{1,4}\b", re.IGNORECASE)
_ISO = re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b")

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
# '12 Mar 2026' / '12th March, 2026'
_DAY_MONTH_YEAR = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?[\s,.\-]+({_MONTH_ALT})[\s,.\-]+(\d{{2,4}})\b",
    re.IGNORECASE)
# 'March 12 2026' / 'Mar 12th, 2026'
_MONTH_DAY_YEAR = re.compile(
    rf"\b({_MONTH_ALT})[\s,.\-]+(\d{{1,2}})(?:st|nd|rd|th)?[\s,.\-]+(\d{{2,4}})\b",
    re.IGNORECASE)


def _try(y: int, m: int, d: int) -> date | None:
    if y < 100:
        y += 2000 if y < 70 else 1900
    try:
        return date(y, m, d)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# six-digit date resolution (ruleset 1.10.0)
#
# A 6-digit read genuinely admits two DIFFERENT, independently-motivated
# structural hypotheses, not one:
#
#   (a) a genuine short year - the cheque's date box was printed/written as
#       DDMMYY or MMDDYY, so the read is complete as-is. This is the
#       ORIGINAL heuristic this module has always used, unchanged here:
#       _try() already pivots any 2-digit year (< 70 -> 20xx, else 19xx),
#       the same universal convention every date parser uses for a short
#       year, not something invented for this batch.
#   (b) a truncated CPA Standard 006 grid - Canadian cheques print the date
#       as a YYYY MM DD boxed grid, and this project has independently
#       confirmed (via the raw Azure responses) that DI's read of that grid
#       loses digits from the LEADING edge, never the middle or the end -
#       every clean read in this batch is a full 8 digits, and every short
#       read shows the deficit at the front. A 6-digit read under this
#       hypothesis is missing the whole leading century+decade portion of
#       the year, leaving exactly the year's own last 2 digits followed by
#       month and day - which is STRUCTURALLY the same shape as (a), just
#       with the 2-digit fragment in a different position, so the exact
#       same pivot convention applies to it too.
#
# Both hypotheses are equally well-justified by the form/extractor, and
# NEITHER is preferred over the other - not by which one looks more
# "recent", not by any other plausibility judgment (see rules.check_date /
# Config.max_age_months and reject_postdated, which are where that
# judgment belongs, not the parser). Two records use the SAME 6 digits and
# the SAME pivot rule to reach genuinely different dates (a real cheque
# from this batch: 2026-08-30 under (b), 2030-08-26 under (a)) - that is
# real, structural ambiguity, not noise, and the only honest answer is
# UNABLE, not a confident pick of either one. A digit string that, under
# BOTH hypotheses' every valid sub-reading, converges on one date is
# resolved; anything else is not - this is the same "generate every
# interpretation, converge or refuse" discipline as normalize_amount_words'
# token-match fallback, applied here to a different field.
# ---------------------------------------------------------------------------

def _resolve_six_digit_date(digits: str, text: str, **kw) -> Field:
    candidates: dict[date, list[str]] = {}

    def add(value: date | None, label: str) -> None:
        if value is not None:
            candidates.setdefault(value, []).append(label)

    # (a) genuine short year: DDMMYY and MMDDYY, both trying the trailing
    # 2 digits as the (pivoted) year - the module's original heuristic.
    add(_try(int(digits[4:6]), int(digits[2:4]), int(digits[0:2])), "DDMMYY")
    add(_try(int(digits[4:6]), int(digits[0:2]), int(digits[2:4])), "MMDDYY")
    # (b) truncated CPA year-leading grid: the year's own last 2 digits
    # (pivoted the same way) lead, followed by month and day.
    add(_try(int(digits[0:2]), int(digits[2:4]), int(digits[4:6])), "truncated-YYYYMMDD")

    if len(candidates) == 1:
        (value, labels), = candidates.items()
        return Field("cheque_date", value=value, raw_text=text,
                     parse_status=ParseStatus.OK,
                     note=f"6-digit date read resolved via {'/'.join(labels)} "
                          f"(every other valid reading of the same digits "
                          f"agreed)", **kw)
    if len(candidates) > 1:
        readings = ", ".join(f"{v.isoformat()} ({'/'.join(ls)})"
                             for v, ls in sorted(candidates.items()))
        return Field("cheque_date", raw_text=text,
                     parse_status=ParseStatus.UNPARSEABLE,
                     note=f"6-digit date read supports multiple non-converging "
                          f"interpretations: {readings}", **kw)
    return Field("cheque_date", raw_text=text,
                 parse_status=ParseStatus.UNPARSEABLE,
                 note="no valid calendar date among the possible 6-digit "
                     "readings", **kw)


def normalize_date(raw: str | None, prefer: str = "DMY", **kw) -> Field:
    """Handles '17 08 2026' and '15062025\\nDDM MY YYY'.

    `prefer` resolves genuinely ambiguous dates (both halves <= 12).
    Those are returned as AMBIGUOUS, not silently accepted.
    """
    text = _clean(raw)
    if not text:
        return Field("cheque_date", parse_status=ParseStatus.ABSENT,
                     raw_text=raw, **kw)

    iso = _ISO.search(text)
    if iso:
        d = _try(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        if d:
            return Field("cheque_date", value=d, raw_text=text,
                         parse_status=ParseStatus.OK, **kw)

    # written month names are unambiguous - no DD/MM guessing needed
    for pattern, order in ((_DAY_MONTH_YEAR, "dmy"), (_MONTH_DAY_YEAR, "mdy")):
        mt = pattern.search(text)
        if mt:
            if order == "dmy":
                day, mon, year = mt.group(1), mt.group(2), mt.group(3)
            else:
                mon, day, year = mt.group(1), mt.group(2), mt.group(3)
            d = _try(int(year), _MONTHS[mon.lower()], int(day))
            if d:
                return Field("cheque_date", value=d, raw_text=text,
                             parse_status=ParseStatus.OK, **kw)

    # strip pre-printed date-box guides (DD MM YYYY) before reading digits
    stripped = _GUIDE_TOKEN.sub(" ", text)
    digits = re.sub(r"\D", "", stripped)

    if len(digits) == 6:
        return _resolve_six_digit_date(digits, text, **kw)

    if len(digits) != 8:
        # Ruleset 1.10.0: 7 (or >8) digits get no repair attempt - see
        # _resolve_six_digit_date's module note for why 6 digits is the
        # one case with a second, independently-motivated hypothesis to
        # check convergence against. A 7-digit read is missing exactly one
        # digit from an assumed 8-digit grid with no such second reading:
        # every one of the 10 possible values for that missing digit
        # produces an equally structurally-valid date (day/month validity
        # essentially never depends on which digit is missing from the
        # year, except the rare Feb-29 case) - there is no way to narrow
        # it down without preferring one century as more plausible, which
        # is exactly the judgment this ruleset refuses to make.
        return Field("cheque_date", raw_text=text,
                     parse_status=ParseStatus.UNPARSEABLE,
                     note=f"expected 8 digits, got {len(digits)}", **kw)

    ymd = _try(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    dmy = _try(int(digits[4:]), int(digits[2:4]), int(digits[:2]))
    mdy = _try(int(digits[4:]), int(digits[:2]), int(digits[2:4]))

    if ymd and not (dmy or mdy):
        return Field("cheque_date", value=ymd, raw_text=text,
                     parse_status=ParseStatus.OK, **kw)
    if dmy and mdy and dmy != mdy:
        chosen = dmy if prefer == "DMY" else mdy
        other = mdy if prefer == "DMY" else dmy
        return Field("cheque_date", value=chosen, raw_text=text,
                     parse_status=ParseStatus.AMBIGUOUS,
                     note=f"could also read as {other.isoformat()}", **kw)
    resolved = dmy or mdy or ymd
    if resolved:
        return Field("cheque_date", value=resolved, raw_text=text,
                     parse_status=ParseStatus.OK, **kw)

    return Field("cheque_date", raw_text=text,
                 parse_status=ParseStatus.UNPARSEABLE,
                 note="no valid calendar date", **kw)


# ---------------------------------------------------------------------------
# signature
# ---------------------------------------------------------------------------

def normalize_signature(raw: str | None, detected: bool | None = None,
                        ambiguous_reason: str | None = None, **kw) -> Field:
    """`detected` comes from the extractor's signature verdict.

    `ambiguous_reason` is the pipeline boundary for imageprep.py's
    `OrientationIndeterminate`: when the cheque's orientation couldn't be
    confidently resolved, ANY signature reading taken from that crop is
    unreliable regardless of what a detector says. AMBIGUOUS (not ABSENT)
    because there may well be a signature-shaped mark on the page - what's
    unreliable is which crop it came from, not whether ink was seen at all.
    check_signature routes this straight to UNABLE.
    """
    if ambiguous_reason is not None:
        return Field("signature", raw_text=raw,
                     parse_status=ParseStatus.AMBIGUOUS,
                     note=ambiguous_reason, **kw)
    if detected is None:
        return Field("signature", parse_status=ParseStatus.ABSENT,
                     raw_text=raw,
                     note="extractor returned no signature verdict", **kw)
    return Field("signature", value=bool(detected), raw_text=raw,
                 parse_status=ParseStatus.OK, **kw)