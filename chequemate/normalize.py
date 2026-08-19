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

from .models import Field, ParseStatus

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
_CENT_WORD = re.compile(r"\bcents?\b", re.IGNORECASE)


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


def normalize_amount_words(raw: str | None, **kw) -> Field:
    """'One Hundred and twenty five dollars and 50/100 DOLLARS' -> 125.50."""
    text = _clean(raw)
    if not text:
        return Field("amount_words", parse_status=ParseStatus.ABSENT,
                     raw_text=raw, **kw)

    lowered = text.lower()
    cents: int | None = None
    m = _CENTS_FRACTION.search(lowered)
    if m:
        num = (m.group("num") or "").lower()
        cents = int(num) if num.isdigit() else 0
        lowered = lowered[: m.start()] + " " + lowered[m.end():]

    # Any digit left after the fraction is removed is contamination: the OCR
    # merged the courtesy amount into the legal line, or misread a word.
    if re.search(r"\d", lowered):
        stray = "".join(re.findall(r"\d+", lowered))
        return Field("amount_words", raw_text=text,
                     parse_status=ParseStatus.UNPARSEABLE,
                     note=f"stray digits in word amount: {stray!r}", **kw)

    tokens = re.findall(r"[a-z]+", lowered.replace("-", " "))
    if cents is None:
        tokens, cents = _split_spelled_cents(tokens)
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
    return Field("amount_words", value=value,
                 raw_text=text, parse_status=ParseStatus.OK, **kw)


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

    if len(digits) == 6:  # DDMMYY / MMDDYY
        digits = digits[:4] + ("20" if int(digits[4:]) < 70 else "19") + digits[4:]

    if len(digits) != 8:
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
                        **kw) -> Field:
    """`detected` comes from the extractor's signature verdict."""
    if detected is None:
        return Field("signature", parse_status=ParseStatus.ABSENT,
                     raw_text=raw,
                     note="extractor returned no signature verdict", **kw)
    return Field("signature", value=bool(detected), raw_text=raw,
                 parse_status=ParseStatus.OK, **kw)