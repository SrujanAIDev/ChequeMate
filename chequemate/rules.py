"""The four validation rules. Each is independent and returns a RuleResult."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from dateutil.relativedelta import relativedelta

from .models import NormalizedCheque, ParseStatus, RuleResult, RuleStatus
from .normalize import canonical_payee, levenshtein, verify_amount_by_tokens


@dataclass(frozen=True)
class Config:
    expected_payee: str = "Town of Whitby"
    # Per-token edit-distance tolerance against the expected payee's
    # distinctive word (e.g. 'whitby'). 2 absorbs the common OCR misreads
    # seen on real cheques (whitty, whitley, white, whit...); real cheques
    # phrase the payee many ways ('Whitby Taxes', 'Town of Whitby', a
    # drawer's own name printed next to it) so this checks token-by-token
    # rather than the whole string.
    payee_edit_tolerance: int = 2
    amount_tolerance: Decimal = Decimal("0.00")
    max_age_months: int = 6
    reject_postdated: bool = False
    # As of ruleset 1.6.0 this no longer changes check_memo's RuleStatus:
    # a blank/absent memo always routes to UNABLE, never FAIL - see
    # check_memo's docstring for why "the field was present but empty" was
    # judged not to be a trustworthy-enough signal of a genuinely
    # human-confirmed blank line to keep a hard FAIL on. Kept in Config
    # (rather than removed) so a future genuine confirmed-blank signal
    # (e.g. a visual-verification-style human check) has a place to plug
    # into without another schema change, and so `Config(require_memo=...)`
    # doesn't become an error for any existing caller.
    require_memo: bool = True


def _unable(rule_id: str, field) -> RuleResult:
    reason = {
        ParseStatus.ABSENT: field.note or "field not returned by extractor",
        ParseStatus.UNPARSEABLE: field.note or "could not be parsed",
        ParseStatus.AMBIGUOUS: field.note or "ambiguous - could not determine a reliable value",
    }.get(field.parse_status, "unavailable")
    detail = f" (raw: {field.raw_text!r})" if field.raw_text else ""
    return RuleResult(rule_id, RuleStatus.UNABLE, f"{reason}{detail}")


def _disagreement_ambiguous(field) -> bool:
    """True when a field's AMBIGUOUS status came from repeat-extraction
    disagreement (extract.reconcile_cheques - chequemate.extract.analyze_multi
    / to_normalized_multi), not from a field's own internal ambiguity (e.g.
    check_date's DMY/MDY case, which always carries a real `value`).
    Discriminated by the absence of a value: reconciliation deliberately
    never fabricates one to choose between disagreeing runs, so `value is
    None` is a reliable, non-overloaded signal that nothing here should be
    trusted as-is - every rule must route this to UNABLE, not attempt to
    read `.value`."""
    return field.parse_status is ParseStatus.AMBIGUOUS and field.value is None


# ---------------------------------------------------------------------------

_GENERIC_PAYEE_WORDS = {"town", "city", "of", "the", "corporation",
                        "municipality", "regional", "township"}


def _distinctive_token(expected_payee: str) -> str:
    """The one word in the expected payee that actually identifies it —
    'whitby' out of 'Town of Whitby' — so matching isn't thrown off by
    generic municipal boilerplate ('Town of', 'Corporation of the', ...)."""
    tokens = canonical_payee(expected_payee).split()
    candidates = [t for t in tokens if t not in _GENERIC_PAYEE_WORDS] or tokens
    return max(candidates, key=len)


def check_payee(cheque: NormalizedCheque, cfg: Config) -> RuleResult:
    """Ruleset 1.7.0: presence is sufficient. If any token in the extracted
    text is within `payee_edit_tolerance` of the expected payee's
    distinctive word, this PASSes - extra text (a bank's own letterhead
    merged in by a bad extraction, a drawer's name, 'Taxes') is ignored,
    not treated as evidence of anything. Confirmed policy change: earlier
    rulesets special-cased bank/branch letterhead in the extracted text
    (_BANK_KEYWORDS) as UNABLE rather than FAIL, on the theory that it was
    probably a missed extraction rather than a genuinely wrong payee. That
    carve-out is removed - text containing zero reference to the expected
    payee, bank letterhead or not, is now a plain FAIL like any other wrong
    payee; the only thing that changes the outcome is whether the
    distinctive token is present, not what kind of text surrounds it."""
    f = cheque.payee
    if _disagreement_ambiguous(f):
        return RuleResult("payee", RuleStatus.UNABLE,
                          f"repeat-extraction runs disagreed on the payee: "
                          f"{f.note} — confirm by hand", f.confidence)
    if not f.ok:
        return _unable("payee", f)

    target = _distinctive_token(cfg.expected_payee)
    tokens = f.value.split()
    distances = [levenshtein(tok, target) for tok in tokens]
    if distances and min(distances) <= cfg.payee_edit_tolerance:
        return RuleResult("payee", RuleStatus.PASS,
                          f"{f.raw_text!r} references {cfg.expected_payee!r}",
                          f.confidence)

    return RuleResult("payee", RuleStatus.FAIL,
                      f"{f.raw_text!r} does not reference "
                      f"{cfg.expected_payee!r}", f.confidence)


def _token_match_result(num_value: Decimal, words_raw: str | None) -> RuleResult | None:
    """Ruleset 1.9.0 fallback: verify the written amount against the
    numeral we already have by fuzzy token presence + a scale-word guard
    (normalize.verify_amount_by_tokens), instead of independently parsing
    the words cold. Only called once the exact parse has already failed to
    produce a clean, matching value - see check_amounts_match's docstring.
    Returns None (never a "no evidence either way" RuleResult) when the
    fallback found nothing new (tokens_missing) - the caller keeps
    whatever its own existing behaviour for that case already is."""
    numeral_cents = int((num_value * 100) % 100)
    result = verify_amount_by_tokens(words_raw, int(num_value), numeral_cents)
    if result.outcome == "matched":
        found_desc = ", ".join(f"{f['expected']!r}" +
                               (f" (as {f['matched_text']!r}, distance {f['distance']})"
                                if f["distance"] else "")
                               for f in result.found)
        return RuleResult(
            "amount_match", RuleStatus.PASS,
            f"figures ${num_value} == words — resolved by token-match: "
            f"every word the numeral implies ({found_desc}) is present in "
            f"the written amount, and no conflicting scale word is stated "
            f"(the exact written-amount parse did not independently "
            f"succeed and agree)")
    if result.outcome == "contradiction":
        claims = ", ".join(f"{mult} {scale}" for scale, mult in result.unexpected_claims.items())
        return RuleResult(
            "amount_match", RuleStatus.FAIL,
            f"numeric amount reads ${num_value} but the written amount "
            f"states {claims}, which no valid reading of ${num_value} "
            f"permits — words govern under BEA s.28(2)")
    return None  # tokens_missing: no new evidence, caller's existing path stands


def check_amounts_match(cheque: NormalizedCheque, cfg: Config) -> RuleResult:
    """Numeral vs. written amount. Ruleset 1.9.0 decision order, primary
    path first, token-match strictly as a fallback behind it (never
    instead of it - see normalize.verify_amount_by_tokens' module docstring
    for the mechanism):

      1. Exact parse produces a value that matches the numeral (clean or
         degraded/corroborated) -> PASS. Unchanged from 1.8.0 - this is
         still the primary path, tried first, and a clean match here never
         even reaches token-match.
      2. Exact parse produced no value, OR produced one that disagrees ->
         try normalize.verify_amount_by_tokens() against the SAME raw
         text:
           - "matched" (every token the numeral implies is present, no
             contradicting scale word) -> PASS, resolved-by-token-match.
           - "contradiction" (a scale word/claim the numeral doesn't
             permit) -> FAIL, a genuine contradiction on its own evidence.
           - "tokens_missing" -> no new evidence; falls through to 3.
      3. Neither the exact parse nor token-match produced a confirmed
         reading: a degraded value's disagreement stays UNABLE (the 1.8.0
         invariant - a repair can never be promoted to FAIL), a clean
         parse's disagreement is FAIL (BEA s.28(2)), and no value at all
         is UNABLE with the numeric figure stated plainly.
    """
    num, words = cheque.amount_numeric, cheque.amount_words
    if _disagreement_ambiguous(num):
        return RuleResult("amount_match", RuleStatus.UNABLE,
                          f"repeat-extraction runs disagreed on the numeric "
                          f"amount: {num.note} — confirm by hand", num.confidence)
    if not num.ok:
        return _unable("amount_match", num)

    if _disagreement_ambiguous(words):
        return RuleResult(
            "amount_match", RuleStatus.UNABLE,
            f"numeric amount reads ${num.value} but repeat-extraction runs "
            f"disagreed on the written amount ({words.note}) — the written "
            f"amount could not be confirmed; do not rely on the numeric "
            f"figure alone", words.confidence)
    if not words.ok:
        # Ruleset 1.9.0: before giving up, try verifying the written text
        # against the numeral by token-match (see _token_match_result) -
        # this is exactly the population where the exact parse has nothing
        # to offer, so it is the primary beneficiary of this fallback.
        token_result = _token_match_result(num.value, words.raw_text)
        if token_result is not None:
            return token_result
        # Numeric parsed cleanly - surface that explicitly rather than only
        # describing the words failure in isolation, so a reviewer sees the
        # numeric figure immediately instead of having to look it up
        # separately. Still UNABLE, never PASS: words govern under BEA
        # s.28(2) and could not be confirmed, so the numeric alone is never
        # treated as sufficient on its own.
        reason = {
            ParseStatus.ABSENT: "written amount field not returned by extractor",
            ParseStatus.UNPARSEABLE: words.note or "written amount could not be parsed",
        }.get(words.parse_status, "written amount unavailable")
        return RuleResult(
            "amount_match", RuleStatus.UNABLE,
            f"numeric amount reads ${num.value} but the written amount could "
            f"not be read ({reason}; raw: {words.raw_text!r}) — confirm the "
            f"written amount by hand before relying on the numeric figure "
            f"alone", words.confidence)

    # Ruleset 1.8.0: corroboration, not failure-counting. The numeral and
    # the written amount are two INDEPENDENT statements of the same fact -
    # when they agree, that agreement is stronger evidence than either
    # reading alone, precisely because OCR noise on the two is uncorrelated
    # while the true amount isn't. That holds even when one reading needed
    # repair to get there (words.degraded - see normalize_amount_words and
    # Field.degraded's docstring): a degraded reading converging with an
    # independent clean one is still real corroboration, so it PASSes, with
    # its fallback provenance stated plainly rather than hidden.
    #
    # Disagreement is the asymmetric case. Two CLEAN readings disagreeing is
    # a genuine contradiction - FAIL, as always, words governing under BEA
    # s.28(2). But a DEGRADED reading disagreeing with the numeral is NOT
    # equivalent evidence: the repair already means this wasn't a fully
    # independent read, so its disagreement doesn't prove a wrong cheque,
    # only that the two don't corroborate. That must resolve to UNABLE
    # (insufficient evidence to decide either way), never FAIL - a general
    # invariant, not specific to any one record: no transformation may
    # convert what would otherwise be UNABLE into a confident FAIL. Record
    # 0010 of the 2026-08-24 batch ('one thousand nine hundred -eight-one
    # 4/kg 100 Dollars', a real cheque) is the case that surfaced this: the
    # cents-suffix repair correctly discarded '4/kg', but the remaining
    # words text still silently mis-parses a hyphenated compound number
    # ('-eight-one' for 'eighty-one') to the wrong dollar figure - a
    # pre-existing, unrelated gap in _words_to_int, not something this rule
    # can detect directly. Without this asymmetry, that latent bug would
    # assert a confident, false BEA mismatch instead of asking a human to
    # look, exactly the "extraction gap stated as fact" failure mode this
    # codebase has already fixed elsewhere (bank-letterhead payee,
    # SIGNATURE_ZONE_FRAC - see CLAUDE.md).
    degraded = num.degraded or words.degraded

    delta = abs(num.value - words.value)
    if delta <= cfg.amount_tolerance:
        if degraded:
            return RuleResult(
                "amount_match", RuleStatus.PASS,
                f"figures ${num.value} == words ${words.value} — resolved "
                f"by corroboration: the written amount's printed cents "
                f"suffix could not be read as printed, but the dollar "
                f"figure independently written out in words agrees with "
                f"the numeral ({words.note})")
        return RuleResult("amount_match", RuleStatus.PASS,
                          f"figures ${num.value} == words ${words.value}")

    # Ruleset 1.9.0: the exact-parsed value disagrees with the numeral -
    # before falling back to the existing degraded/clean handling below,
    # give token-match a chance at the SAME raw text. This is what
    # resolves cases where the exact parser produced a value at all only
    # because it silently mis-parsed something (record 0010's "-eight-one"
    # -> 1909.46 instead of 1981.46) - token-match works from the raw text
    # directly, not from the exact parser's possibly-wrong result, so it
    # isn't blocked by the same mistake. Applies regardless of `degraded`:
    # token-match's own PASS/FAIL licensing (see verify_amount_by_tokens'
    # docstring) is independent, evidence-based justification, not a
    # promotion of the degraded value - it earns its own verdict.
    token_result = _token_match_result(num.value, words.raw_text)
    if token_result is not None:
        return token_result

    if degraded:
        return RuleResult(
            "amount_match", RuleStatus.UNABLE,
            f"numeric amount reads ${num.value} but the written amount "
            f"(${words.value}, only reached after repairing an unreadable "
            f"cents suffix) does not corroborate it — a repaired reading "
            f"disagreeing with the numeral is insufficient evidence of a "
            f"genuine mismatch, not proof of one; confirm the written "
            f"amount by hand")

    # Bills of Exchange Act s.28(2): the written amount governs. Surface it,
    # but a mismatch still stops the cheque for human decision. Only
    # reachable here when the exact parse was clean AND token-match found
    # no rescuing evidence either - a genuine contradiction.
    return RuleResult("amount_match", RuleStatus.FAIL,
                      f"figures ${num.value} != words ${words.value} "
                      f"(difference ${delta}); words govern under BEA "
                      f"s.28(2)")


def check_signature(cheque: NormalizedCheque, cfg: Config) -> RuleResult:
    f = cheque.signature
    if f.parse_status is ParseStatus.AMBIGUOUS:
        # Orientation-indeterminate source crop (imageprep.OrientationIndeterminate)
        # - a signature reading taken from an unknown orientation can't be
        # trusted either way, so this must route to UNABLE, never PASS/FAIL.
        return RuleResult("signature", RuleStatus.UNABLE,
                          f"signature verdict unreliable: "
                          f"{f.note or 'ambiguous source image orientation'}",
                          f.confidence)
    if not f.ok:
        return _unable("signature", f)
    if f.value:
        return RuleResult("signature", RuleStatus.PASS,
                          "signature detected in signature region", f.confidence)
    return RuleResult("signature", RuleStatus.FAIL,
                      "no signature detected", f.confidence)


def check_date(cheque: NormalizedCheque, cfg: Config,
               today: date | None = None) -> RuleResult:
    f = cheque.cheque_date
    if _disagreement_ambiguous(f):
        return RuleResult("date", RuleStatus.UNABLE,
                          f"repeat-extraction runs disagreed on the date: "
                          f"{f.note} — confirm by hand", f.confidence)
    if not f.ok:
        return _unable("date", f)

    today = today or date.today()
    cutoff = today - relativedelta(months=cfg.max_age_months)
    written: date = f.value
    # Ruleset 1.10.0: surface the note whenever one is present, not just for
    # AMBIGUOUS - a 6-digit date resolved via converging interpretations
    # (normalize._resolve_six_digit_date) is ParseStatus.OK, not AMBIGUOUS
    # (only one value survived, there's nothing left to disclose two
    # readings of), but its provenance - a repaired read, not a clean one -
    # must still be visible in the report, same principle as amount_match's
    # corroboration/token-match notes.
    if f.parse_status is ParseStatus.AMBIGUOUS:
        note = f" [ambiguous: {f.note}]"
    elif f.note:
        note = f" [{f.note}]"
    else:
        note = ""

    if written < cutoff:
        age = relativedelta(today, written)
        return RuleResult("date", RuleStatus.FAIL,
                          f"{written.isoformat()} is stale-dated "
                          f"({age.years * 12 + age.months} months old, "
                          f"limit {cfg.max_age_months}){note}", f.confidence)

    if written > today:
        if cfg.reject_postdated:
            return RuleResult("date", RuleStatus.FAIL,
                              f"{written.isoformat()} is post-dated "
                              f"(today {today.isoformat()}){note}", f.confidence)
        return RuleResult("date", RuleStatus.PASS,
                          f"{written.isoformat()} is post-dated "
                          f"(today {today.isoformat()}) — accepted; route to "
                          f"post-dated batch handling{note}", f.confidence)

    return RuleResult("date", RuleStatus.PASS,
                      f"{written.isoformat()} is within "
                      f"{cfg.max_age_months} months{note}", f.confidence)


def check_memo(cheque: NormalizedCheque, cfg: Config) -> RuleResult:
    """Ruleset 1.6.0: a blank/absent memo is UNABLE (-> REVIEW), never FAIL
    (-> INVALID), regardless of `cfg.require_memo`.

    This reverses 1.1.0's original design, which treated any blank memo as
    a hard business FAIL. Investigation (prompted by a real cheque that
    was INVALID purely on a missing memo) found every checkable case in
    this project's real corpus was Azure's `Memo` key being entirely
    absent from the response - a genuine extraction gap, not a confirmed
    blank line. `extract.normalize_memo` CAN technically distinguish "key
    absent" from "key present but empty" (see MEMO_KEY_ABSENT_NOTE and the
    `field_present` parameter, preserved in `f.note` for audit purposes),
    but "key present, no text" is not itself trustworthy evidence of a
    genuinely blank memo line either - Azure could just as easily have
    looked at the wrong region and reported nothing, the same failure
    mode `_BANK_KEYWORDS` above exists to catch for payee. Treating that
    weak signal as FAIL-worthy would repeat exactly the "confident wrong
    assertion from an extraction gap" bug this project has already found
    and fixed more than once (bank-letterhead payee, the original
    SIGNATURE_ZONE_FRAC false positive). So both cases route to UNABLE:
    the cheque still is not silently waved through (REVIEW still requires
    a human look), it just no longer asserts INVALID for what may well be
    Azure's own extraction miss.
    """
    f = cheque.memo
    if _disagreement_ambiguous(f):
        return RuleResult("memo", RuleStatus.UNABLE,
                          f"repeat-extraction runs disagreed on the memo: "
                          f"{f.note} — confirm by hand", f.confidence)
    if f.ok:
        return RuleResult("memo", RuleStatus.PASS,
                          f"memo present: {f.raw_text!r}", f.confidence)
    return _unable("memo", f)


ALL_RULES = (check_payee, check_amounts_match, check_signature, check_date,
            check_memo)