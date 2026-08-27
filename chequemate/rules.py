"""The four validation rules. Each is independent and returns a RuleResult."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from dateutil.relativedelta import relativedelta

from .models import NormalizedCheque, ParseStatus, RuleResult, RuleStatus
from .normalize import canonical_payee, levenshtein


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
    require_memo: bool = True


def _unable(rule_id: str, field) -> RuleResult:
    reason = {
        ParseStatus.ABSENT: "field not returned by extractor",
        ParseStatus.UNPARSEABLE: field.note or "could not be parsed",
    }.get(field.parse_status, "unavailable")
    detail = f" (raw: {field.raw_text!r})" if field.raw_text else ""
    return RuleResult(rule_id, RuleStatus.UNABLE, f"{reason}{detail}")


# ---------------------------------------------------------------------------

# Azure's PayTo field sometimes latches onto the drawee bank's own
# letterhead/branch text instead of the handwritten payee line. That text
# never contains the town name, so it would FAIL like a genuinely wrong
# payee — but "wrong extraction" and "wrong payee" call for different staff
# action, so route it to UNABLE (needs a human look) instead of a
# confident-sounding FAIL.
_BANK_KEYWORDS = {"bank", "banking", "branch", "trust", "financial",
                  "scotiabank", "tangerine"}
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
    f = cheque.payee
    if not f.ok:
        return _unable("payee", f)

    target = _distinctive_token(cfg.expected_payee)
    tokens = f.value.split()
    distances = [levenshtein(tok, target) for tok in tokens]
    if distances and min(distances) <= cfg.payee_edit_tolerance:
        return RuleResult("payee", RuleStatus.PASS,
                          f"{f.raw_text!r} references {cfg.expected_payee!r}",
                          f.confidence)

    if _BANK_KEYWORDS & set(tokens):
        return RuleResult("payee", RuleStatus.UNABLE,
                          f"{f.raw_text!r} looks like a bank/branch name, not "
                          f"a payee — extraction likely captured the wrong "
                          f"text; confirm the payee by hand", f.confidence)

    return RuleResult("payee", RuleStatus.FAIL,
                      f"{f.raw_text!r} does not reference "
                      f"{cfg.expected_payee!r}", f.confidence)


def check_amounts_match(cheque: NormalizedCheque, cfg: Config) -> RuleResult:
    num, words = cheque.amount_numeric, cheque.amount_words
    if not num.ok:
        return _unable("amount_match", num)
    if not words.ok:
        return _unable("amount_match", words)

    delta = abs(num.value - words.value)
    if delta <= cfg.amount_tolerance:
        return RuleResult("amount_match", RuleStatus.PASS,
                          f"figures ${num.value} == words ${words.value}")

    # Bills of Exchange Act s.28(2): the written amount governs. Surface it,
    # but a mismatch still stops the cheque for human decision.
    return RuleResult("amount_match", RuleStatus.FAIL,
                      f"figures ${num.value} != words ${words.value} "
                      f"(difference ${delta}); words govern under BEA s.28(2)")


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
    if not f.ok:
        return _unable("date", f)

    today = today or date.today()
    cutoff = today - relativedelta(months=cfg.max_age_months)
    written: date = f.value
    note = f" [ambiguous: {f.note}]" if f.parse_status is ParseStatus.AMBIGUOUS else ""

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
    f = cheque.memo
    if f.ok:
        return RuleResult("memo", RuleStatus.PASS,
                          f"memo present: {f.raw_text!r}", f.confidence)
    if cfg.require_memo:
        return RuleResult("memo", RuleStatus.FAIL,
                          "memo is missing or blank — a memo is required",
                          f.confidence)
    return _unable("memo", f)


ALL_RULES = (check_payee, check_amounts_match, check_signature, check_date,
            check_memo)