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
    # 0 = exact match required. Raise to 1-2 only if OCR noise proves to be
    # the dominant failure mode on your golden set.
    payee_edit_tolerance: int = 0
    amount_tolerance: Decimal = Decimal("0.00")
    max_age_months: int = 6
    reject_postdated: bool = True
    require_memo: bool = True


def _unable(rule_id: str, field) -> RuleResult:
    reason = {
        ParseStatus.ABSENT: "field not returned by extractor",
        ParseStatus.UNPARSEABLE: field.note or "could not be parsed",
    }.get(field.parse_status, "unavailable")
    detail = f" (raw: {field.raw_text!r})" if field.raw_text else ""
    return RuleResult(rule_id, RuleStatus.UNABLE, f"{reason}{detail}")


# ---------------------------------------------------------------------------

def check_payee(cheque: NormalizedCheque, cfg: Config) -> RuleResult:
    f = cheque.payee
    if not f.ok:
        return _unable("payee", f)

    expected = canonical_payee(cfg.expected_payee)
    actual = f.value
    if actual == expected:
        return RuleResult("payee", RuleStatus.PASS,
                          f"{f.raw_text!r} matches {cfg.expected_payee!r}",
                          f.confidence)

    distance = levenshtein(actual, expected)
    if distance <= cfg.payee_edit_tolerance:
        return RuleResult("payee", RuleStatus.PASS,
                          f"{f.raw_text!r} within tolerance "
                          f"(edit distance {distance})", f.confidence)

    kind = "misspelled" if distance <= 3 else "different payee"
    return RuleResult("payee", RuleStatus.FAIL,
                      f"{f.raw_text!r} != {cfg.expected_payee!r} — {kind} "
                      f"(edit distance {distance})", f.confidence)


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

    if cfg.reject_postdated and written > today:
        return RuleResult("date", RuleStatus.FAIL,
                          f"{written.isoformat()} is post-dated "
                          f"(today {today.isoformat()}){note}", f.confidence)

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