"""Run the four rules over a NormalizedCheque and produce a verdict."""

from __future__ import annotations

from datetime import date

from .models import NormalizedCheque, RuleStatus, ValidationResult, Verdict
from .rules import (
    Config,
    check_amounts_match,
    check_date,
    check_memo,
    check_payee,
    check_signature,
)

RULE_SET_VERSION = "1.5.0"


def validate(cheque: NormalizedCheque, cfg: Config | None = None,
             today: date | None = None) -> ValidationResult:
    cfg = cfg or Config()
    results = [
        check_payee(cheque, cfg),
        check_amounts_match(cheque, cfg),
        check_signature(cheque, cfg),
        check_date(cheque, cfg, today=today),
        check_memo(cheque, cfg),
    ]
    # Three-valued verdict, deliberately not collapsed back to two: UNABLE
    # means "couldn't tell" (unreadable field, extraction gap, indeterminate
    # image prep) - that's not evidence the cheque is wrong, but it is NOT
    # evidence it's fine either, and reporting it as VALID (the previous
    # behaviour) means every cheque a rule couldn't assess gets waved
    # through unreviewed. A genuine FAIL (something we could read and know
    # is wrong) still invalidates outright - FAIL always wins over UNABLE.
    # Everything else, with no UNABLE anywhere, is a clean VALID.
    if any(r.status is RuleStatus.FAIL for r in results):
        verdict = Verdict.INVALID
    elif any(r.status is RuleStatus.UNABLE for r in results):
        verdict = Verdict.REVIEW
    else:
        verdict = Verdict.VALID
    return ValidationResult(verdict=verdict, rules=results, cheque=cheque)