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

RULE_SET_VERSION = "1.3.0"


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
    # UNABLE means "couldn't tell" (unreadable field, extraction gap) — that's
    # not evidence the cheque is wrong, just that a human needs to look. Only
    # a genuine FAIL (something we could read and know is wrong) invalidates.
    verdict = Verdict.INVALID if any(r.status is RuleStatus.FAIL for r in results) \
        else Verdict.VALID
    return ValidationResult(verdict=verdict, rules=results, cheque=cheque)