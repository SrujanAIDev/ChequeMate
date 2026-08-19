"""Run the four rules over a NormalizedCheque and produce a verdict."""

from __future__ import annotations

from datetime import date

from .models import NormalizedCheque, ValidationResult, Verdict
from .rules import (
    Config,
    check_amounts_match,
    check_date,
    check_payee,
    check_signature,
)

RULE_SET_VERSION = "1.0.0"


def validate(cheque: NormalizedCheque, cfg: Config | None = None,
             today: date | None = None) -> ValidationResult:
    cfg = cfg or Config()
    results = [
        check_payee(cheque, cfg),
        check_amounts_match(cheque, cfg),
        check_signature(cheque, cfg),
        check_date(cheque, cfg, today=today),
    ]
    verdict = Verdict.VALID if all(r.status.value == "PASS" for r in results) \
        else Verdict.INVALID
    return ValidationResult(verdict=verdict, rules=results, cheque=cheque)