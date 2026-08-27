"""Core data contracts for cheque validation.

Every extracted value carries provenance so that "absent", "unparseable" and
"parsed successfully" are never confused with one another.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any


class ParseStatus(str, Enum):
    OK = "OK"                    # parsed to a typed value
    ABSENT = "ABSENT"            # extractor returned no such field
    UNPARSEABLE = "UNPARSEABLE"  # text present, could not be normalised
    AMBIGUOUS = "AMBIGUOUS"      # parsed, but more than one reading was valid


class RuleStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNABLE = "UNABLE"  # could not evaluate (missing/unparseable input)


class Verdict(str, Enum):
    VALID = "VALID"
    REVIEW = "REVIEW"    # no rule FAILed, but at least one couldn't be evaluated
    INVALID = "INVALID"


@dataclass
class Field:
    """A single extracted field plus everything needed to audit it."""

    name: str
    value: Any = None                 # typed: Decimal | date | str | bool
    raw_text: str | None = None       # exactly what the extractor returned
    confidence: float | None = None
    bbox: list | None = None
    parse_status: ParseStatus = ParseStatus.ABSENT
    note: str | None = None           # why it is AMBIGUOUS/UNPARSEABLE

    @property
    def ok(self) -> bool:
        return self.parse_status in (ParseStatus.OK, ParseStatus.AMBIGUOUS)


@dataclass
class NormalizedCheque:
    """Typed, extractor-agnostic view of one cheque."""

    payee: Field
    amount_numeric: Field       # Decimal
    amount_words: Field         # Decimal
    cheque_date: Field          # date
    signature: Field            # bool (present / not present)
    memo: Field = dc_field(default_factory=lambda: Field(name="memo"))  # str
    source_id: str | None = None
    raw_response: dict | None = dc_field(default=None, repr=False)


@dataclass
class RuleResult:
    rule_id: str
    status: RuleStatus
    evidence: str
    confidence: float | None = None

    @property
    def blocking(self) -> bool:
        """Anything that is not an affirmative PASS blocks the cheque."""
        return self.status is not RuleStatus.PASS


@dataclass
class ValidationResult:
    verdict: Verdict
    rules: list[RuleResult]
    cheque: NormalizedCheque | None = None

    @property
    def failures(self) -> list[RuleResult]:
        return [r for r in self.rules if r.blocking]

    def summary(self) -> str:
        lines = [f"VERDICT: {self.verdict.value}"]
        for r in self.rules:
            conf = f" (conf {r.confidence:.2f})" if r.confidence is not None else ""
            lines.append(
                f"  [{r.status.value:<6}] {r.rule_id:<14} {r.evidence}{conf}")
        return "\n".join(lines)