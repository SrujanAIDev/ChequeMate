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
    # Populated only by extract.reconcile_cheques() (repeat-extraction
    # reconciliation): every individual run's reading, preserved even when
    # `value`/`raw_text` above reflect just one chosen/agreed reading (or
    # are None because the runs disagreed and nothing was safe to choose).
    # None for every ordinary single-run Field - existing callers/records
    # are unaffected.
    alternate_readings: list[dict] | None = None
    # Ruleset 1.8.0: True when `value` was reconstructed via a repair/
    # fallback transformation (e.g. normalize_amount_words discarding an
    # unreadable printed cents suffix and taking cents from a DIFFERENT
    # field instead), rather than being a clean, fully independent read of
    # THIS field's own source text. A degraded value is still a real
    # observation - strong enough to CORROBORATE another independent
    # source and produce a confident PASS - but is never strong enough on
    # its own to prove a genuine contradiction: a rule must never let a
    # degraded reading's disagreement with another source promote to
    # FAIL, only to UNABLE (see rules.check_amounts_match). This is a
    # general invariant, not a per-field special case - "no transformation
    # may convert an UNABLE into a FAIL."
    degraded: bool = False

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
    # Secondary-verification observations (TrOCR field-level cross-check),
    # keyed by field_name. Empty unless the caller opted into OCR
    # verification (see ocr_verify.py) - additive, so every existing
    # NormalizedCheque(...) call site keeps working untouched.
    ocr_verifications: dict[str, "FieldVerification"] = dc_field(default_factory=dict)


# ---------------------------------------------------------------------------
# Field-level secondary OCR verification (Document Intelligence vs TrOCR).
#
# TrOCR is a field-level secondary OBSERVER, never an equal voter and never
# an authority: it only ever sees a crop already associated with one
# specific semantic field (Document Intelligence remains the sole semantic
# locator), and it can never silently overwrite a Document Intelligence
# value. See ocr_verify.py's module docstring for the full policy.
# ---------------------------------------------------------------------------

class CropStatus(str, Enum):
    ACCEPTED = "accepted"        # crop validated and used for inference as-is
    ADJUSTED = "adjusted"        # crop validated after padding/clamping changed it
    REJECTED = "rejected"        # a candidate crop existed but failed validation
    NOT_RUN = "not_run"          # no polygon / field not eligible / DI gave nothing to crop


class TrOCRStatus(str, Enum):
    COMPLETED = "completed"      # model produced decoded text
    SKIPPED = "skipped"          # feature disabled, or trigger policy excluded this field
    NOT_RUN = "not_run"          # no usable crop (see CropStatus)
    FAILED = "failed"            # model/adapter raised (load failure, generation error, timeout)
    UNDETERMINED = "undetermined"  # model ran but produced no usable text


class ComparisonStatus(str, Enum):
    AGREE_EXACT = "AGREE_EXACT"
    AGREE_NORMALIZED = "AGREE_NORMALIZED"
    AGREE_ALLOWLIST = "AGREE_ALLOWLIST"
    DISAGREE = "DISAGREE"
    DI_ONLY = "DI_ONLY"
    TROCR_ONLY = "TROCR_ONLY"
    CROP_REJECTED = "CROP_REJECTED"
    TROCR_NOT_RUN = "TROCR_NOT_RUN"
    TROCR_FAILED = "TROCR_FAILED"
    BOTH_UNDETERMINED = "BOTH_UNDETERMINED"


@dataclass
class PrimaryObservation:
    """The Document Intelligence side of a field-level verification record.

    Never mutated once created - re-derivation (as revalidate.py does for
    the deterministic rules) produces a new FieldVerification, it does not
    edit this one in place.
    """

    engine: str = "azure_document_intelligence"
    raw_value: str | None = None
    normalized_value: str | None = None
    confidence: float | None = None
    polygon: list | None = None        # raw DI polygon points, untouched
    page_number: int | None = None
    model_id: str | None = None
    model_version: str | None = None


@dataclass
class CropResult:
    """Whether/how a DI polygon became a pixel crop actually sent to TrOCR.

    `image_reference` is an audit-safe pointer (e.g. a debug-mode file path
    under a controlled temp/debug directory), never raw image bytes and
    never embedded pixel data - see crops.py's retention policy.
    """

    status: CropStatus = CropStatus.NOT_RUN
    page_number: int | None = None
    pixel_bbox: tuple[int, int, int, int] | None = None   # (x1, y1, x2, y2), padded+clamped
    padding_pixels: dict | None = None
    validation_reasons: list[str] = dc_field(default_factory=list)
    image_reference: str | None = None


@dataclass
class SecondaryObservation:
    """The TrOCR side of a field-level verification record.

    `score` is None unless a genuinely calibrated confidence exists (it
    does not, as of this pipeline) - raw generation evidence is reported
    via `score` with `score_type="raw_generation_evidence"` ONLY when the
    report/consumer is expected to treat it as uncalibrated evidence, never
    displayed as a percentage confidence. See ocr_verify.py's confidence
    discipline section.
    """

    engine: str = "trocr"
    status: TrOCRStatus = TrOCRStatus.NOT_RUN
    raw_value: str | None = None
    normalized_value: str | None = None
    score: float | None = None
    score_type: str = "unavailable"    # "raw_generation_evidence" | "unavailable"
    model_id: str | None = None
    model_version: str | None = None
    preprocessing_variant: str | None = None
    latency_ms: float | None = None
    error_code: str | None = None


@dataclass
class ComparisonResult:
    """The transparent, non-voting comparison of primary vs. secondary.

    `selection_source` is always "document_intelligence" except for the one
    documented case (Policy D: DI undetermined, TrOCR produced a reading) -
    and even then the rules engine does not treat it as authoritative; see
    ocr_verify.py's SELECTION POLICY section for the full decision table.
    """

    status: ComparisonStatus = ComparisonStatus.TROCR_NOT_RUN
    similarity: float | None = None
    selected_display_value: str | None = None
    selection_source: str = "document_intelligence"
    manual_review_required: bool = True
    reason_codes: list[str] = dc_field(default_factory=list)


@dataclass
class FieldVerification:
    """One field's complete DI-vs-TrOCR verification record."""

    field_name: str
    primary: PrimaryObservation
    crop: CropResult
    secondary: SecondaryObservation
    comparison: ComparisonResult


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
class TokenMatchResult:
    """Ruleset 1.9.0: the result of normalize.verify_amount_by_tokens() -
    verifying a written-amount text against an ALREADY-KNOWN numeral by
    fuzzy token presence plus a scale-word guard, rather than independently
    parsing the words cold. Only used as a fallback in check_amounts_match
    when the exact word-amount parse doesn't produce a clean, matching
    value - see that function's docstring for the full decision table.

    `outcome` is one of:
      "matched"       - every expected token for some valid reading of the
                         numeral was found, and no scale word/claim in the
                         text contradicts that reading. Licenses PASS.
      "tokens_missing" - no candidate reading's tokens were all found. The
                         words may be unreadable rather than wrong; we
                         can't tell those apart. Licenses UNABLE, never FAIL.
      "contradiction"  - tokens were found, but the text also states a
                         scale word or (scale, multiplier) claim that no
                         valid reading of the numeral permits (the "$933
                         altered to look like $9,330" trap). Licenses FAIL.
    """
    outcome: str
    form_used: list[str] | None = None
    found: list[dict] = dc_field(default_factory=list)
    missing: list[str] = dc_field(default_factory=list)
    unexpected_claims: dict[str, int] = dc_field(default_factory=dict)
    # Set only when `outcome == "tokens_missing"` because of an unaccounted
    # stray digit (see normalize.verify_amount_by_tokens) rather than a
    # genuinely missing word - kept distinct so the caller/report can say
    # WHY this stayed UNABLE instead of a generic "tokens missing".
    stray_digit: str | None = None


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