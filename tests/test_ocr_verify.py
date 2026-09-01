"""Tests for chequemate.ocr_verify: field-specific normalization, the
transparent comparison/decision engine, and end-to-end orchestration via an
injected fake TrOCR client (no torch/transformers required, no model
weights downloaded, no real cheque data used anywhere in this file).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest
from PIL import Image

from chequemate.extract import to_normalized
from chequemate.models import ComparisonStatus, CropStatus, TrOCRStatus
from chequemate.ocr_verify import (
    TrOCRVerificationConfig, compare_field, normalize_amount_words_for_comparison,
    normalize_memo_for_comparison, normalize_payee_for_comparison,
    payee_matches_expected, strip_known_payee_label, verify_cheque_fields,
)
from chequemate.trocr_adapter import TrOCRRawResult


# ---------------------------------------------------------------------------
# payee normalization
# ---------------------------------------------------------------------------

def test_payee_normalization_case_and_whitespace():
    assert normalize_payee_for_comparison("  Town   of  Whitby  ") == "town of whitby"
    assert (normalize_payee_for_comparison("TOWN OF WHITBY")
           == normalize_payee_for_comparison("town of whitby"))


def test_payee_normalization_punctuation_and_leading_article():
    assert normalize_payee_for_comparison("The Town of Whitby.") == "town of whitby"


def test_payee_label_prefix_stripped_only_when_known():
    assert strip_known_payee_label("PAY TO THE ORDER OF Town of Whitby") == "Town of Whitby"
    assert strip_known_payee_label("Pay to the order of: Town of Whitby") == "Town of Whitby"
    # not a configured prefix - left untouched, never guessed at
    assert strip_known_payee_label("Remit to Town of Whitby") == "Remit to Town of Whitby"


def test_payee_normalization_empty_and_none():
    assert normalize_payee_for_comparison(None) is None
    assert normalize_payee_for_comparison("   ") is None


def test_payee_allowlist_match_tolerates_common_ocr_misreads():
    assert payee_matches_expected("town of whitty", "Town of Whitby", tolerance=2)
    assert payee_matches_expected("whitby taxes", "Town of Whitby", tolerance=2)
    assert not payee_matches_expected("royal bank of canada", "Town of Whitby", tolerance=2)


# ---------------------------------------------------------------------------
# legal amount normalization
# ---------------------------------------------------------------------------

def test_amount_words_normalization_supported_phrase():
    value, determined = normalize_amount_words_for_comparison(
        "One Hundred and twenty five dollars and 50/100 DOLLARS")
    assert determined is True
    assert value == Decimal("125.50")


def test_amount_words_normalization_and_00_100_only_phrase():
    value, determined = normalize_amount_words_for_comparison(
        "Three hundred dollars and 00/100 only")
    assert determined is True
    assert value == Decimal("300.00")


def test_amount_words_ambiguous_returns_undetermined():
    value, determined = normalize_amount_words_for_comparison(
        "One hundred twenty blorp dollars")
    assert determined is False
    assert value is None


def test_amount_words_uses_decimal_never_float():
    value, _ = normalize_amount_words_for_comparison("ten dollars and 10/100")
    assert isinstance(value, Decimal)


# ---------------------------------------------------------------------------
# memo normalization - no invented roll-number format
# ---------------------------------------------------------------------------

def test_memo_normalization_whitespace_and_case_only():
    assert normalize_memo_for_comparison("  roll  num = 1049368 ") == "ROLL NUM = 1049368"


def test_memo_normalization_does_not_reformat_or_infer():
    raw = "Property Tax - 100936318"
    assert normalize_memo_for_comparison(raw) == raw.upper()


def test_memo_normalization_empty():
    assert normalize_memo_for_comparison("") is None
    assert normalize_memo_for_comparison(None) is None


# ---------------------------------------------------------------------------
# comparison engine
# ---------------------------------------------------------------------------

def _base_kwargs(**overrides):
    kwargs = dict(field_name="payee", di_ok=True, di_raw="Town of Whitby",
                 di_normalized="town of whitby", crop_status=CropStatus.ACCEPTED,
                 trocr_status=TrOCRStatus.COMPLETED, trocr_raw="Town of Whitby",
                 trocr_normalized="town of whitby")
    kwargs.update(overrides)
    return kwargs


def test_comparison_agree_exact():
    result = compare_field(**_base_kwargs())
    assert result.status is ComparisonStatus.AGREE_EXACT
    assert result.manual_review_required is False
    assert result.selection_source == "document_intelligence"


def test_comparison_agree_normalized_when_case_differs():
    result = compare_field(**_base_kwargs(trocr_raw="TOWN OF WHITBY",
                                          trocr_normalized="town of whitby"))
    assert result.status is ComparisonStatus.AGREE_NORMALIZED
    assert result.manual_review_required is False


def test_comparison_agree_allowlist_when_both_hit_allowlist_but_differ():
    result = compare_field(**_base_kwargs(
        di_raw="Whitby Taxes", di_normalized="whitby taxes",
        trocr_raw="Town of Whitby", trocr_normalized="town of whitby",
        allowlist_expected="Town of Whitby"))
    assert result.status is ComparisonStatus.AGREE_ALLOWLIST
    assert "ALLOWLIST_SPELLING_DIFFERENCE" in result.reason_codes
    assert result.manual_review_required is False


def test_comparison_disagree_preserves_both_and_requires_review():
    result = compare_field(**_base_kwargs(
        di_raw="Royal Bank of Canada", di_normalized="royal bank of canada",
        trocr_raw="Some Other Text", trocr_normalized="some other text",
        allowlist_expected="Town of Whitby"))
    assert result.status is ComparisonStatus.DISAGREE
    assert result.manual_review_required is True
    assert "OCR_DISAGREEMENT" in result.reason_codes
    # the selected display value must still be DI's own, never swapped
    assert result.selected_display_value == "Royal Bank of Canada"


def test_comparison_di_only_when_trocr_skipped():
    result = compare_field(**_base_kwargs(
        crop_status=CropStatus.NOT_RUN, trocr_status=TrOCRStatus.SKIPPED,
        trocr_raw=None, trocr_normalized=None))
    assert result.status is ComparisonStatus.DI_ONLY
    assert result.manual_review_required is False
    assert result.selected_display_value == "Town of Whitby"


def test_comparison_trocr_only_when_di_undetermined():
    result = compare_field(**_base_kwargs(
        di_ok=False, di_raw=None, di_normalized=None))
    assert result.status is ComparisonStatus.TROCR_ONLY
    assert result.manual_review_required is True
    assert "TROCR_UNVERIFIED_SUGGESTION" in result.reason_codes
    # never promoted to the authoritative display value
    assert result.selected_display_value is None
    assert result.selection_source == "document_intelligence"


def test_comparison_crop_rejected_di_still_stands():
    result = compare_field(**_base_kwargs(
        crop_status=CropStatus.REJECTED, trocr_status=TrOCRStatus.NOT_RUN,
        trocr_raw=None, trocr_normalized=None))
    assert result.status is ComparisonStatus.CROP_REJECTED
    assert result.manual_review_required is False  # DI is fine on its own
    assert result.selected_display_value == "Town of Whitby"


def test_comparison_trocr_not_run_when_no_polygon():
    result = compare_field(**_base_kwargs(
        crop_status=CropStatus.NOT_RUN, trocr_status=TrOCRStatus.NOT_RUN,
        trocr_raw=None, trocr_normalized=None))
    assert result.status is ComparisonStatus.TROCR_NOT_RUN
    assert result.manual_review_required is False


def test_comparison_trocr_failed_does_not_penalize_di_value():
    result = compare_field(**_base_kwargs(
        trocr_status=TrOCRStatus.FAILED, trocr_raw=None, trocr_normalized=None))
    assert result.status is ComparisonStatus.TROCR_FAILED
    assert result.manual_review_required is False
    assert result.selected_display_value == "Town of Whitby"


def test_comparison_both_undetermined_forces_review():
    result = compare_field(**_base_kwargs(
        di_ok=False, di_raw=None, di_normalized=None,
        trocr_status=TrOCRStatus.UNDETERMINED, trocr_raw=None, trocr_normalized=None))
    assert result.status is ComparisonStatus.BOTH_UNDETERMINED
    assert result.manual_review_required is True
    assert result.selected_display_value is None


def test_comparison_never_silently_swaps_di_for_trocr_on_disagreement():
    result = compare_field(**_base_kwargs(
        di_raw="Original DI Value", di_normalized="original di value",
        trocr_raw="Different TrOCR Value", trocr_normalized="different trocr value"))
    assert result.selected_display_value == "Original DI Value"
    assert result.selection_source == "document_intelligence"


# ---------------------------------------------------------------------------
# uncalibrated score handling (fake client controls this directly - see below)
# ---------------------------------------------------------------------------

@dataclass
class _FakeClient:
    text_by_field: dict
    score: float | None = None
    score_type: str = "unavailable"
    seen_image_sizes: list = None

    def __post_init__(self):
        if self.seen_image_sizes is None:
            self.seen_image_sizes = []

    def generate(self, image, *, field_name):
        self.seen_image_sizes.append(image.size)
        text = self.text_by_field.get(field_name)
        return TrOCRRawResult(
            status=TrOCRStatus.COMPLETED if text else TrOCRStatus.UNDETERMINED,
            raw_text=text, sequence_score=self.score, score_type=self.score_type,
            error_code=None, latency_ms=1.0, model_id="fake-trocr",
            model_version="test", device="cpu")


def _azure_doc(payee_polygon=(120, 300, 900, 300, 900, 400, 120, 400),
              amount_words_polygon=(120, 500, 900, 500, 900, 560, 120, 560),
              memo_polygon=(120, 600, 500, 600, 500, 650, 120, 650),
              payee_text="Town of Whitby", amount_words_text="Two Hundred dollars and 00/100",
              memo_text="Roll number 12345"):
    fields = {}
    if payee_text is not None:
        fields["PayTo"] = {"valueString": payee_text, "confidence": 0.6,
                           "boundingRegions": [{"pageNumber": 1,
                                               "polygon": list(payee_polygon)}]}
    if amount_words_text is not None:
        fields["WordAmount"] = {"content": amount_words_text, "confidence": 0.5,
                               "boundingRegions": [{"pageNumber": 1,
                                                   "polygon": list(amount_words_polygon)}]}
    if memo_text is not None:
        fields["Memo"] = {"valueString": memo_text, "confidence": 0.8,
                         "boundingRegions": [{"pageNumber": 1,
                                             "polygon": list(memo_polygon)}]}
    return {
        "documents": [{"fields": fields}],
        "pages": [{"pageNumber": 1, "width": 1860, "height": 780, "unit": "pixel"}],
    }


def _synthetic_image() -> Image.Image:
    return Image.new("RGB", (1860, 780), (255, 255, 255))


def test_verify_cheque_fields_disabled_returns_empty_dict():
    raw = _azure_doc()
    cheque = to_normalized(raw["documents"][0])
    cfg = TrOCRVerificationConfig(enabled=False)
    result = verify_cheque_fields(raw, _synthetic_image(), cheque, cfg)
    assert result == {}


def test_verify_cheque_fields_end_to_end_agreement():
    raw = _azure_doc()
    cheque = to_normalized(raw["documents"][0])
    fake = _FakeClient(text_by_field={"payee": "Town of Whitby",
                                      "amount_words": "Two Hundred dollars and 00/100",
                                      "memo": "Roll number 12345"})
    cfg = TrOCRVerificationConfig(enabled=True, trigger_policy="always_on_eligible_fields")
    result = verify_cheque_fields(raw, _synthetic_image(), cheque, cfg, trocr_client=fake)

    assert set(result) == {"payee", "amount_words", "memo"}
    assert result["payee"].comparison.status is ComparisonStatus.AGREE_EXACT
    assert result["amount_words"].comparison.status is ComparisonStatus.AGREE_EXACT
    assert result["memo"].comparison.status is ComparisonStatus.AGREE_EXACT


def test_verify_cheque_fields_disagreement_preserves_both_values():
    raw = _azure_doc()
    cheque = to_normalized(raw["documents"][0])
    fake = _FakeClient(text_by_field={"payee": "Completely Different Text"})
    cfg = TrOCRVerificationConfig(enabled=True, eligible_fields=("payee",))
    result = verify_cheque_fields(raw, _synthetic_image(), cheque, cfg, trocr_client=fake)

    fv = result["payee"]
    assert fv.comparison.status is ComparisonStatus.DISAGREE
    assert fv.primary.raw_value == "Town of Whitby"
    assert fv.secondary.raw_value == "Completely Different Text"
    assert fv.comparison.manual_review_required is True


def test_verify_cheque_fields_uncalibrated_score_never_labeled_confidence():
    raw = _azure_doc()
    cheque = to_normalized(raw["documents"][0])
    fake = _FakeClient(text_by_field={"payee": "Town of Whitby"},
                       score=0.9123, score_type="raw_generation_evidence")
    cfg = TrOCRVerificationConfig(enabled=True, eligible_fields=("payee",))
    result = verify_cheque_fields(raw, _synthetic_image(), cheque, cfg, trocr_client=fake)

    secondary = result["payee"].secondary
    assert secondary.score == pytest.approx(0.9123)
    assert secondary.score_type == "raw_generation_evidence"
    assert secondary.score_type != "confidence"


def test_verify_cheque_fields_missing_score_reports_unavailable_not_zero():
    raw = _azure_doc()
    cheque = to_normalized(raw["documents"][0])
    fake = _FakeClient(text_by_field={"payee": "Town of Whitby"},
                       score=None, score_type="unavailable")
    cfg = TrOCRVerificationConfig(enabled=True, eligible_fields=("payee",))
    result = verify_cheque_fields(raw, _synthetic_image(), cheque, cfg, trocr_client=fake)

    secondary = result["payee"].secondary
    assert secondary.score is None
    assert secondary.score_type == "unavailable"


def test_verify_cheque_fields_never_sends_full_cheque_dimensions_to_trocr():
    """Explicit regression guard: TrOCR must only ever receive a field crop,
    never the full cheque image, for a field-level verification request."""
    raw = _azure_doc()
    cheque = to_normalized(raw["documents"][0])
    fake = _FakeClient(text_by_field={"payee": "Town of Whitby",
                                      "amount_words": "Two Hundred dollars and 00/100",
                                      "memo": "Roll number 12345"})
    cfg = TrOCRVerificationConfig(enabled=True)
    full_image = _synthetic_image()
    full_size = full_image.size

    verify_cheque_fields(raw, full_image, cheque, cfg, trocr_client=fake)

    assert len(fake.seen_image_sizes) == 3
    for size in fake.seen_image_sizes:
        assert size != full_size, (
            f"TrOCR received an image sized {size}, identical to the full "
            f"cheque canvas {full_size} - field-level crops must never "
            f"match the whole-cheque dimensions")


def test_verify_cheque_fields_does_not_mutate_original_cheque_fields():
    raw = _azure_doc()
    cheque = to_normalized(raw["documents"][0])
    original_payee_value = cheque.payee.value
    original_amount_value = cheque.amount_words.value
    fake = _FakeClient(text_by_field={"payee": "Something Else Entirely"})
    cfg = TrOCRVerificationConfig(enabled=True, eligible_fields=("payee",))

    verify_cheque_fields(raw, _synthetic_image(), cheque, cfg, trocr_client=fake)

    # TrOCR disagreement must never alter the DI-derived Field objects that
    # rules.py/validate.py already consumed.
    assert cheque.payee.value == original_payee_value
    assert cheque.amount_words.value == original_amount_value


def test_verify_cheque_fields_di_undetermined_trocr_suggestion_not_promoted():
    raw = _azure_doc(payee_text=None)  # PayTo entirely absent from Azure's response
    cheque = to_normalized(raw["documents"][0])
    assert cheque.payee.ok is False

    fake = _FakeClient(text_by_field={})  # payee has no polygon at all -> no crop possible
    cfg = TrOCRVerificationConfig(enabled=True, eligible_fields=("payee",))
    result = verify_cheque_fields(raw, _synthetic_image(), cheque, cfg, trocr_client=fake)

    fv = result["payee"]
    assert fv.comparison.status in (ComparisonStatus.TROCR_NOT_RUN,
                                    ComparisonStatus.BOTH_UNDETERMINED)
    assert fv.comparison.selected_display_value is None


# ---------------------------------------------------------------------------
# whole-cheque rotation awareness - this repo's entire real corpus is
# stored portrait/un-rotated in the same pixel space DI's polygons use
# (see crops.py's and ocr_verify.resolve_rotation's docstrings).
# ---------------------------------------------------------------------------

from chequemate import ocr_verify as _ocr_verify_module  # noqa: E402
from chequemate.imageprep import NoChequeFound, OrientationIndeterminate, RotationDecision  # noqa: E402
from chequemate.ocr_verify import resolve_rotation  # noqa: E402


def _landscape_image() -> Image.Image:
    return Image.new("RGB", (1860, 780), (255, 255, 255))


def _portrait_image() -> Image.Image:
    return Image.new("RGB", (780, 1860), (255, 255, 255))


def test_resolve_rotation_none_when_no_source_path():
    direction, refusal = resolve_rotation(None, _portrait_image())
    assert direction is None
    assert refusal is None


def test_resolve_rotation_none_for_landscape_image_even_with_path():
    direction, refusal = resolve_rotation("some/path.png", _landscape_image())
    assert direction is None
    assert refusal is None


def test_resolve_rotation_returns_direction_for_portrait_image(monkeypatch):
    monkeypatch.setattr(
        _ocr_verify_module, "detect_rotation_only",
        lambda path: RotationDecision(
            direction="CCW", fundamental_score=0.5, harmonic_score=0.3,
            fundamental_lag_px=10, dpi=300.0, dpi_source="jfif", confident=True))
    direction, refusal = resolve_rotation("some/path.png", _portrait_image())
    assert direction == "CCW"
    assert refusal is None


def test_resolve_rotation_refuses_rather_than_guess_on_indeterminate(monkeypatch):
    def _raise(path):
        raise OrientationIndeterminate("neither orientation cleared the floor",
                                       top_score=0.1, bottom_score=0.1,
                                       dpi=300.0, dpi_source="jfif")
    monkeypatch.setattr(_ocr_verify_module, "detect_rotation_only", _raise)
    direction, refusal = resolve_rotation("some/path.png", _portrait_image())
    assert direction is None
    assert refusal is not None
    assert "orientation" in refusal


def test_resolve_rotation_refuses_on_no_cheque_found(monkeypatch):
    def _raise(path):
        raise NoChequeFound("no ink cleared the isolation floor")
    monkeypatch.setattr(_ocr_verify_module, "detect_rotation_only", _raise)
    direction, refusal = resolve_rotation("some/path.png", _portrait_image())
    assert direction is None
    assert refusal is not None


def test_verify_cheque_fields_applies_rotation_correction_end_to_end(monkeypatch):
    """A field whose raw polygon is tall/narrow (as every real cheque in
    this repo's corpus produces, since they're all stored portrait) must
    be correctly recovered once rotation is resolved and passed through -
    not rejected as an implausible aspect ratio."""
    monkeypatch.setattr(
        _ocr_verify_module, "detect_rotation_only",
        lambda path: RotationDecision(
            direction="CCW", fundamental_score=0.5, harmonic_score=0.3,
            fundamental_lag_px=10, dpi=300.0, dpi_source="jfif", confident=True))

    portrait_w, portrait_h = 600, 900
    raw = {
        "documents": [{"fields": {
            "PayTo": {"valueString": "Town of Whitby", "confidence": 0.6,
                     "boundingRegions": [{"pageNumber": 1,
                                         # tall/narrow in raw (portrait) space -
                                         # exactly this corpus's real shape.
                                         "polygon": [300, 100, 340, 100,
                                                   340, 500, 300, 500]}]},
        }}],
        "pages": [{"pageNumber": 1, "width": portrait_w, "height": portrait_h,
                  "unit": "pixel"}],
    }
    cheque = to_normalized(raw["documents"][0])
    fake = _FakeClient(text_by_field={"payee": "Town of Whitby"})
    cfg = TrOCRVerificationConfig(enabled=True, eligible_fields=("payee",))
    image = Image.new("RGB", (portrait_w, portrait_h), (255, 255, 255))

    result = verify_cheque_fields(raw, image, cheque, cfg, trocr_client=fake,
                                  source_path="fake/path.png")

    fv = result["payee"]
    assert fv.crop.status.value in ("accepted", "adjusted")
    assert fv.comparison.status is ComparisonStatus.AGREE_EXACT


def test_verify_cheque_fields_without_source_path_keeps_old_behaviour(monkeypatch):
    """Backward compatibility: omitting source_path must behave exactly as
    it did before this fix (no rotation correction attempted at all), so
    the same tall/narrow polygon is correctly rejected without a path."""
    portrait_w, portrait_h = 600, 900
    raw = {
        "documents": [{"fields": {
            "PayTo": {"valueString": "Town of Whitby", "confidence": 0.6,
                     "boundingRegions": [{"pageNumber": 1,
                                         "polygon": [300, 100, 340, 100,
                                                   340, 500, 300, 500]}]},
        }}],
        "pages": [{"pageNumber": 1, "width": portrait_w, "height": portrait_h,
                  "unit": "pixel"}],
    }
    cheque = to_normalized(raw["documents"][0])
    fake = _FakeClient(text_by_field={"payee": "Town of Whitby"})
    cfg = TrOCRVerificationConfig(enabled=True, eligible_fields=("payee",))
    image = Image.new("RGB", (portrait_w, portrait_h), (255, 255, 255))

    result = verify_cheque_fields(raw, image, cheque, cfg, trocr_client=fake)

    assert result["payee"].crop.status.value == "rejected"


def test_resolve_rotation_override_skips_detector_entirely(monkeypatch):
    def _fail(path):
        raise AssertionError("detector must not be called when an override is given")
    monkeypatch.setattr(_ocr_verify_module, "detect_rotation_only", _fail)
    direction, refusal = resolve_rotation("some/path.png", _portrait_image(),
                                          rotation_override="CCW")
    assert direction == "CCW"
    assert refusal is None


def test_resolve_rotation_override_rejects_invalid_direction():
    with pytest.raises(ValueError):
        resolve_rotation("some/path.png", _portrait_image(), rotation_override="sideways")


def test_verify_cheque_fields_rotation_override_recovers_a_previously_refused_file(monkeypatch):
    """Mirrors this repo's real _0001/_0006/_0011 files: the detector
    refuses (OrientationIndeterminate), but a standing operator
    confirmation (scripts/apply_rotation_override.py) already resolved the
    direction - that confirmation must be usable directly, without ever
    re-invoking (and re-refusing via) the detector."""
    def _raise(path):
        raise OrientationIndeterminate("neither orientation cleared the floor",
                                       top_score=0.1, bottom_score=0.1,
                                       dpi=300.0, dpi_source="jfif")
    monkeypatch.setattr(_ocr_verify_module, "detect_rotation_only", _raise)

    portrait_w, portrait_h = 600, 900
    raw = {
        "documents": [{"fields": {
            "PayTo": {"valueString": "Town of Whitby", "confidence": 0.6,
                     "boundingRegions": [{"pageNumber": 1,
                                         "polygon": [300, 100, 340, 100,
                                                   340, 500, 300, 500]}]},
        }}],
        "pages": [{"pageNumber": 1, "width": portrait_w, "height": portrait_h,
                  "unit": "pixel"}],
    }
    cheque = to_normalized(raw["documents"][0])
    fake = _FakeClient(text_by_field={"payee": "Town of Whitby"})
    cfg = TrOCRVerificationConfig(enabled=True, eligible_fields=("payee",))
    image = Image.new("RGB", (portrait_w, portrait_h), (255, 255, 255))

    result = verify_cheque_fields(raw, image, cheque, cfg, trocr_client=fake,
                                  source_path="fake/path.png", rotation_override="CCW")

    fv = result["payee"]
    assert fv.crop.status.value in ("accepted", "adjusted")
    assert fv.comparison.status is ComparisonStatus.AGREE_EXACT


def test_verify_cheque_fields_rotation_refusal_blocks_all_fields_not_silent_guess(monkeypatch):
    def _raise(path):
        raise OrientationIndeterminate("neither orientation cleared the floor",
                                       top_score=0.1, bottom_score=0.1,
                                       dpi=300.0, dpi_source="jfif")
    monkeypatch.setattr(_ocr_verify_module, "detect_rotation_only", _raise)

    portrait_w, portrait_h = 600, 900
    raw = {
        "documents": [{"fields": {
            "PayTo": {"valueString": "Town of Whitby", "confidence": 0.6,
                     "boundingRegions": [{"pageNumber": 1,
                                         "polygon": [300, 100, 340, 100,
                                                   340, 500, 300, 500]}]},
        }}],
        "pages": [{"pageNumber": 1, "width": portrait_w, "height": portrait_h,
                  "unit": "pixel"}],
    }
    cheque = to_normalized(raw["documents"][0])
    fake = _FakeClient(text_by_field={"payee": "Town of Whitby"})
    cfg = TrOCRVerificationConfig(enabled=True, eligible_fields=("payee",))
    image = Image.new("RGB", (portrait_w, portrait_h), (255, 255, 255))

    result = verify_cheque_fields(raw, image, cheque, cfg, trocr_client=fake,
                                  source_path="fake/path.png")

    fv = result["payee"]
    # never silently falls back to unrotated geometry and guesses - the
    # fake client must never even have been called for this field.
    assert len(fake.seen_image_sizes) == 0
    assert fv.crop.status.value == "not_run"
    assert "orientation" in fv.crop.validation_reasons[0]
