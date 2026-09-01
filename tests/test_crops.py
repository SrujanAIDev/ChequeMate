"""Tests for chequemate.crops's crop validation/generation. All images are
synthetic, generated in-memory with PIL - no real cheque data anywhere in
this file, so this suite runs identically with or without cheques/ present.
"""

from __future__ import annotations

from PIL import Image

from chequemate.crops import FieldCropConfig, pad_to_aspect, validate_and_generate_crop
from chequemate.geometry import ConversionStatus, PixelPolygonResult
from chequemate.models import CropStatus

CANVAS_W, CANVAS_H = 1860, 780  # matches imageprep.py's normalized canvas


def _blank_canvas() -> Image.Image:
    return Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))


def _ok_conversion(bbox: tuple[int, int, int, int]) -> PixelPolygonResult:
    x1, y1, x2, y2 = bbox
    return PixelPolygonResult(
        ConversionStatus.OK, page_number=1, pixel_bbox=bbox,
        pixel_polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


# ---------------------------------------------------------------------------
# valid crop
# ---------------------------------------------------------------------------

def test_valid_payee_crop_is_accepted():
    image = _blank_canvas()
    conversion = _ok_conversion((200, 300, 700, 380))  # plausible text-line box
    cfg = FieldCropConfig(field_name="payee")
    result, variants = validate_and_generate_crop(image, conversion, cfg)
    assert result.status in (CropStatus.ACCEPTED, CropStatus.ADJUSTED)
    assert variants is not None
    assert "rgb_normalized" in variants
    assert variants["rgb_normalized"].size[0] > 0


def test_accepted_crop_records_padding_and_bbox():
    image = _blank_canvas()
    conversion = _ok_conversion((200, 300, 700, 380))
    cfg = FieldCropConfig(field_name="payee", padding_frac=0.1)
    result, variants = validate_and_generate_crop(image, conversion, cfg)
    assert result.pixel_bbox is not None
    assert result.padding_pixels is not None
    # padded box must be at least as large as the raw polygon box
    px1, py1, px2, py2 = result.pixel_bbox
    assert px1 <= 200 and py1 <= 300 and px2 >= 700 and py2 >= 380


# ---------------------------------------------------------------------------
# rejection: too much of the cheque
# ---------------------------------------------------------------------------

def test_crop_covering_most_of_cheque_is_rejected():
    image = _blank_canvas()
    conversion = _ok_conversion((0, 0, CANVAS_W, CANVAS_H))  # 100% of the image
    cfg = FieldCropConfig(field_name="payee", max_area_ratio=0.35)
    result, variants = validate_and_generate_crop(image, conversion, cfg)
    assert result.status is CropStatus.REJECTED
    assert variants is None
    assert any("exceeds" in r or "covers" in r for r in result.validation_reasons)


# ---------------------------------------------------------------------------
# ROI overlap
# ---------------------------------------------------------------------------

def test_crop_overlapping_expected_roi_is_accepted():
    image = _blank_canvas()
    # ROI covering the left half of the cheque; crop sits inside it.
    roi = (0.0, 0.0, 0.5, 1.0)
    conversion = _ok_conversion((100, 300, 400, 380))
    cfg = FieldCropConfig(field_name="payee", expected_roi_frac=roi,
                          roi_min_overlap_frac=0.5)
    result, variants = validate_and_generate_crop(image, conversion, cfg)
    assert result.status in (CropStatus.ACCEPTED, CropStatus.ADJUSTED)
    assert variants is not None


def test_crop_far_outside_expected_roi_is_rejected():
    image = _blank_canvas()
    roi = (0.0, 0.0, 0.3, 1.0)  # left 30% of the cheque only
    conversion = _ok_conversion((1500, 300, 1800, 380))  # far right
    cfg = FieldCropConfig(field_name="payee", expected_roi_frac=roi,
                          roi_min_overlap_frac=0.5)
    result, variants = validate_and_generate_crop(image, conversion, cfg)
    assert result.status is CropStatus.REJECTED
    assert variants is None
    assert any("region of interest" in r for r in result.validation_reasons)


def test_crop_next_to_bank_name_region_rejected_when_roi_configured():
    # Simulates the exact confirmed failure mode from signature.py's own
    # history (a field landing on adjacent letterhead/bank text instead of
    # its own line) - ROI enforcement is the guard against the equivalent
    # mistake for payee/amount crops, when a real ROI has been configured.
    image = _blank_canvas()
    payee_roi = (0.05, 0.35, 0.55, 0.55)
    bank_name_bbox = (1400, 50, 1800, 120)  # top-right corner, nowhere near payee_roi
    conversion = _ok_conversion(bank_name_bbox)
    cfg = FieldCropConfig(field_name="payee", expected_roi_frac=payee_roi,
                          roi_min_overlap_frac=0.5)
    result, variants = validate_and_generate_crop(image, conversion, cfg)
    assert result.status is CropStatus.REJECTED
    assert variants is None


# ---------------------------------------------------------------------------
# rejection: tiny crop
# ---------------------------------------------------------------------------

def test_tiny_crop_is_rejected():
    image = _blank_canvas()
    conversion = _ok_conversion((100, 100, 110, 108))  # 10x8 = 80px^2
    cfg = FieldCropConfig(field_name="payee", min_area_px=400)
    result, variants = validate_and_generate_crop(image, conversion, cfg)
    assert result.status is CropStatus.REJECTED
    assert variants is None
    assert any("below the minimum" in r for r in result.validation_reasons)


# ---------------------------------------------------------------------------
# excessive padding clamps rather than escaping the image
# ---------------------------------------------------------------------------

def test_excessive_padding_is_clamped_to_image_bounds():
    image = _blank_canvas()
    conversion = _ok_conversion((5, 5, 60, 40))  # near the top-left corner
    cfg = FieldCropConfig(field_name="payee", padding_frac=5.0)  # huge padding
    result, variants = validate_and_generate_crop(image, conversion, cfg)
    assert result.status is CropStatus.ADJUSTED
    x1, y1, x2, y2 = result.pixel_bbox
    assert x1 >= 0 and y1 >= 0
    assert x2 <= CANVAS_W and y2 <= CANVAS_H
    assert "padding clamped" in result.validation_reasons[0]


# ---------------------------------------------------------------------------
# implausible aspect ratio
# ---------------------------------------------------------------------------

def test_implausible_aspect_ratio_rejected():
    image = _blank_canvas()
    conversion = _ok_conversion((100, 100, 108, 500))  # very tall, narrow sliver
    cfg = FieldCropConfig(field_name="payee", min_aspect_ratio=0.15, max_aspect_ratio=25.0)
    result, variants = validate_and_generate_crop(image, conversion, cfg)
    assert result.status is CropStatus.REJECTED
    assert variants is None


# ---------------------------------------------------------------------------
# invalid source image
# ---------------------------------------------------------------------------

def test_invalid_source_image_dimensions():
    # PIL forbids constructing a genuinely 0-size image, so the guard is
    # exercised via geometry.py's own INVALID_IMAGE_DIMENSIONS conversion
    # status feeding into crops.py's NOT_RUN handling instead (crops.py
    # never re-reads image.width/height as anything other than what PIL
    # already reports for a real, already-opened image).
    tiny = Image.new("RGB", (1, 1))
    conversion = PixelPolygonResult(
        ConversionStatus.INVALID_IMAGE_DIMENSIONS,
        reason="image dimensions must be positive, got 0x0")
    cfg = FieldCropConfig(field_name="payee", min_area_px=1)
    result, variants = validate_and_generate_crop(tiny, conversion, cfg)
    assert result.status is CropStatus.NOT_RUN
    assert variants is None


def test_not_run_when_conversion_failed():
    image = _blank_canvas()
    conversion = PixelPolygonResult(ConversionStatus.NO_POLYGON, reason="no polygon")
    cfg = FieldCropConfig(field_name="payee")
    result, variants = validate_and_generate_crop(image, conversion, cfg)
    assert result.status is CropStatus.NOT_RUN
    assert variants is None


# ---------------------------------------------------------------------------
# never falls back to the full image
# ---------------------------------------------------------------------------

def test_rejected_crop_never_returns_full_image_variants():
    image = _blank_canvas()
    conversion = _ok_conversion((0, 0, CANVAS_W, CANVAS_H))
    cfg = FieldCropConfig(field_name="payee", max_area_ratio=0.35)
    result, variants = validate_and_generate_crop(image, conversion, cfg)
    assert variants is None  # never a full-cheque fallback


# ---------------------------------------------------------------------------
# preprocessing variants
# ---------------------------------------------------------------------------

def test_multiple_preprocessing_variants_all_generated():
    image = _blank_canvas()
    conversion = _ok_conversion((200, 300, 700, 380))
    cfg = FieldCropConfig(field_name="payee",
                          preprocessing_variants=("rgb_normalized", "grayscale",
                                                  "contrast_enhanced",
                                                  "adaptive_binarized"))
    result, variants = validate_and_generate_crop(image, conversion, cfg)
    assert set(variants.keys()) == {"rgb_normalized", "grayscale",
                                    "contrast_enhanced", "adaptive_binarized"}
    for img in variants.values():
        assert img.mode == "RGB"


def test_pad_to_aspect_preserves_content_and_widens_short_ratio():
    narrow = Image.new("RGB", (400, 200), (0, 0, 0))
    padded = pad_to_aspect(narrow, min_aspect=3.0)
    assert padded.size[0] / padded.size[1] >= 3.0
    assert padded.size[1] == 200


def test_pad_to_aspect_leaves_already_wide_image_untouched():
    wide = Image.new("RGB", (900, 100), (0, 0, 0))
    padded = pad_to_aspect(wide, min_aspect=3.0)
    assert padded.size == wide.size


# ---------------------------------------------------------------------------
# whole-cheque rotation awareness (this repo's entire real corpus is stored
# portrait/un-rotated - see crops.py's module docstring)
# ---------------------------------------------------------------------------

def test_rotation_none_behaves_exactly_as_before():
    image = _blank_canvas()
    conversion = _ok_conversion((200, 300, 700, 380))  # wide box, upright-plausible
    cfg = FieldCropConfig(field_name="payee")
    result, variants = validate_and_generate_crop(image, conversion, cfg, rotation=None)
    assert result.status in (CropStatus.ACCEPTED, CropStatus.ADJUSTED)
    assert variants is not None


def test_rotation_aware_aspect_check_accepts_a_raw_tall_box():
    # A box that is tall/narrow in RAW (un-rotated) pixel space is exactly
    # what a normal wide text line looks like once the whole cheque's known
    # 90-degree rotation is accounted for - this must be ACCEPTED once
    # `rotation` is supplied, even though the same raw box would fail the
    # plain (unrotated) aspect check.
    image = _blank_canvas()
    conversion = _ok_conversion((300, 100, 340, 500))  # width=40, height=400 -> aspect 0.10
    cfg = FieldCropConfig(field_name="payee")  # min_aspect_ratio=0.15 by default

    without_rotation, variants_none = validate_and_generate_crop(image, conversion, cfg)
    assert without_rotation.status is CropStatus.REJECTED
    assert variants_none is None

    with_rotation, variants = validate_and_generate_crop(
        image, conversion, cfg, rotation="CCW")
    assert with_rotation.status in (CropStatus.ACCEPTED, CropStatus.ADJUSTED)
    assert variants is not None


def test_rotation_aware_crop_is_physically_rotated_to_upright():
    # Paint a horizontal black bar into the RAW (portrait) image at a tall,
    # narrow box location; after the field crop is corrected for a 90-degree
    # rotation, the resulting crop's own aspect ratio must be wide, not tall
    # - confirming the pixels were actually rotated, not just relabeled.
    from PIL import ImageDraw
    image = Image.new("RGB", (600, 900), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([300, 100, 340, 500], fill=(0, 0, 0))  # tall black bar
    conversion = _ok_conversion((300, 100, 340, 500))
    cfg = FieldCropConfig(field_name="payee", padding_frac=0.0)

    result, variants = validate_and_generate_crop(image, conversion, cfg, rotation="CCW")
    assert result.status in (CropStatus.ACCEPTED, CropStatus.ADJUSTED)
    crop_img = variants["rgb_normalized"]
    # after physical rotation, width must exceed height (was 40x400 raw)
    assert crop_img.size[0] > crop_img.size[1]


def test_rotation_cw_and_ccw_produce_different_orientations():
    from PIL import ImageDraw
    image = Image.new("RGB", (600, 900), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    # an asymmetric mark (not a uniform fill) - a symmetric solid rectangle
    # would look identical after either 90-degree rotation, defeating the
    # point of this test.
    draw.rectangle([300, 100, 340, 500], fill=(0, 0, 0))
    draw.rectangle([300, 100, 340, 140], fill=(255, 0, 0))  # red mark near the top only
    conversion = _ok_conversion((300, 100, 340, 500))
    cfg = FieldCropConfig(field_name="payee", padding_frac=0.0)

    _, variants_ccw = validate_and_generate_crop(image, conversion, cfg, rotation="CCW")
    _, variants_cw = validate_and_generate_crop(image, conversion, cfg, rotation="CW")
    import numpy as np
    arr_ccw = np.asarray(variants_ccw["rgb_normalized"])
    arr_cw = np.asarray(variants_cw["rgb_normalized"])
    # same size (both are 90-degree rotations of the same source crop) but
    # not pixel-identical - CW and CCW are opposite rotations.
    assert arr_ccw.shape == arr_cw.shape
    assert not np.array_equal(arr_ccw, arr_cw)


def test_rotation_validation_reason_notes_correction():
    image = _blank_canvas()
    conversion = _ok_conversion((300, 100, 340, 500))
    cfg = FieldCropConfig(field_name="payee")
    result, _ = validate_and_generate_crop(image, conversion, cfg, rotation="CCW")
    assert any("rotated CCW" in r for r in result.validation_reasons)
