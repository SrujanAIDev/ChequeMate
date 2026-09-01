"""Field-region crop validation and generation for TrOCR field verification.

TrOCR must NEVER receive the whole cheque image - only a crop that has
already been validated as belonging to one specific semantic field. This
module is the gate: `validate_and_generate_crop()` takes a DI polygon
conversion result (from geometry.py) plus the source cheque image and
either returns an ACCEPTED/ADJUSTED crop plus inference-ready image
variants, or a REJECTED/NOT_RUN result and no images at all. There is no
fallback path in this module that ever crops or returns the full image -
callers that receive a non-accepted CropResult must not invoke TrOCR.

Crops are validated, not trusted: a real, confirmed DI extraction failure
already happened once in this project (`_BANK_KEYWORDS` in rules.py, and
signature.py's whole "derive the zone from the line, don't assume it"
redesign) - DI's polygon is a locator, not a guarantee.

Whole-cheque rotation: this repo's entire real corpus is stored portrait,
~90 degrees off from upright (the same rotation imageprep.py's signature
pipeline already corrects for, confirmed by hand against every cheque in
cheques/ - see imageprep.py's module docstring). DI's polygon is defined
in that SAME un-rotated pixel space, so a straightforward crop is
geometrically correct but visually sideways. Rather than transform the
polygon coordinates through imageprep's full isolate/rotate/deskew/resize
chain (which would need to replicate deskew's `expand=True` recentring
exactly to stay pixel-accurate - fragile, easy to get subtly wrong), this
module takes the opposite, much simpler approach: crop the field region
directly from the original image using DI's own coordinates (exact, no
transform needed), THEN physically rotate that one small crop by the same
CW/CCW direction imageprep.py's detector already determined for the whole
cheque. Aspect-ratio plausibility is checked against the crop's dimensions
AFTER accounting for that rotation (width/height swap), since that is the
shape TrOCR will actually see.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .geometry import ConversionStatus, PixelPolygonResult
from .imageprep import otsu_threshold
from .models import CropResult, CropStatus

PREPROCESSING_VARIANTS = ("rgb_normalized", "grayscale", "contrast_enhanced",
                          "adaptive_binarized")


@dataclass(frozen=True)
class FieldCropConfig:
    """Per-field crop acceptance thresholds and padding. Deliberately does
    NOT default to a fixed expected-ROI rectangle for payee/amount/memo:
    this codebase already burned itself once on an eyeballed, unvalidated
    fixed rectangle (signature.py's original SIGNATURE_ZONE_FRAC, replaced
    after a confirmed false positive - see CLAUDE.md). `expected_roi_frac`
    stays None until it is actually calibrated against a labeled batch for
    this field; set it explicitly per field only once that exists.
    """

    field_name: str
    padding_frac: float = 0.15
    max_area_ratio: float = 0.35
    min_area_px: int = 400
    min_aspect_ratio: float = 0.15
    max_aspect_ratio: float = 25.0
    expected_roi_frac: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1), 0..1
    roi_min_overlap_frac: float = 0.3
    preprocessing_variants: tuple[str, ...] = ("rgb_normalized",)


def _clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int
              ) -> tuple[int, int, int, int]:
    return (max(0, min(x1, w - 1)), max(0, min(y1, h - 1)),
            max(1, min(x2, w)), max(1, min(y2, h)))


def _roi_overlap_frac(bbox: tuple[int, int, int, int], image_w: int, image_h: int,
                      roi_frac: tuple[float, float, float, float]) -> float:
    """Fraction of `bbox`'s area that falls inside `roi_frac` (0..1 image-relative)."""
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = (roi_frac[0] * image_w, roi_frac[1] * image_h,
                         roi_frac[2] * image_w, roi_frac[3] * image_h)
    ix1, iy1 = max(x1, rx1), max(y1, ry1)
    ix2, iy2 = min(x2, rx2), min(y2, ry2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(1, (x2 - x1) * (y2 - y1))
    return inter / area


def pad_to_aspect(image: Image.Image, min_aspect: float = 3.0,
                  fill: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Add neutral padding on the shorter side so a wide, short text-line
    crop isn't later squished by a naive fixed-size resize. Preserves the
    original content untouched (padding only, no scaling/cropping)."""
    w, h = image.size
    if w / h >= min_aspect:
        return image
    target_w = max(w, round(h * min_aspect))
    canvas = Image.new(image.mode, (target_w, h), fill)
    canvas.paste(image, ((target_w - w) // 2, 0))
    return canvas


def _make_variant(color_crop: Image.Image, variant: str) -> Image.Image:
    if variant == "rgb_normalized":
        return color_crop.convert("RGB")
    if variant == "grayscale":
        return ImageOps.grayscale(color_crop).convert("RGB")
    if variant == "contrast_enhanced":
        gray = ImageOps.grayscale(color_crop)
        return ImageEnhance.Contrast(gray).enhance(1.6).convert("RGB")
    if variant == "adaptive_binarized":
        gray_arr = np.asarray(ImageOps.grayscale(color_crop), dtype=np.float64)
        # Otsu's method already exists for exactly this purpose (imageprep.py,
        # calibrated to separate ink from background per-image) - reused
        # rather than re-implemented so there is one thresholding algorithm
        # in the codebase, not two.
        threshold = otsu_threshold(gray_arr)
        binary = np.where(gray_arr < threshold, 0, 255).astype(np.uint8)
        return Image.fromarray(binary).convert("RGB")
    raise ValueError(f"unknown preprocessing variant {variant!r}")


def _rotate_crop(image: Image.Image, direction: str) -> Image.Image:
    """Matches imageprep.rotate()'s exact convention (np.rot90 k=1 for
    "CCW", k=-1 for "CW") - applied to one small field crop rather than the
    whole cheque, so its text is upright before variants are generated."""
    arr = np.asarray(image)
    k = 1 if direction == "CCW" else -1
    return Image.fromarray(np.ascontiguousarray(np.rot90(arr, k=k, axes=(0, 1))))


def validate_and_generate_crop(
    image: Image.Image,
    conversion: PixelPolygonResult,
    cfg: FieldCropConfig,
    *, rotation: str | None = None,
) -> tuple[CropResult, dict[str, Image.Image] | None]:
    """Validate a DI polygon conversion and, if accepted, produce inference
    variants. Returns (CropResult, None) for anything not ACCEPTED/ADJUSTED -
    callers must treat a None second element as "do not call TrOCR".

    `rotation` is the whole-cheque CW/CCW direction already determined by
    imageprep.detect_rotation_only() (None when the source image needs no
    correction, or when that determination wasn't attempted) - see this
    module's docstring for why width/height are swapped for the aspect
    check and the crop is physically rotated afterward, rather than
    transforming DI's polygon coordinates instead."""
    reasons: list[str] = []

    if conversion.status is not ConversionStatus.OK:
        return CropResult(
            status=CropStatus.NOT_RUN,
            page_number=conversion.page_number,
            validation_reasons=[f"polygon conversion failed: "
                                f"{conversion.status.value} "
                                f"({conversion.reason or 'no reason given'})"],
        ), None

    if image.width <= 0 or image.height <= 0:
        return CropResult(
            status=CropStatus.NOT_RUN, page_number=conversion.page_number,
            validation_reasons=["source image has invalid dimensions"],
        ), None

    x1, y1, x2, y2 = conversion.pixel_bbox
    width, height = x2 - x1, y2 - y1
    area = width * height
    image_area = image.width * image.height

    if area < cfg.min_area_px:
        reasons.append(f"crop area {area}px is below the minimum "
                       f"{cfg.min_area_px}px for field {cfg.field_name!r}")
    if area / image_area > cfg.max_area_ratio:
        reasons.append(f"crop covers {area / image_area:.1%} of the cheque, "
                       f"exceeding the {cfg.max_area_ratio:.0%} limit for "
                       f"field {cfg.field_name!r}")
    eff_width, eff_height = (height, width) if rotation in ("CW", "CCW") else (width, height)
    aspect = eff_width / eff_height if eff_height else float("inf")
    if not (cfg.min_aspect_ratio <= aspect <= cfg.max_aspect_ratio):
        rot_note = f", after correcting for the cheque's {rotation} rotation" \
            if rotation in ("CW", "CCW") else ""
        reasons.append(f"aspect ratio {aspect:.2f} is implausible for field "
                       f"{cfg.field_name!r} (expected "
                       f"{cfg.min_aspect_ratio}-{cfg.max_aspect_ratio}{rot_note})")
    if cfg.expected_roi_frac is not None:
        overlap = _roi_overlap_frac((x1, y1, x2, y2), image.width, image.height,
                                    cfg.expected_roi_frac)
        if overlap < cfg.roi_min_overlap_frac:
            reasons.append(f"crop overlaps only {overlap:.0%} of the "
                           f"expected region of interest for field "
                           f"{cfg.field_name!r} (minimum "
                           f"{cfg.roi_min_overlap_frac:.0%})")

    if reasons:
        return CropResult(
            status=CropStatus.REJECTED, page_number=conversion.page_number,
            pixel_bbox=(x1, y1, x2, y2), validation_reasons=reasons,
        ), None

    pad_x = round(width * cfg.padding_frac)
    pad_y = round(height * cfg.padding_frac)
    px1, py1, px2, py2 = _clamp_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y,
                                    image.width, image.height)
    adjusted = (px1, py1, px2, py2) != (x1, y1, x2, y2)

    color_crop = image.crop((px1, py1, px2, py2))
    if rotation in ("CW", "CCW"):
        color_crop = _rotate_crop(color_crop, rotation)
    variants = {name: pad_to_aspect(_make_variant(color_crop, name))
               for name in cfg.preprocessing_variants}

    base_reason = "polygon converted and padded within bounds" if not adjusted \
        else "padding clamped to image boundary"
    if rotation in ("CW", "CCW"):
        base_reason += f"; crop rotated {rotation} to upright before inference"

    result = CropResult(
        status=CropStatus.ADJUSTED if adjusted else CropStatus.ACCEPTED,
        page_number=conversion.page_number,
        pixel_bbox=(px1, py1, px2, py2),
        padding_pixels={"x": pad_x, "y": pad_y},
        validation_reasons=[base_reason],
    )
    return result, variants


def save_debug_crop(image: Image.Image, field_name: str, record_id: str,
                    debug_dir: Path) -> str:
    """Persist a crop for audit/debug purposes only. Never called unless
    the caller has explicitly enabled debug crop retention (default off) -
    see ocr_verify.TrOCRConfig.debug_retain_crops."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    dest = debug_dir / f"{record_id}_{field_name}.png"
    image.save(dest)
    return str(dest)
