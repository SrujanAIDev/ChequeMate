"""Centralized Document Intelligence polygon/bounding-region -> pixel-space
conversion. Every consumer of a DI polygon (crops.py's crop generation, any
future region-based detector) goes through this module - coordinate math is
not duplicated anywhere else.

Handles the actual azure-ai-documentintelligence response shape found in
this repo (confirmed against a real captured response, raw/*.json):

    document["fields"]["PayTo"]["boundingRegions"] ==
        [{"pageNumber": 1, "polygon": [x1, y1, x2, y2, x3, y3, x4, y4]}]
    response["pages"] ==
        [{"pageNumber": 1, "width": 319, "height": 545, "unit": "pixel"}]

The SDK's `.as_dict()` form uses camelCase keys; some call sites in this
repo also tolerate snake_case (extract.py's `_meta()` does the same
dual-key check), so this module does too.

Coordinate model: DI reports polygon coordinates in the SAME unit as the
page's declared width/height (either "pixel" or "inch" - Azure's only two
documented units for prebuilt models). Rather than assuming the page
dimensions equal the source image's pixel dimensions (they may not, if the
image handed to this function was resized after DI analyzed the original),
this module always computes an explicit proportional scale factor from
declared page space into actual source-image pixel space:

    scale_x = image_width_px / page_width
    scale_y = image_height_px / page_height
    pixel_x = round(polygon_x * scale_x)

This one formula is correct for "pixel" units when image dimensions match
the analyzed page (scale == 1.0), and is also the correct conversion for
"inch" units. It deliberately never assumes coordinates are already pixels.

Never raises for malformed/out-of-range input - every failure mode returns
a structured PixelPolygonResult with a ConversionStatus and a human-
readable reason, so a bad polygon can never propagate as an uncontrolled
exception into the crop or TrOCR pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any

SUPPORTED_UNITS = {"pixel", "inch"}


class ConversionStatus(str, Enum):
    OK = "ok"
    NO_POLYGON = "no_polygon"                    # no bounding region for this field at all
    WRONG_PAGE = "wrong_page"                     # regions exist, none for the expected page
    MISSING_PAGE_INFO = "missing_page_info"       # no page entry / no width-height for that page
    UNSUPPORTED_UNIT = "unsupported_unit"         # unit is not "pixel" or "inch"
    MALFORMED_POLYGON = "malformed_polygon"       # wrong point count / non-finite coordinates
    INVALID_IMAGE_DIMENSIONS = "invalid_image_dimensions"  # image_width_px/height_px <= 0
    ZERO_AREA = "zero_area"                       # resulting pixel bbox has no area


@dataclass
class PixelPolygonResult:
    status: ConversionStatus
    page_number: int | None = None
    pixel_bbox: tuple[int, int, int, int] | None = None       # (x1, y1, x2, y2), clamped, x1<x2, y1<y2
    pixel_polygon: list[tuple[int, int]] | None = None
    reason: str | None = None
    clamped: bool = False

    @property
    def ok(self) -> bool:
        return self.status is ConversionStatus.OK


def _get(d: dict, *keys: str) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _as_dict(region: Any) -> dict:
    if isinstance(region, dict):
        return region
    if hasattr(region, "as_dict"):
        return region.as_dict()
    return {}


def _finite_numbers(values: Any) -> list[float] | None:
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        return None
    out: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(f):
            return None
        out.append(f)
    return out


def _find_page(pages: list[dict] | None, page_number: int) -> dict | None:
    for p in pages or []:
        p = _as_dict(p)
        if _get(p, "pageNumber", "page_number") == page_number:
            return p
    return None


def convert_bounding_region_to_pixels(
    bounding_regions: list | None,
    *,
    image_width_px: int,
    image_height_px: int,
    pages: list[dict] | None,
    expected_page_number: int = 1,
) -> PixelPolygonResult:
    """Convert one field's DI boundingRegions into a pixel bbox + polygon.

    `bounding_regions` is exactly what `field["boundingRegions"]` (or
    `field.bounding_regions`) contains - a list with normally one entry per
    page the field's text spans. `pages` is the analysis response's
    top-level `pages` array (needed for width/height/unit - a single
    boundingRegion never carries that itself).
    """
    if image_width_px <= 0 or image_height_px <= 0:
        return PixelPolygonResult(
            ConversionStatus.INVALID_IMAGE_DIMENSIONS,
            reason=f"image dimensions must be positive, got "
                   f"{image_width_px}x{image_height_px}")

    if not bounding_regions:
        return PixelPolygonResult(ConversionStatus.NO_POLYGON,
                                  reason="field has no boundingRegions")

    region = None
    for r in bounding_regions:
        rd = _as_dict(r)
        if _get(rd, "pageNumber", "page_number") == expected_page_number:
            region = rd
            break
    if region is None:
        found = sorted({_get(_as_dict(r), "pageNumber", "page_number")
                       for r in bounding_regions})
        return PixelPolygonResult(
            ConversionStatus.WRONG_PAGE,
            reason=f"no boundingRegion for page {expected_page_number} "
                   f"(found pages: {found})")

    page = _find_page(pages, expected_page_number)
    if page is None:
        return PixelPolygonResult(
            ConversionStatus.MISSING_PAGE_INFO,
            page_number=expected_page_number,
            reason=f"no page entry for page {expected_page_number} in "
                   f"the analysis response's pages[]")

    page_width = _get(page, "width")
    page_height = _get(page, "height")
    unit = _get(page, "unit")
    try:
        page_width, page_height = float(page_width), float(page_height)
    except (TypeError, ValueError):
        return PixelPolygonResult(
            ConversionStatus.MISSING_PAGE_INFO, page_number=expected_page_number,
            reason=f"page {expected_page_number} has non-numeric width/height")
    if page_width <= 0 or page_height <= 0:
        return PixelPolygonResult(
            ConversionStatus.MISSING_PAGE_INFO, page_number=expected_page_number,
            reason=f"page {expected_page_number} has non-positive "
                   f"width/height ({page_width}x{page_height})")

    if unit not in SUPPORTED_UNITS:
        return PixelPolygonResult(
            ConversionStatus.UNSUPPORTED_UNIT, page_number=expected_page_number,
            reason=f"unsupported page unit {unit!r} (supported: "
                   f"{sorted(SUPPORTED_UNITS)})")

    raw_polygon = _get(region, "polygon")
    coords = _finite_numbers(raw_polygon)
    if coords is None or len(coords) < 6 or len(coords) % 2 != 0:
        return PixelPolygonResult(
            ConversionStatus.MALFORMED_POLYGON, page_number=expected_page_number,
            reason=f"polygon must be a flat list of >=3 finite (x, y) pairs, "
                   f"got {raw_polygon!r}")

    scale_x = image_width_px / page_width
    scale_y = image_height_px / page_height

    clamped = False
    pixel_polygon: list[tuple[int, int]] = []
    for i in range(0, len(coords), 2):
        px = coords[i] * scale_x
        py = coords[i + 1] * scale_y
        cx = min(max(px, 0.0), image_width_px - 1)
        cy = min(max(py, 0.0), image_height_px - 1)
        if cx != px or cy != py:
            clamped = True
        pixel_polygon.append((round(cx), round(cy)))

    xs = [p[0] for p in pixel_polygon]
    ys = [p[1] for p in pixel_polygon]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return PixelPolygonResult(
            ConversionStatus.ZERO_AREA, page_number=expected_page_number,
            pixel_polygon=pixel_polygon,
            reason=f"pixel bbox has zero area after conversion/clamping "
                   f"({x1}, {y1}, {x2}, {y2})")

    return PixelPolygonResult(
        ConversionStatus.OK, page_number=expected_page_number,
        pixel_bbox=(x1, y1, x2, y2), pixel_polygon=pixel_polygon,
        clamped=clamped)
