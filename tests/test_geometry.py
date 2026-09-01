"""Tests for chequemate.geometry's centralized polygon-to-pixel conversion.

All fixtures are synthetic dicts shaped like the real
azure-ai-documentintelligence response (confirmed against a real captured
raw/*.json in this repo: pageNumber/polygon/width/height/unit keys, flat
[x1,y1,x2,y2,...] polygon lists) - no real cheque data is used anywhere in
this file.
"""

from __future__ import annotations

import math

from chequemate.geometry import ConversionStatus, convert_bounding_region_to_pixels

PAGE_PIXEL = [{"pageNumber": 1, "width": 300, "height": 600, "unit": "pixel"}]
PAGE_INCH = [{"pageNumber": 1, "width": 3.0, "height": 6.0, "unit": "inch"}]


def _regions(polygon, page=1):
    return [{"pageNumber": page, "polygon": polygon}]


# ---------------------------------------------------------------------------
# pixel units
# ---------------------------------------------------------------------------

def test_pixel_units_1to1_when_image_matches_page():
    result = convert_bounding_region_to_pixels(
        _regions([10, 20, 110, 20, 110, 70, 10, 70]),
        image_width_px=300, image_height_px=600, pages=PAGE_PIXEL)
    assert result.ok
    assert result.pixel_bbox == (10, 20, 110, 70)
    assert result.clamped is False


def test_pixel_units_scaled_when_image_differs_from_page():
    # page declared 300x600 but the actual image handed in is 2x that -
    # coordinates must scale proportionally, never assumed to be raw pixels.
    result = convert_bounding_region_to_pixels(
        _regions([10, 20, 110, 20, 110, 70, 10, 70]),
        image_width_px=600, image_height_px=1200, pages=PAGE_PIXEL)
    assert result.ok
    assert result.pixel_bbox == (20, 40, 220, 140)


# ---------------------------------------------------------------------------
# inch units
# ---------------------------------------------------------------------------

def test_inch_units_convert_using_dpi_implied_by_image_size():
    # 3x6 inch page, image is 300x600px -> 100 DPI; a 1x1 inch box at
    # (0.1,0.1)-(1.1,1.1) inches should land at (10,10)-(110,110) pixels.
    result = convert_bounding_region_to_pixels(
        _regions([0.1, 0.1, 1.1, 0.1, 1.1, 1.1, 0.1, 1.1]),
        image_width_px=300, image_height_px=600, pages=PAGE_INCH)
    assert result.ok
    assert result.pixel_bbox == (10, 10, 110, 110)


# ---------------------------------------------------------------------------
# out-of-bounds / clamping / rounding
# ---------------------------------------------------------------------------

def test_out_of_bounds_coordinates_are_clamped():
    result = convert_bounding_region_to_pixels(
        _regions([-50, -50, 400, -50, 400, 700, -50, 700]),
        image_width_px=300, image_height_px=600, pages=PAGE_PIXEL)
    assert result.ok
    assert result.clamped is True
    x1, y1, x2, y2 = result.pixel_bbox
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 300 and y2 <= 600


def test_negative_values_clamp_to_zero_not_rejected():
    result = convert_bounding_region_to_pixels(
        _regions([-10, -10, 50, -10, 50, 50, -10, 50]),
        image_width_px=300, image_height_px=600, pages=PAGE_PIXEL)
    assert result.ok
    assert result.pixel_bbox[0] == 0
    assert result.pixel_bbox[1] == 0


def test_rounding_produces_integers():
    result = convert_bounding_region_to_pixels(
        _regions([10.4, 20.6, 110.4, 20.6, 110.4, 70.6, 10.4, 70.6]),
        image_width_px=300, image_height_px=600, pages=PAGE_PIXEL)
    assert result.ok
    for coord in result.pixel_bbox:
        assert isinstance(coord, int)


# ---------------------------------------------------------------------------
# zero-area
# ---------------------------------------------------------------------------

def test_zero_area_box_is_rejected():
    result = convert_bounding_region_to_pixels(
        _regions([10, 20, 10, 20, 10, 20, 10, 20]),
        image_width_px=300, image_height_px=600, pages=PAGE_PIXEL)
    assert result.status is ConversionStatus.ZERO_AREA
    assert not result.ok


def test_zero_area_after_clamping_is_rejected():
    # entirely outside the image on one axis -> clamps to a zero-width slice
    result = convert_bounding_region_to_pixels(
        _regions([-100, 10, -50, 10, -50, 50, -100, 50]),
        image_width_px=300, image_height_px=600, pages=PAGE_PIXEL)
    assert result.status is ConversionStatus.ZERO_AREA


# ---------------------------------------------------------------------------
# missing / malformed input
# ---------------------------------------------------------------------------

def test_no_polygon_when_bounding_regions_is_none():
    result = convert_bounding_region_to_pixels(
        None, image_width_px=300, image_height_px=600, pages=PAGE_PIXEL)
    assert result.status is ConversionStatus.NO_POLYGON


def test_no_polygon_when_bounding_regions_is_empty_list():
    result = convert_bounding_region_to_pixels(
        [], image_width_px=300, image_height_px=600, pages=PAGE_PIXEL)
    assert result.status is ConversionStatus.NO_POLYGON


def test_malformed_polygon_odd_coordinate_count():
    result = convert_bounding_region_to_pixels(
        _regions([10, 20, 110, 20, 110]),
        image_width_px=300, image_height_px=600, pages=PAGE_PIXEL)
    assert result.status is ConversionStatus.MALFORMED_POLYGON


def test_malformed_polygon_too_few_points():
    result = convert_bounding_region_to_pixels(
        _regions([10, 20, 110, 20]),
        image_width_px=300, image_height_px=600, pages=PAGE_PIXEL)
    assert result.status is ConversionStatus.MALFORMED_POLYGON


def test_malformed_polygon_non_numeric():
    result = convert_bounding_region_to_pixels(
        _regions([10, 20, "bad", 20, 110, 70, 10, 70]),
        image_width_px=300, image_height_px=600, pages=PAGE_PIXEL)
    assert result.status is ConversionStatus.MALFORMED_POLYGON


def test_malformed_polygon_nan_and_infinite_rejected():
    for bad_value in (math.nan, math.inf, -math.inf):
        result = convert_bounding_region_to_pixels(
            _regions([10, 20, bad_value, 20, 110, 70, 10, 70]),
            image_width_px=300, image_height_px=600, pages=PAGE_PIXEL)
        assert result.status is ConversionStatus.MALFORMED_POLYGON


def test_missing_page_dimensions():
    result = convert_bounding_region_to_pixels(
        _regions([10, 20, 110, 20, 110, 70, 10, 70]),
        image_width_px=300, image_height_px=600,
        pages=[{"pageNumber": 1, "unit": "pixel"}])
    assert result.status is ConversionStatus.MISSING_PAGE_INFO


def test_missing_page_entirely():
    result = convert_bounding_region_to_pixels(
        _regions([10, 20, 110, 20, 110, 70, 10, 70]),
        image_width_px=300, image_height_px=600, pages=[])
    assert result.status is ConversionStatus.MISSING_PAGE_INFO


def test_zero_or_negative_page_dimensions_rejected():
    result = convert_bounding_region_to_pixels(
        _regions([10, 20, 110, 20, 110, 70, 10, 70]),
        image_width_px=300, image_height_px=600,
        pages=[{"pageNumber": 1, "width": 0, "height": 600, "unit": "pixel"}])
    assert result.status is ConversionStatus.MISSING_PAGE_INFO


def test_unsupported_unit():
    result = convert_bounding_region_to_pixels(
        _regions([10, 20, 110, 20, 110, 70, 10, 70]),
        image_width_px=300, image_height_px=600,
        pages=[{"pageNumber": 1, "width": 300, "height": 600, "unit": "millimeter"}])
    assert result.status is ConversionStatus.UNSUPPORTED_UNIT


def test_invalid_image_dimensions():
    result = convert_bounding_region_to_pixels(
        _regions([10, 20, 110, 20, 110, 70, 10, 70]),
        image_width_px=0, image_height_px=600, pages=PAGE_PIXEL)
    assert result.status is ConversionStatus.INVALID_IMAGE_DIMENSIONS


# ---------------------------------------------------------------------------
# multi-page
# ---------------------------------------------------------------------------

def test_multi_page_selects_matching_page_region():
    regions = [
        {"pageNumber": 1, "polygon": [1, 1, 2, 1, 2, 2, 1, 2]},
        {"pageNumber": 2, "polygon": [10, 20, 110, 20, 110, 70, 10, 70]},
    ]
    pages = [
        {"pageNumber": 1, "width": 300, "height": 600, "unit": "pixel"},
        {"pageNumber": 2, "width": 300, "height": 600, "unit": "pixel"},
    ]
    result = convert_bounding_region_to_pixels(
        regions, image_width_px=300, image_height_px=600, pages=pages,
        expected_page_number=2)
    assert result.ok
    assert result.page_number == 2
    assert result.pixel_bbox == (10, 20, 110, 70)


def test_wrong_page_when_no_region_matches_expected_page():
    result = convert_bounding_region_to_pixels(
        _regions([10, 20, 110, 20, 110, 70, 10, 70], page=1),
        image_width_px=300, image_height_px=600, pages=PAGE_PIXEL,
        expected_page_number=2)
    assert result.status is ConversionStatus.WRONG_PAGE


# ---------------------------------------------------------------------------
# dict-shaped input tolerance (as_dict() SDK objects vs plain dicts)
# ---------------------------------------------------------------------------

class _FakeSdkRegion:
    """Stands in for the SDK's BoundingRegion object, which exposes
    .as_dict() rather than being a plain dict."""

    def __init__(self, d):
        self._d = d

    def as_dict(self):
        return self._d


def test_accepts_sdk_style_objects_with_as_dict():
    region = _FakeSdkRegion({"pageNumber": 1,
                            "polygon": [10, 20, 110, 20, 110, 70, 10, 70]})
    result = convert_bounding_region_to_pixels(
        [region], image_width_px=300, image_height_px=600, pages=PAGE_PIXEL)
    assert result.ok
    assert result.pixel_bbox == (10, 20, 110, 70)
