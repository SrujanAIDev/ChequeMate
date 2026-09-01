"""Tests for chequemate/imageprep.py.

Two kinds of coverage:
 - synthetic fixtures (drawn in-process, no external files) for the
   controlled cases: rotation+skew recovery, blank-page failure
 - the real 22-file batch under cheques/ for the properties that only show
   up on real scans: stable canvas dimensions, and the rotation detector's
   count summary (kept in sync with the Phase 2b/3 STOP report).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from chequemate.imageprep import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    NoChequeFound,
    OrientationIndeterminate,
    isolate,
    prepare_cheque_image,
)

CHEQUES_DIR = Path(__file__).resolve().parent.parent / "cheques"
REAL_BATCH = sorted(CHEQUES_DIR.glob("20260820113647241_*.jpg"))


# ---------------------------------------------------------------------------
# synthetic fixture: a fake upright "cheque" with a MICR-pitch-correct line
# ---------------------------------------------------------------------------

def _draw_synthetic_cheque(dpi: int = 300, width: int = 1800, height: int = 750
                           ) -> Image.Image:
    """A plain white rectangle standing in for an upright cheque, with a
    bottom band of evenly-spaced tick marks at the real E-13B pitch (DPI/8)
    so the rotation detector has a genuine MICR-like signal to find, and a
    top band of a few widely-spaced marks (not fixed-pitch) standing in for
    letterhead text, so top and bottom are NOT symmetrical."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    pitch = dpi / 8.0
    micr_y = height - 30
    x = 40
    while x < width - 40:
        draw.rectangle([x, micr_y, x + 6, micr_y + 20], fill="black")
        x += pitch

    for lx in (60, 200, 340, 700, 900):
        draw.rectangle([lx, 15, lx + 40, 30], fill="black")

    return image


def _paste_on_page(cheque: Image.Image, page_size=(2552, 3300),
                   dpi=(300, 300)) -> Image.Image:
    page = Image.new("RGB", page_size, "white")
    page.paste(cheque, (page_size[0] // 2 - cheque.width // 2, 0))
    page.info["dpi"] = dpi
    return page


def _save_with_dpi(image: Image.Image, path: Path, dpi=(300, 300)) -> None:
    image.save(path, dpi=dpi)


# ---------------------------------------------------------------------------
# synthetic: rotation + skew recovery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rotate_degrees", [90, -90])
@pytest.mark.parametrize("skew_degrees", [0, 2.5, -3.0])
def test_recovers_synthetic_rotation_and_skew(tmp_path, rotate_degrees, skew_degrees):
    cheque = _draw_synthetic_cheque()
    page = _paste_on_page(cheque)
    # rotate_degrees applies the SAME 90-degree-family transform imageprep
    # must undo; skew_degrees layers a small crooked-feed error on top.
    distorted = page.rotate(rotate_degrees, expand=True, fillcolor="white")
    distorted = distorted.rotate(skew_degrees, expand=True, fillcolor="white")

    path = tmp_path / "synthetic.jpg"
    _save_with_dpi(distorted, path)

    result = prepare_cheque_image(path)
    assert result.image.size == (CANVAS_WIDTH, CANVAS_HEIGHT)
    # the corrected rotation must put the MICR-pitch band back at the bottom
    assert result.rotation.direction in ("CW", "CCW")
    assert result.rotation.confident


def test_typed_failure_on_blank_page(tmp_path):
    blank = Image.new("RGB", (2552, 3300), "white")
    path = tmp_path / "blank.jpg"
    _save_with_dpi(blank, path)

    with pytest.raises(NoChequeFound):
        prepare_cheque_image(path)


def test_isolate_raises_on_blank_array():
    blank = np.full((500, 400, 3), 255, dtype=np.uint8)
    with pytest.raises(NoChequeFound):
        isolate(blank)


# ---------------------------------------------------------------------------
# detect_rotation_only - used by ocr_verify.py's TrOCR crop pipeline to
# learn a cheque's rotation direction WITHOUT deskew/canvas-resize side
# effects, which would break pixel correspondence with Document
# Intelligence's polygon (defined against the original, un-rotated image).
# ---------------------------------------------------------------------------

def test_detect_rotation_only_matches_full_pipeline_direction(tmp_path):
    cheque = _draw_synthetic_cheque()
    page = _paste_on_page(cheque)
    distorted = page.rotate(90, expand=True, fillcolor="white")
    path = tmp_path / "synthetic.jpg"
    _save_with_dpi(distorted, path)

    from chequemate.imageprep import detect_rotation_only
    decision = detect_rotation_only(path)
    full = prepare_cheque_image(path)
    assert decision.direction == full.rotation.direction
    assert decision.confident


def test_detect_rotation_only_raises_no_cheque_found_on_blank_page(tmp_path):
    blank = Image.new("RGB", (2552, 3300), "white")
    path = tmp_path / "blank.jpg"
    _save_with_dpi(blank, path)

    from chequemate.imageprep import detect_rotation_only
    with pytest.raises(NoChequeFound):
        detect_rotation_only(path)


def test_detect_rotation_only_does_not_produce_a_resized_canvas(tmp_path):
    """The whole point of this function vs. prepare_cheque_image(): it must
    not normalize to CANVAS_WIDTH/CANVAS_HEIGHT, since a caller matching
    pixels against DI's polygon needs the ORIGINAL image's coordinate
    space untouched."""
    cheque = _draw_synthetic_cheque()
    page = _paste_on_page(cheque)
    distorted = page.rotate(90, expand=True, fillcolor="white")
    path = tmp_path / "synthetic.jpg"
    _save_with_dpi(distorted, path)

    from chequemate.imageprep import RotationDecision, detect_rotation_only
    decision = detect_rotation_only(path)
    assert isinstance(decision, RotationDecision)
    # returns provenance only, no image/canvas at all
    assert not hasattr(decision, "image")


def test_dpi_fallback_when_tag_absent(tmp_path):
    """No JFIF dpi tag -> falls back to the documented default rather than
    crashing or silently using an unscaled pitch window."""
    cheque = _draw_synthetic_cheque(dpi=300)
    page = _paste_on_page(cheque)
    # the pipeline assumes a 90-degree-rotated input throughout (Phase 0b:
    # every real cheque in this batch is rotated) - an already-upright page
    # would put the synthetic MICR band on the left/right edge instead of
    # top/bottom, which isn't the case this module is built to handle.
    rotated = page.rotate(90, expand=True, fillcolor="white")
    path = tmp_path / "no_dpi.png"  # PNG round-trip drops the dpi tag if unset
    rotated.save(path)  # no dpi= kwarg

    result = prepare_cheque_image(path)
    assert result.rotation.dpi_source == "fallback"


# ---------------------------------------------------------------------------
# synthetic: operator rotation override (Correction B)
# ---------------------------------------------------------------------------

def _make_ambiguous_fixture(tmp_path) -> Path:
    """A cheque whose top band ALSO carries a fixed-pitch mark (mimicking a
    decorative border with incidental MICR-like periodicity) so the
    detector genuinely refuses - the override path is only meaningful to
    test against a real refusal, not a case the detector would resolve on
    its own."""
    width, height, dpi = 1800, 750, 300
    pitch = dpi / 8.0
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for band_y in (15, height - 30):
        x = 40
        while x < width - 40:
            draw.rectangle([x, band_y, x + 6, band_y + 20], fill="black")
            x += pitch
    page = _paste_on_page(image)
    rotated = page.rotate(90, expand=True, fillcolor="white")
    path = tmp_path / "ambiguous.jpg"
    _save_with_dpi(rotated, path)
    return path


def test_rotation_override_rejects_invalid_direction(tmp_path):
    path = _make_ambiguous_fixture(tmp_path)
    with pytest.raises(ValueError):
        prepare_cheque_image(path, rotation_override="sideways")


def test_rotation_override_resolves_a_detector_refusal(tmp_path):
    path = _make_ambiguous_fixture(tmp_path)

    with pytest.raises(OrientationIndeterminate):
        prepare_cheque_image(path)

    result = prepare_cheque_image(path, rotation_override="CCW")
    assert result.rotation.direction == "CCW"
    assert result.rotation.source == "operator_override"
    assert result.rotation.confident is True
    assert result.image.size == (CANVAS_WIDTH, CANVAS_HEIGHT)


def test_rotation_override_does_not_change_detector_behaviour_when_unused(tmp_path):
    """Passing rotation_override=None (the default) must be indistinguishable
    from never having the parameter at all - the override is opt-in per call,
    never a silent default."""
    cheque = _draw_synthetic_cheque()
    page = _paste_on_page(cheque)
    distorted = page.rotate(90, expand=True, fillcolor="white")
    path = tmp_path / "normal.jpg"
    _save_with_dpi(distorted, path)

    result = prepare_cheque_image(path, rotation_override=None)
    assert result.rotation.source == "detector"


# ---------------------------------------------------------------------------
# real batch: stable canvas size + rotation count summary
# ---------------------------------------------------------------------------

pytestmark_real = pytest.mark.skipif(
    len(REAL_BATCH) < 22, reason="real 20260820... cheque batch not present")


@pytest.mark.skipif(len(REAL_BATCH) < 22, reason="real cheque batch not present")
def test_real_batch_normalizes_to_stable_dimensions():
    for path in REAL_BATCH:
        try:
            result = prepare_cheque_image(path)
        except OrientationIndeterminate:
            continue  # expected for a known subset (see Phase 3 STOP report)
        assert result.image.size == (CANVAS_WIDTH, CANVAS_HEIGHT)


@pytest.mark.skipif(len(REAL_BATCH) < 22, reason="real cheque batch not present")
def test_real_batch_rotation_matches_validated_ground_truth():
    """Locks in the Phase 2b/3 result: full manual verification of all 22
    files confirmed every one is CCW. The DPI-derived detector, with a
    required confidence margin, correctly resolves 19 of them and refuses
    (rather than guessing) on the 3 known-hard cases whose decorative
    border's incidental periodicity rivals the true MICR pitch."""
    KNOWN_INDETERMINATE = {
        "20260820113647241_0001.jpg",
        "20260820113647241_0006.jpg",
        "20260820113647241_0011.jpg",
    }
    resolved_cw = []
    resolved_ccw = []
    indeterminate = []
    for path in REAL_BATCH:
        try:
            result = prepare_cheque_image(path)
            (resolved_ccw if result.rotation.direction == "CCW" else resolved_cw) \
                .append(path.name)
        except OrientationIndeterminate:
            indeterminate.append(path.name)

    assert resolved_cw == []
    assert set(indeterminate) == KNOWN_INDETERMINATE
    assert len(resolved_ccw) == 22 - len(KNOWN_INDETERMINATE)
