"""Tests for chequemate/signature.py.

Line removal is the critical path under the "any ink counts as present"
rule (a residual, un-removed printed line would read as "present" even on
a genuinely blank cheque) so it is tested directly here, not only as a step
inside analyze_signature_zone. The real-batch test locks in the current
calibration result so a future change that regresses it is caught.

The zone is DERIVED from a found line (chequemate/signature.py's module
docstring explains why a fixed zone was rejected: it was the root cause of
a confirmed false positive on cheques/cheque2.png.png, a different
template). analyze_signature_zone now takes a PreparedCheque (not a bare
Image), since it needs DPI/canvas provenance for the envelope guard.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from chequemate.imageprep import (
    ChequeImagePrepError,
    PreparedCheque,
    RotationDecision,
    prepare_cheque_image,
)
from chequemate.signature import (
    NOISE_FLOOR,
    SignatureEnvelopeError,
    analyze_signature_zone,
    find_signature_line,
    remove_printed_line,
    synthesize_unsigned_variant,
)

CHEQUES_DIR = Path(__file__).resolve().parent.parent / "cheques"
REAL_BATCH = sorted(CHEQUES_DIR.glob("20260820113647241_*.jpg"))
ROTATION_OVERRIDES = {
    "20260820113647241_0001.jpg": "CCW",
    "20260820113647241_0006.jpg": "CCW",
    "20260820113647241_0011.jpg": "CCW",
}
CHEQUE2_PATH = CHEQUES_DIR / "cheque2.png.png"


def _fake_prepared(image: Image.Image, *, dpi: float = 300.0,
                   dpi_source: str = "jfif") -> PreparedCheque:
    """A minimal PreparedCheque for tests that build a synthetic canvas
    directly rather than running the full imageprep pipeline."""
    rotation = RotationDecision(
        direction="CCW", fundamental_score=0.5, harmonic_score=0.3,
        fundamental_lag_px=37, dpi=dpi, dpi_source=dpi_source, confident=True)
    return PreparedCheque(
        image=image, rotation=rotation, skew_angle_deg=0.0,
        source_crop_size=image.size, canvas_size=image.size, scale=(1.0, 1.0))


# ---------------------------------------------------------------------------
# line removal, tested directly (the critical path)
# ---------------------------------------------------------------------------


def _solid_line_mask(width=1860, height=200, line_row=100, line_len=None) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    line_len = line_len or int(width * 0.5)
    start = (width - line_len) // 2
    mask[line_row, start:start + line_len] = True
    return mask


def _microprint_line_mask(width=1860, height=200, line_row=100) -> np.ndarray:
    """Simulates a microprinted security line: short word-length runs with
    gaps, spanning most of the width but with no single long run."""
    mask = np.zeros((height, width), dtype=bool)
    x = 20
    while x < width - 40:
        mask[line_row, x:x + 25] = True  # a "word"
        x += 40  # gap between words
    return mask


def test_solid_line_is_found_and_removed():
    ink = _solid_line_mask()
    cleaned, line_rows = remove_printed_line(ink)
    assert line_rows > 0
    assert not cleaned.any()  # nothing left - the mask WAS only the line


def test_microprint_line_is_found_and_removed():
    """The real reason this module doesn't use single-run-length detection:
    a microprinted line (this batch's most common case) never forms one
    long run, only many short ones with gaps."""
    ink = _microprint_line_mask()
    cleaned, line_rows = remove_printed_line(ink)
    assert line_rows > 0
    assert not cleaned.any()


def test_signature_ink_survives_line_removal():
    """A signature stroke sitting well clear of the line's row must not be
    touched by removal."""
    ink = _solid_line_mask()
    ink[40:60, 300:340] = True  # a compact "signature" blob far from the line
    cleaned, line_rows = remove_printed_line(ink)
    assert line_rows > 0
    assert cleaned[40:60, 300:340].all()  # signature untouched
    assert not cleaned[100].any()  # line row fully cleared


def test_signature_on_an_adjacent_row_survives():
    """Detection is row-based: removal is confined to the specific row(s)
    identified as the line, so a signature stroke one row above or below
    the line - even directly overlapping its x-range - must survive."""
    ink = _solid_line_mask(line_len=1000, line_row=100)
    ink[95, 300:340] = True  # a stroke one row above, overlapping the line's x-range
    cleaned, line_rows = remove_printed_line(ink)
    assert line_rows > 0
    assert cleaned[95, 300:340].all()
    assert not cleaned[100].any()


def test_wide_signature_alone_is_not_mistaken_for_the_line():
    """A tall, scattered, low-density region (a large looping signature, no
    printed line at all) must not be misread as the line."""
    ink = np.zeros((200, 1860), dtype=bool)
    rng = np.random.default_rng(0)
    for r in range(60, 160):
        cols = rng.choice(1860, size=15, replace=False)
        ink[r, cols] = True
    cleaned, line_rows = remove_printed_line(ink)
    assert line_rows == 0


def test_no_line_found_gives_zero_row_count():
    ink = np.zeros((200, 1860), dtype=bool)
    ink[50:55, 100:150] = True  # a small mark, nowhere near line-width
    _, line_rows = remove_printed_line(ink)
    assert line_rows == 0


# ---------------------------------------------------------------------------
# find_signature_line / analyze_signature_zone: ambiguous vs decided
# ---------------------------------------------------------------------------


def _canvas_with_line(width=1860, height=780, line_frac=0.4) -> Image.Image:
    """A blank canvas with a printed line inside the search region, plus an
    unrelated ink block elsewhere so zone_ink_threshold has real content to
    anchor an Otsu split on."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    line_w = int(width * line_frac)
    x1 = int(width * 0.5)
    y = int(height * 0.65)
    draw.rectangle([x1, y, x1 + line_w, y + 2], fill="black")
    draw.rectangle([50, 50, 400, 90], fill="black")  # unrelated ink for Otsu
    return image


def test_ambiguous_when_no_line_found():
    blank = Image.new("RGB", (1860, 780), "white")
    ImageDraw.Draw(blank).rectangle([50, 50, 400, 90], fill="black")
    result = analyze_signature_zone(_fake_prepared(blank))
    assert result.ambiguous is True
    assert result.present is None
    assert result.reason


def test_present_when_line_removed_and_ink_remains():
    image = _canvas_with_line()
    line = find_signature_line(image)
    assert line is not None
    draw = ImageDraw.Draw(image)
    # low-chroma dark navy, not pure "blue" (0,0,255) - pure blue's chroma
    # (255) would be filtered as "too colored", same as a real decorative
    # graphic should be; real ballpoint ink is dark and only mildly tinted.
    draw.ellipse([line.col_start + 20, line.row_start - 60,
                 line.col_start + 100, line.row_start - 10],
                outline=(20, 20, 55), width=4)
    result = analyze_signature_zone(_fake_prepared(image))
    assert result.ambiguous is False
    assert result.present is True


def test_absent_when_only_the_line_is_present():
    image = _canvas_with_line()
    result = analyze_signature_zone(_fake_prepared(image))
    assert result.ambiguous is False
    assert result.present is False
    assert result.ink_coverage < NOISE_FLOOR


# ---------------------------------------------------------------------------
# envelope guard
# ---------------------------------------------------------------------------


def test_envelope_rejects_wrong_dpi():
    image = _canvas_with_line()
    with pytest.raises(SignatureEnvelopeError):
        analyze_signature_zone(_fake_prepared(image, dpi=96.0))


def test_envelope_rejects_fallback_dpi_source():
    image = _canvas_with_line()
    with pytest.raises(SignatureEnvelopeError):
        analyze_signature_zone(_fake_prepared(image, dpi_source="fallback"))


def test_envelope_rejects_wrong_canvas_size():
    image = Image.new("RGB", (900, 400), "white")
    with pytest.raises(SignatureEnvelopeError):
        analyze_signature_zone(_fake_prepared(image))


def test_envelope_accepts_validated_input():
    image = _canvas_with_line()
    result = analyze_signature_zone(_fake_prepared(image))
    assert result.ambiguous is False  # no exception raised


# ---------------------------------------------------------------------------
# real batch: locked-in calibration result
# ---------------------------------------------------------------------------


@pytest.mark.skipif(len(REAL_BATCH) < 22, reason="real cheque batch not present")
def test_real_batch_signature_detection_matches_calibration():
    """Locks in the current result: 20/22 real (known-signed) cheques
    resolve confidently to present, 2 (_0005, _0017 - a shorter, lighter
    printed line design whose span never reaches the threshold anywhere in
    the search region) correctly refuse rather than guess. Zero real files
    are misread as absent."""
    KNOWN_AMBIGUOUS = {"20260820113647241_0005.jpg", "20260820113647241_0017.jpg"}

    present = absent = ambiguous = 0
    ambiguous_files = set()
    for path in REAL_BATCH:
        result = prepare_cheque_image(path, rotation_override=ROTATION_OVERRIDES.get(path.name))
        analysis = analyze_signature_zone(result)
        if analysis.ambiguous:
            ambiguous += 1
            ambiguous_files.add(path.name)
        elif analysis.present:
            present += 1
        else:
            absent += 1

    assert absent == 0
    assert ambiguous_files == KNOWN_AMBIGUOUS
    assert present == 22 - len(KNOWN_AMBIGUOUS)


@pytest.mark.skipif(len(REAL_BATCH) < 22, reason="real cheque batch not present")
def test_synthetic_line_intact_negative_never_false_positive_on_resolved_files():
    """The realistic negative (signature erased, printed line kept) must
    not be called present on any file the detector actually resolves."""
    for path in REAL_BATCH:
        result = prepare_cheque_image(path, rotation_override=ROTATION_OVERRIDES.get(path.name))
        real = analyze_signature_zone(result)
        if real.ambiguous:
            continue
        negative_img = synthesize_unsigned_variant(result, keep_line=True)
        negative = analyze_signature_zone(replace(result, image=negative_img))
        if negative.ambiguous:
            continue
        assert negative.present is False, f"{path.name} false-positived on line-intact negative"


# ---------------------------------------------------------------------------
# the one REAL (non-synthetic) negative available - now refused outright by
# the envelope guard rather than reaching line-search at all. NOT part of
# the 22-file batch or its calibration, and NOT something to tune constants
# against (one sample is not a distribution). Documented here so this
# finding is never silently lost.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CHEQUE2_PATH.is_file(), reason="cheque2.png.png not present")
def test_real_unsigned_cheque_is_refused_by_envelope_guard():
    """cheque2.png.png is a real, visually-confirmed-blank-signature-line
    cheque - a completely different, much older test fixture (96 DPI,
    365x816 native) than the 22-file 300 DPI batch this module was
    calibrated against, and never part of that calibration.

    Before the envelope guard (Correction/Task 2) existed, this file was a
    confident FALSE POSITIVE (present=True), not a coincidental refusal -
    the root cause was SIGNATURE_ZONE_FRAC (a fixed rectangle) landing
    partly on printed "/100 DOLLARS" text for this different template. The
    line-first rewrite (Task 1) plus this guard together convert that into
    an honest, early refusal: the DPI (96) is caught before line-search
    even runs, so this file's specific line-search behavior no longer
    matters for the outcome.
    """
    result = prepare_cheque_image(CHEQUE2_PATH, rotation_override="CCW")
    with pytest.raises(SignatureEnvelopeError):
        analyze_signature_zone(result)
