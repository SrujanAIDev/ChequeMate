"""Signature-zone ink detector, built on imageprep's normalized 1860x780
canvas (chequemate/imageprep.py).

Design rule: ANY ink in the signature zone counts as present, including a
stray mark or a partial stroke. That makes line removal the entire
detector, not a preprocessing step for it - the printed PER/signature line
is itself ink, and if it isn't removed, every cheque (including a
genuinely blank one) reads as "present" simply because the printed line is
there.

The zone is DERIVED from the line, not assumed. An earlier version fixed
SIGNATURE_ZONE_FRAC as a constant rectangle on the canvas, calibrated by
eyeballing a contact sheet of the 22-file batch. That rectangle was the
root cause of a real, confirmed failure: on cheques/cheque2.png.png (a
different, much older template), the fixed zone happened to contain both
the real signature line AND unrelated printed "/100 DOLLARS" text, and the
selection logic picked the wrong one - a confident false positive, not a
refusal. The fix is architectural: search a generous region for the
printed line FIRST (the one feature guaranteed present on every cheque,
signed or not), then build the zone from whatever was actually found - a
band above that specific line, bounded by its own horizontal extent. If no
line is found anywhere in the search region, this refuses
(ParseStatus.AMBIGUOUS) rather than measuring ink in a zone nobody
confirmed is meaningful.

VALIDATED ENVELOPE: every constant here (search region, span/height/density
thresholds, NOISE_FLOOR) was calibrated against 300 DPI scans of the
22-file 20260820... batch's template family. `analyze_signature_zone`
checks DPI and canvas provenance on entry and raises
`SignatureEnvelopeError` outside that envelope (see cheque2.png.png in
CLAUDE.md - 96 DPI, a different template - which is exactly the case this
guard exists to catch before computing a number nobody should trust). This
module has NOT been validated for other DPIs or template families.

Testing notes:
 - `remove_printed_line` is exercised directly, not only as a step inside
   `analyze_signature_zone` - see test_signature.py.
 - `find_signature_line`'s search region was confirmed against a contact
   sheet of all 22 real prepared cheques (scripts/signature_contact_sheet.py).
 - NOISE_FLOOR is calibrated in scripts/calibrate_signature_threshold.py
   against real cheque background noise, not against any single file's
   label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .imageprep import CANVAS_HEIGHT, CANVAS_WIDTH, PreparedCheque, otsu_threshold

# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class SignatureEnvelopeError(Exception):
    """The input falls outside the DPI/template envelope this module's
    constants were calibrated for. Raised instead of computing a number
    nobody should trust."""

    def __init__(self, message: str, *, dpi: float, dpi_source: str,
                canvas_size: tuple[int, int]):
        super().__init__(message)
        self.dpi = dpi
        self.dpi_source = dpi_source
        self.canvas_size = canvas_size


# ---------------------------------------------------------------------------
# validated envelope
# ---------------------------------------------------------------------------

VALIDATED_DPI = 300.0
VALIDATED_DPI_TOLERANCE_FRAC = 0.05  # +/-5% - the batch is exactly 300 via JFIF tag
VALIDATED_CANVAS_SIZE = (CANVAS_WIDTH, CANVAS_HEIGHT)


def _check_envelope(prepared: PreparedCheque) -> None:
    dpi = prepared.rotation.dpi
    dpi_source = prepared.rotation.dpi_source
    canvas_size = prepared.image.size

    if canvas_size != VALIDATED_CANVAS_SIZE:
        raise SignatureEnvelopeError(
            f"canvas size {canvas_size} != validated {VALIDATED_CANVAS_SIZE} "
            f"- every constant in this module assumes imageprep.normalize_canvas's "
            f"fixed output size",
            dpi=dpi, dpi_source=dpi_source, canvas_size=canvas_size)

    if dpi_source != "jfif":
        raise SignatureEnvelopeError(
            f"DPI source is {dpi_source!r}, not a confirmed JFIF tag - this "
            f"module's constants are calibrated against confirmed 300 DPI "
            f"scans, and an assumed/fallback DPI gives no real confidence "
            f"the input matches that envelope",
            dpi=dpi, dpi_source=dpi_source, canvas_size=canvas_size)

    deviation = abs(dpi - VALIDATED_DPI) / VALIDATED_DPI
    if deviation > VALIDATED_DPI_TOLERANCE_FRAC:
        raise SignatureEnvelopeError(
            f"DPI {dpi:.1f} is outside the validated envelope "
            f"({VALIDATED_DPI:.0f} +/-{VALIDATED_DPI_TOLERANCE_FRAC:.0%}) - "
            f"this module's zone geometry and thresholds were calibrated "
            f"against 300 DPI scans of one template family (the "
            f"20260820... batch) and have not been shown to generalize "
            f"beyond it. See CLAUDE.md's signature-detector section: "
            f"cheques/cheque2.png.png (96 DPI, a different template) was a "
            f"confirmed false positive before this guard existed.",
            dpi=dpi, dpi_source=dpi_source, canvas_size=canvas_size)


# ---------------------------------------------------------------------------
# search region + line location
# ---------------------------------------------------------------------------

# Generous lower-right region searched for the printed signature line.
# Confirmed against all 22 real prepared cheques (contact sheet) plus
# cheques/cheque2.png.png: wide/tall enough that the real line is never
# truncated for any of those templates. NOT simply "as wide as possible" -
# a full-width or much-taller search was tried and rejected: it merges the
# separate MEMO line into the same "span" as the signature line (both sit
# on the same cheque row on several templates), and starts matching MICR
# text / bank address blocks that also happen to be wide and dense. This
# specific region was the widest one found that fixed the cheque2.png.png
# failure without introducing a new false-confident match on any of the 22
# real files (see the Phase 4 correction's exploration in this module's
# git history / CLAUDE.md for the specific configurations tried and
# rejected).
SIGNATURE_SEARCH_REGION_FRAC = (0.45, 0.45, 1.0, 0.92)

# Ink is "near-black or near-blue-ballpoint, not a colored graphic": looser
# than imageprep's border-vs-MICR chroma filter (20) because signatures are
# routinely blue ballpoint ink, not pure black, and must not be filtered out
# as "too colored".
NEUTRAL_CHROMA_MAX = 45

# The printed PER/signature line is NOT always a solid rule - on roughly
# three-quarters of this batch's templates it's a microprinted security
# text line (e.g. "MICROPRINT SECURITY FEATURE..." repeated in tiny type),
# which LOOKS like a continuous line to the eye but is actually many short
# word-length ink runs with gaps between them. A single-longest-run test
# misses that line entirely, because no one run is long. What both a solid
# rule and a microprint line share instead is SPAN: on the rows they
# occupy, the leftmost-to-rightmost ink extent covers a large fraction of
# the CANVAS width (not the search region's width, which can vary), even
# though the ink inside that span isn't wall-to-wall.
LINE_MIN_SPAN_CANVAS_FRAC = 0.25

# The printed line, solid or microprint, is physically thin - a few pixels
# of ink height at this canvas's scale. A handwritten signature can ALSO
# produce wide-span rows (a looping flourish), but those either come alone
# or stack into a much TALLER run than the printed line ever does. Capping
# the qualifying window's height is what actually distinguishes "the
# printed line" from "a wide stretch of handwriting".
LINE_MAX_WINDOW_HEIGHT_PX = 16

# A loose floor on the qualifying window's mean fill-within-span, just to
# reject a coincidental pair of specks far apart on the same couple of rows
# from being mistaken for a line.
LINE_MIN_WINDOW_DENSITY = 0.05

# Small padding around the found line's own extent when deriving the zone.
ZONE_MARGIN_X_FRAC = 0.02
ZONE_HEIGHT_ABOVE_LINE_FRAC = 0.35  # generous band above the line for the signature
ZONE_MARGIN_BELOW_LINE_PX = 8       # small margin below for descenders

# Calibrated in scripts/calibrate_signature_threshold.py against REAL
# background noise (JPEG recompression speckle, paper texture, safety-
# pattern chroma bleed) in a zone that has genuinely had all real signature
# ink removed. Re-derived after the line-first zone rewrite (the zone's
# geometry changed, so the noise floor it needs to clear changed too): the
# line-intact-negative ceiling is 0.00142 and the real-signed floor is
# 0.00221 - only a 0.00079 gap, narrower than before this rewrite (0.0015).
# NOISE_FLOOR sits at the midpoint. Narrower headroom is reported honestly,
# not hidden - see the calibration report and CLAUDE.md's signature-
# detector section. NOT tuned against cheques/cheque2.png.png (the one
# real, out-of-batch negative) - that file is now refused outright by the
# DPI envelope guard before this constant is ever consulted.
NOISE_FLOOR = 0.0018


@dataclass(frozen=True)
class LineLocation:
    row_start: int  # absolute canvas coordinates
    row_end: int    # exclusive
    col_start: int
    col_end: int    # exclusive
    density: float


@dataclass(frozen=True)
class SignatureAnalysis:
    present: bool | None        # None only when ambiguous
    ambiguous: bool
    ink_coverage: float         # fraction of zone pixels that are ink after line removal
    stroke_extent_frac: float   # vertical span of remaining ink / zone height - diagnostic
                                # only, does NOT gate present/absent (the "any ink, even a
                                # partial stroke, counts" rule means a real positive can have
                                # near-zero extent) - reported for calibration transparency.
    line: LineLocation | None   # where the printed line was found, if it was
    reason: str | None = None   # populated when ambiguous


# ---------------------------------------------------------------------------
# ink mask
# ---------------------------------------------------------------------------


def _ink_mask(rgb: np.ndarray, thresh: float) -> np.ndarray:
    """Near-black-or-ballpoint-ink mask, using a threshold supplied by the
    caller rather than recomputed from a small crop. Otsu needs a genuinely
    bimodal histogram to mean anything; a small, mostly-blank crop lets it
    lock onto faint paper-texture variation and call it "ink" - exactly
    backwards on the case this detector most needs to get right. The
    threshold is derived once from the FULL cheque image
    (`zone_ink_threshold`), which always has abundant real ink to anchor a
    genuine bimodal split."""
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    rgb16 = rgb.astype(np.int16)
    chroma = rgb16.max(axis=2) - rgb16.min(axis=2)
    return (gray < thresh) & (chroma < NEUTRAL_CHROMA_MAX)


def zone_ink_threshold(image: Image.Image) -> float:
    """Otsu threshold from the FULL prepared cheque image - see
    `_ink_mask`'s docstring for why this must not be recomputed on a small
    crop alone."""
    return otsu_threshold(np.asarray(image.convert("L")))


# ---------------------------------------------------------------------------
# line detection - the critical path
# ---------------------------------------------------------------------------


def _row_spans(ink: np.ndarray) -> list[tuple[int, int, float] | None]:
    """Per row: (leftmost_ink_x, rightmost_ink_x, fill_density_within_span),
    or None for a row with no ink at all."""
    spans: list[tuple[int, int, float] | None] = []
    for row in ink:
        idx = np.flatnonzero(row)
        if idx.size == 0:
            spans.append(None)
            continue
        left, right = int(idx[0]), int(idx[-1])
        span_width = right - left + 1
        spans.append((left, right, idx.size / span_width))
    return spans


def _find_line_window(ink: np.ndarray, min_span_px: int
                      ) -> tuple[int, int, int, int, float] | None:
    """Core primitive: find the printed line's window of rows within `ink`.

    Among all individually-valid windows (span >= min_span_px on every row
    in the window, height <= LINE_MAX_WINDOW_HEIGHT_PX, mean fill-density
    >= LINE_MIN_WINDOW_DENSITY), picks the LOWEST (bottom-most) one - the
    signature/PER line is structurally the last ruled line before a
    cheque's bank-name/MICR block, and on real cheques with multiple
    candidate wide-dense rows (an amount-in-words underline, a bank
    address block, MICR itself), "lowest valid" reliably lands on the
    right one where "highest density" does not (dense printed TEXT, e.g.
    "/100 DOLLARS", routinely beats a genuine but sparser printed line on
    density alone - this was the exact mechanism behind the
    cheque2.png.png false positive).

    Returns (row_start, row_end, col_start, col_end, density) in `ink`'s
    own coordinate system, or None if no qualifying window exists.
    """
    h, w = ink.shape
    spans = _row_spans(ink)
    qualifies = [s is not None and (s[1] - s[0] + 1) >= min_span_px for s in spans]

    valid_windows: list[tuple[int, int, float]] = []
    for window in range(1, min(LINE_MAX_WINDOW_HEIGHT_PX, h) + 1):
        for start in range(0, h - window + 1):
            end = start + window
            if not all(qualifies[start:end]):
                continue
            densities = [spans[r][2] for r in range(start, end)]
            mean_density = sum(densities) / len(densities)
            if mean_density < LINE_MIN_WINDOW_DENSITY:
                continue
            valid_windows.append((start, end, mean_density))

    if not valid_windows:
        return None

    # Pick the lowest-starting valid window, then expand it through any
    # immediately-adjacent qualifying rows (capped at the same height
    # limit): a solid line several rows thick can satisfy the window
    # search at more than one sub-window of itself, and picking only the
    # smallest/latest-starting one would leave the rest of the same
    # physical line as unremoved residual ink - the same failure mode as
    # picking the single densest row of a multi-row line.
    valid_windows.sort(key=lambda t: t[0])
    start, end, density = valid_windows[-1]
    while start - 1 >= 0 and qualifies[start - 1] and (end - (start - 1)) <= LINE_MAX_WINDOW_HEIGHT_PX:
        start -= 1
    while end < h and qualifies[end] and (end + 1 - start) <= LINE_MAX_WINDOW_HEIGHT_PX:
        end += 1

    lefts = [spans[r][0] for r in range(start, end)]
    rights = [spans[r][1] for r in range(start, end)]
    return (start, end, min(lefts), max(rights) + 1, density)


def remove_printed_line(ink: np.ndarray) -> tuple[np.ndarray, int]:
    """Independently-testable entry point for the critical-path step: zero
    out the printed line's pixels within an arbitrary ink mask, return
    (cleaned_mask, line_row_count). line_row_count == 0 means no
    qualifying line was found anywhere in `ink`. Uses `ink`'s own width to
    scale the span threshold, for standalone use on a synthetic array in
    tests - `find_signature_line` instead scales against the real canvas
    width explicitly."""
    h, w = ink.shape
    min_span_px = int(LINE_MIN_SPAN_CANVAS_FRAC * w)
    found = _find_line_window(ink, min_span_px)
    if found is None:
        return ink.copy(), 0
    row_start, row_end, col_start, col_end, _ = found
    line = np.zeros_like(ink)
    line[row_start:row_end, col_start:col_end] = True
    cleaned = ink & ~line
    return cleaned, row_end - row_start


def find_signature_line(image: Image.Image) -> LineLocation | None:
    """Search SIGNATURE_SEARCH_REGION_FRAC for the printed signature line.
    Returns its location in absolute canvas coordinates, or None if no
    qualifying line was found anywhere in the region."""
    thresh = zone_ink_threshold(image)
    x1f, y1f, x2f, y2f = SIGNATURE_SEARCH_REGION_FRAC
    width, height = image.size
    x1, y1 = round(x1f * width), round(y1f * height)
    x2, y2 = round(x2f * width), round(y2f * height)

    region = np.asarray(image.convert("RGB"))[y1:y2, x1:x2]
    ink = _ink_mask(region, thresh)
    min_span_px = int(LINE_MIN_SPAN_CANVAS_FRAC * width)
    found = _find_line_window(ink, min_span_px)
    if found is None:
        return None

    row_start, row_end, col_start, col_end, density = found
    return LineLocation(
        row_start=y1 + row_start, row_end=y1 + row_end,
        col_start=x1 + col_start, col_end=x1 + col_end, density=density)


def derive_zone_from_line(line: LineLocation, canvas_width: int, canvas_height: int
                          ) -> tuple[int, int, int, int]:
    """The zone is a band ABOVE the found line, bounded by the line's own
    horizontal extent (plus a small margin) - not a fixed rectangle assumed
    in advance. A small margin below the line is kept for descenders."""
    margin_x = round(ZONE_MARGIN_X_FRAC * canvas_width)
    x1 = max(0, line.col_start - margin_x)
    x2 = min(canvas_width, line.col_end + margin_x)
    height_above = round(ZONE_HEIGHT_ABOVE_LINE_FRAC * canvas_height)
    y1 = max(0, line.row_start - height_above)
    y2 = min(canvas_height, line.row_end + ZONE_MARGIN_BELOW_LINE_PX)
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def analyze_signature_zone(prepared: PreparedCheque) -> SignatureAnalysis:
    """Find the printed line, derive a zone from it, remove the line, and
    decide present/absent/ambiguous from what's left.

    Raises SignatureEnvelopeError if `prepared` falls outside the DPI/
    canvas envelope this module's constants were calibrated for (see the
    module docstring) - a confident number is never computed on input
    nobody has validated this pipeline against.

    AMBIGUOUS fires when no qualifying printed line was found anywhere in
    the search region - the crop may be misaligned, the template
    unrecognized, or the image otherwise unreadable here - rather than
    trusting a coverage measurement taken from a zone nobody confirmed is
    meaningful.
    """
    _check_envelope(prepared)
    image = prepared.image

    line = find_signature_line(image)
    if line is None:
        return SignatureAnalysis(
            present=None, ambiguous=True, ink_coverage=0.0,
            stroke_extent_frac=0.0, line=None,
            reason="no printed signature line was found anywhere in the "
                  "search region - the crop may be misaligned, the "
                  "template unrecognized, or the zone unreadable; a "
                  "presence call from here is not trustworthy")

    thresh = zone_ink_threshold(image)
    x1, y1, x2, y2 = derive_zone_from_line(line, *image.size)
    zone = np.asarray(image.convert("RGB"))[y1:y2, x1:x2]
    ink = _ink_mask(zone, thresh)

    # mask out the exact (already-known, absolute) line pixels, mapped into
    # zone-relative coordinates
    line_r1 = max(0, line.row_start - y1)
    line_r2 = min(ink.shape[0], line.row_end - y1)
    line_c1 = max(0, line.col_start - x1)
    line_c2 = min(ink.shape[1], line.col_end - x1)
    cleaned = ink.copy()
    cleaned[line_r1:line_r2, line_c1:line_c2] = False

    coverage = float(cleaned.mean())
    rows_with_ink = np.flatnonzero(cleaned.any(axis=1))
    extent = float((rows_with_ink[-1] - rows_with_ink[0] + 1) / cleaned.shape[0]) \
        if rows_with_ink.size else 0.0

    return SignatureAnalysis(
        present=coverage > NOISE_FLOOR, ambiguous=False,
        ink_coverage=coverage, stroke_extent_frac=extent, line=line)


# ---------------------------------------------------------------------------
# synthetic negatives (calibration + tests only - not part of detection)
# ---------------------------------------------------------------------------


def synthesize_unsigned_variant(prepared: PreparedCheque, *, keep_line: bool
                                ) -> Image.Image:
    """Paint over real signature ink in a REAL, known-signed cheque's zone
    to fabricate the negative class the real batch doesn't otherwise
    contain.

    `keep_line=True` erases everything the ink mask finds in the derived
    zone EXCEPT the found printed line - "the printed line exists, nobody
    signed it", the case that actually tests line removal. `keep_line=False`
    erases everything including the line - the easier, fully-blank case
    (which `analyze_signature_zone` can't score at all: erasing the line
    removes the fingerprint it needs, correctly triggering AMBIGUOUS).

    Only as good as the line/ink detection it reuses: any real signature
    pixel this mask fails to flag as ink survives into the "negative"
    image, contaminating that specific sample's label. Requires a
    resolvable line (raises the same way analyze_signature_zone would if
    none is found) since the zone is derived from it.
    """
    image = prepared.image
    line = find_signature_line(image)
    if line is None:
        raise ValueError("no printed signature line found - cannot derive "
                         "a zone to synthesize a negative from")

    thresh = zone_ink_threshold(image)
    x1, y1, x2, y2 = derive_zone_from_line(line, *image.size)
    arr = np.asarray(image.convert("RGB")).copy()
    zone = arr[y1:y2, x1:x2].copy()
    ink = _ink_mask(zone, thresh)

    line_r1 = max(0, line.row_start - y1)
    line_r2 = min(ink.shape[0], line.row_end - y1)
    line_c1 = max(0, line.col_start - x1)
    line_c2 = min(ink.shape[1], line.col_end - x1)
    line_mask = np.zeros_like(ink)
    line_mask[line_r1:line_r2, line_c1:line_c2] = True

    erase = (ink & ~line_mask) if keep_line else ink
    zone[erase] = (255, 255, 255)
    arr[y1:y2, x1:x2] = zone
    return Image.fromarray(arr)
