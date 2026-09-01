r"""Turn a raw, physically-scanned cheque photo into a normalized, upright
cheque image ready for region-based detectors (Phase 4's signature check).

Pure functions only: no Azure, no network, no I/O beyond reading the one
input path (and, optionally, writing debug dumps behind an explicit flag).
Pipeline order is fixed and matters:

    isolate -> rotate -> deskew -> normalize to a fixed canvas

Every step either succeeds or raises a typed `ChequeImagePrepError` — there
is no best-guess fallback. A silently wrong crop or rotation would produce a
silently wrong signature verdict downstream, which is the exact failure mode
this project exists to remove (see models.py's module docstring on
provenance). Callers MUST catch these at the pipeline boundary and map them
onto the existing ParseStatus/RuleStatus vocabulary rather than letting them
propagate — see `normalize.normalize_signature(..., ambiguous_reason=...)`
and `rules.check_signature`'s handling of `ParseStatus.AMBIGUOUS`, which
exist specifically to receive an `OrientationIndeterminate` from here.

Background: Phase 0b established that every cheque in the 20260820... scan
batch is rotated 90 degrees with no consistent leading edge, so nothing
downstream can assume orientation. Phase 2b found that the batch's rotation
direction cannot be told apart via generic ink density or texture, because
several cheque templates carry a decorative border that is denser and more
"texty" than the real MICR line. The one signal that IS specific to MICR is
that E-13B (the MICR font) is fixed-pitch: 8 characters per inch. A genuine
MICR line produces a strong autocorrelation peak at that exact pitch (plus
its 2x harmonic); a decorative border does not reproduce both. The pitch is
derived from each image's own DPI tag rather than hardcoded, so the detector
degrades gracefully (or refuses) on a batch scanned at a different
resolution instead of confidently picking the wrong rotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# errors — fail loudly, never a best-guess crop/rotation
# ---------------------------------------------------------------------------


class ChequeImagePrepError(Exception):
    """Base for every typed failure raised by this module."""


class NoChequeFound(ChequeImagePrepError):
    """The isolation step found no ink object against the page background."""


class OrientationIndeterminate(ChequeImagePrepError):
    """The MICR-pitch rotation detector could not confidently pick CW/CCW.

    Carries the diagnostic scores so a caller can log or persist *why* this
    file was refused, not just that it was.
    """

    def __init__(self, message: str, *, top_score: float, bottom_score: float,
                dpi: float, dpi_source: str):
        super().__init__(message)
        self.top_score = top_score
        self.bottom_score = bottom_score
        self.dpi = dpi
        self.dpi_source = dpi_source


# ---------------------------------------------------------------------------
# provenance: what the rotation step decided, and how confident it was
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RotationDecision:
    """Persist this alongside the signature verdict (Amendment 2) — a
    reviewer six months from now should be able to see whether a signature
    result rested on a confident rotation call or a marginal one."""

    direction: str              # "CW" or "CCW"
    fundamental_score: float    # normalized autocorrelation peak at the MICR pitch
    harmonic_score: float       # normalized autocorrelation peak at 2x the pitch
    fundamental_lag_px: int
    dpi: float
    dpi_source: str             # "jfif" or "fallback"
    confident: bool
    source: str = "detector"    # "detector" or "operator_override"

    def as_dict(self) -> dict:
        return {
            "direction": self.direction,
            "fundamental_score": round(self.fundamental_score, 4),
            "harmonic_score": round(self.harmonic_score, 4),
            "fundamental_lag_px": self.fundamental_lag_px,
            "dpi": self.dpi,
            "dpi_source": self.dpi_source,
            "confident": self.confident,
            "source": self.source,
        }


@dataclass(frozen=True)
class PreparedCheque:
    image: Image.Image
    rotation: RotationDecision
    skew_angle_deg: float
    source_crop_size: tuple[int, int]   # (width, height) after isolate + rotate
    canvas_size: tuple[int, int]        # (width, height) of the final image
    scale: tuple[float, float]          # (sx, sy) applied to reach canvas_size


# ---------------------------------------------------------------------------
# named constants — every threshold here was validated against the full
# 22-file batch (see chequemate_phase3 STOP report), not hand-waved.
# ---------------------------------------------------------------------------

# isolate(): a column/row is "part of the cheque" once its ink count clears
# this floor (as a fraction of the opposite dimension). Low enough to survive
# a faint scan, high enough to ignore stray specks in the blank margin.
ISOLATION_INK_FLOOR_FRAC = 0.002
ISOLATION_MIN_FLOOR_PX = 3

# rotate(): chroma filter that separates real (near-black) ink from colored
# decorative borders/watermarks. Channel-spread (max-min) below this value is
# treated as "achromatic enough to be ink".
NEUTRAL_CHROMA_MAX = 20

# the band (as a fraction of candidate height) searched at each end for the
# MICR line / competing decorative border.
ROTATION_BAND_FRAC = 0.20

# E-13B (MICR) is fixed-pitch at this many characters per inch. The expected
# autocorrelation peak in pixels is DPI / MICR_CHARS_PER_INCH - a physical
# constant, not a fitted one. See Amendment 1.
MICR_CHARS_PER_INCH = 8.0
DEFAULT_DPI = 300.0
# search window around the expected pitch, expressed as a fraction of the
# pitch rather than a fixed pixel count, so a differently-scanned batch (a
# rescan at 200 or 600 DPI) gets its own correctly-scaled window instead of
# silently reusing pixel bounds tuned for 300 DPI.
MICR_PITCH_TOLERANCE = 0.06

# a band whose chroma-filtered ink density is below this is too sparse for
# its autocorrelation to mean anything - without this floor, a near-blank
# band can produce a spuriously high normalized peak from noise alone.
ROTATION_MIN_DENSITY = 0.015

# combined_score = fundamental + 0.5 * harmonic must clear this to be trusted
# at all. A genuine MICR line reinforces at both the pitch and its harmonic;
# an incidental match from a decorative border rarely does.
ROTATION_MIN_SCORE = 0.20

# the winning band's score must also beat the losing band's by this ratio.
# Clearing the floor alone isn't enough: on some templates a decorative
# border's incidental periodicity lands close enough to the true MICR pitch
# that it can win by a hair, which is a confident-WRONG call - worse than an
# honest refusal. Requiring a real margin trades a few extra
# OrientationIndeterminate results for zero confident-wrong ones, which is
# the correct trade per this project's core design rule (models.py): never
# return a best-guess when the alternative is failing loudly.
ROTATION_MIN_MARGIN = 1.5

# deskew(): small-angle search range/step (degrees). Cheques arrive already
# corrected to the right quadrant by rotate() - this only fixes crooked
# feeding, not gross misorientation.
DESKEW_MAX_ANGLE_DEG = 5.0
DESKEW_STEP_DEG = 0.25

# normalize(): fixed output canvas. Chosen from the median of this batch's
# post-isolate-post-rotate dimensions (width 1733-2188px, height 677-912px
# at 300 DPI) - not a magic number, but also not a universal constant: a
# batch scanned at a different resolution should recalibrate this.
CANVAS_WIDTH = 1860
CANVAS_HEIGHT = 780


# ---------------------------------------------------------------------------
# step 0: binarization helper (shared by isolate + rotate + deskew)
# ---------------------------------------------------------------------------


def otsu_threshold(gray: np.ndarray) -> float:
    """Otsu's method, pure numpy. Picks the threshold that best separates
    ink from background for THIS image - not a fixed brightness cutoff,
    since scan exposure varies across the batch.

    A pure bimodal 0/255 image (no JPEG noise, no antialiasing - e.g. a
    cleanly-drawn synthetic fixture) makes the between-class variance flat
    across every threshold between the two populated bins, not peaked at
    one. Taking the first tied maximum would land on t=0, where `gray < 0`
    is never true and isolate() would see no ink at all. Taking the
    midpoint of the tied plateau instead always lands between the two
    clusters, matching the real-scan case (which has no plateau, so the
    midpoint of a single-point 'plateau' is just that point).
    """
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)
    sum_bg = weight_bg = 0.0
    var_between = np.zeros(256)
    for t in range(256):
        weight_bg += hist[t]
        weight_fg = total - weight_bg
        if weight_bg == 0 or weight_fg == 0:
            continue
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var_between[t] = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    best_var = var_between.max()
    if best_var <= 0:
        return 0.0
    tied = np.where(var_between >= best_var - 1e-9)[0]
    return float(tied.mean())


# ---------------------------------------------------------------------------
# step 1: isolate
# ---------------------------------------------------------------------------


def isolate(rgb: np.ndarray) -> np.ndarray:
    """Column/row-projection decomposition: crop to the one dense ink object
    against the blank-page background. Raises NoChequeFound on a blank page
    (or anything with no ink clearing the floor)."""
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    thresh = otsu_threshold(gray)
    ink = gray < thresh

    col_floor = max(ISOLATION_MIN_FLOOR_PX, ISOLATION_INK_FLOOR_FRAC * ink.shape[0])
    row_floor = max(ISOLATION_MIN_FLOOR_PX, ISOLATION_INK_FLOOR_FRAC * ink.shape[1])
    cols = np.where(ink.sum(axis=0) > col_floor)[0]
    rows = np.where(ink.sum(axis=1) > row_floor)[0]
    if cols.size == 0 or rows.size == 0:
        raise NoChequeFound(
            "no ink cleared the isolation floor - this looks like a blank "
            "or near-blank page, not a cheque")

    top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return rgb[top:bottom, left:right]


# ---------------------------------------------------------------------------
# step 2: rotate (Phase 2b MICR-pitch detector)
# ---------------------------------------------------------------------------


def _get_dpi(image: Image.Image) -> tuple[float, str]:
    dpi = image.info.get("dpi")
    if dpi and dpi[0]:
        return float(dpi[0]), "jfif"
    return DEFAULT_DPI, "fallback"


def _detrend(sig: np.ndarray, window: int = 41) -> np.ndarray:
    if window >= sig.size:
        return sig - sig.mean()
    kernel = np.ones(window) / window
    baseline = np.convolve(sig, kernel, mode="same")
    return sig - baseline


def _periodicity_score(mask_band: np.ndarray, lo: int, hi: int,
                       harm_lo: int, harm_hi: int) -> tuple[float, float, float, int]:
    """Returns (combined, fundamental, harmonic, fundamental_lag) for the
    column-wise ink projection of one chroma-filtered near-black band."""
    density = float(mask_band.mean())
    if density < ROTATION_MIN_DENSITY:
        return 0.0, 0.0, 0.0, -1

    projection = mask_band.sum(axis=0).astype(np.float64)
    if projection.std() < 1e-6:
        return 0.0, 0.0, 0.0, -1

    sig = _detrend(projection)
    sig = sig - sig.mean()
    n = sig.size
    fsize = 1
    while fsize < 2 * n:
        fsize *= 2
    f = np.fft.rfft(sig, fsize)
    acf = np.fft.irfft(f * np.conj(f), fsize)[:n]
    acf /= acf[0] + 1e-9

    lo, hi = max(lo, 1), min(hi, n - 1)
    if lo >= hi:
        return 0.0, 0.0, 0.0, -1
    fund_window = acf[lo:hi + 1]
    fund_lag = int(np.argmax(fund_window)) + lo
    fund_score = float(fund_window.max())

    harm_lo, harm_hi = max(harm_lo, 1), min(harm_hi, n - 1)
    harm_score = float(acf[harm_lo:harm_hi + 1].max()) if harm_lo < harm_hi else 0.0

    combined = fund_score + 0.5 * max(harm_score, 0.0)
    return combined, fund_score, harm_score, fund_lag


def _near_black_mask(rgb: np.ndarray, thresh: float) -> np.ndarray:
    rgb16 = rgb.astype(np.int16)
    chroma = rgb16.max(axis=2) - rgb16.min(axis=2)
    value = rgb16.max(axis=2)
    return (value < thresh) & (chroma < NEUTRAL_CHROMA_MAX)


def detect_rotation(isolated_rgb: np.ndarray, dpi: float,
                    dpi_source: str) -> RotationDecision:
    """Decide CW vs CCW for a 90-degree-rotated cheque crop.

    Rotates both ways, profiles the chroma-filtered near-black ink in the
    top/bottom 20% bands of each candidate, and picks whichever orientation
    puts a genuine MICR-pitch signal (fundamental + 2x harmonic) at the
    bottom. Raises OrientationIndeterminate rather than guessing when
    neither band clears the confidence floor.
    """
    target_pitch = dpi / MICR_CHARS_PER_INCH
    lo = round(target_pitch * (1 - MICR_PITCH_TOLERANCE))
    hi = round(target_pitch * (1 + MICR_PITCH_TOLERANCE))
    harm_lo = round(2 * target_pitch * (1 - MICR_PITCH_TOLERANCE))
    harm_hi = round(2 * target_pitch * (1 + MICR_PITCH_TOLERANCE))

    gray = np.asarray(Image.fromarray(isolated_rgb).convert("L"))
    thresh = otsu_threshold(gray)

    # candidate A: rotate 90 CCW (np.rot90 default direction)
    cand = np.rot90(isolated_rgb, k=1, axes=(0, 1))
    mask = _near_black_mask(cand, thresh)
    band_h = int(mask.shape[0] * ROTATION_BAND_FRAC)

    top_combined, top_fund, top_harm, top_lag = _periodicity_score(
        mask[:band_h], lo, hi, harm_lo, harm_hi)
    bot_combined, bot_fund, bot_harm, bot_lag = _periodicity_score(
        mask[mask.shape[0] - band_h:], lo, hi, harm_lo, harm_hi)

    if bot_combined >= ROTATION_MIN_SCORE and bot_combined > top_combined * ROTATION_MIN_MARGIN:
        return RotationDecision(
            direction="CCW", fundamental_score=bot_fund, harmonic_score=bot_harm,
            fundamental_lag_px=bot_lag, dpi=dpi, dpi_source=dpi_source, confident=True)
    if top_combined >= ROTATION_MIN_SCORE and top_combined > bot_combined * ROTATION_MIN_MARGIN:
        return RotationDecision(
            direction="CW", fundamental_score=top_fund, harmonic_score=top_harm,
            fundamental_lag_px=top_lag, dpi=dpi, dpi_source=dpi_source, confident=True)

    raise OrientationIndeterminate(
        f"neither orientation cleared the confidence floor "
        f"({ROTATION_MIN_SCORE}): top={top_combined:.3f} bottom={bot_combined:.3f} "
        f"at target MICR pitch {target_pitch:.1f}px (DPI={dpi:.0f}, "
        f"source={dpi_source})",
        top_score=top_combined, bottom_score=bot_combined,
        dpi=dpi, dpi_source=dpi_source)


def _measure_band(isolated_rgb: np.ndarray, dpi: float, direction: str
                  ) -> tuple[float, float, int]:
    """The real fundamental/harmonic/lag for a SPECIFIC chosen direction's
    bottom band - used to give an operator override honest provenance
    (the real measured score, typically just below ROTATION_MIN_MARGIN,
    not a fabricated placeholder) rather than zeros."""
    target_pitch = dpi / MICR_CHARS_PER_INCH
    lo = round(target_pitch * (1 - MICR_PITCH_TOLERANCE))
    hi = round(target_pitch * (1 + MICR_PITCH_TOLERANCE))
    harm_lo = round(2 * target_pitch * (1 - MICR_PITCH_TOLERANCE))
    harm_hi = round(2 * target_pitch * (1 + MICR_PITCH_TOLERANCE))

    gray = np.asarray(Image.fromarray(isolated_rgb).convert("L"))
    thresh = otsu_threshold(gray)
    k = 1 if direction == "CCW" else -1
    cand = np.rot90(isolated_rgb, k=k, axes=(0, 1))
    mask = _near_black_mask(cand, thresh)
    band_h = int(mask.shape[0] * ROTATION_BAND_FRAC)
    _, fund, harm, lag = _periodicity_score(
        mask[mask.shape[0] - band_h:], lo, hi, harm_lo, harm_hi)
    return fund, harm, lag


def rotate(isolated_rgb: np.ndarray, dpi: float, dpi_source: str, *,
          override: str | None = None) -> tuple[np.ndarray, RotationDecision]:
    """`override` (Amendment/Correction B): an operator-confirmed direction
    for a file the detector refused on (OrientationIndeterminate). This
    bypasses the confidence gate entirely - it does NOT lower the bar the
    detector itself must clear, it substitutes a human's judgment for the
    detector's refusal on that one file. The detector's own refusal is
    never silenced by this: callers are expected to have already caught
    OrientationIndeterminate and recorded it (see
    scripts/apply_rotation_override.py) before calling with an override.
    """
    if override is not None:
        if override not in ("CW", "CCW"):
            raise ValueError(f"rotation override must be 'CW' or 'CCW', got {override!r}")
        fund, harm, lag = _measure_band(isolated_rgb, dpi, override)
        decision = RotationDecision(
            direction=override, fundamental_score=fund, harmonic_score=harm,
            fundamental_lag_px=lag, dpi=dpi, dpi_source=dpi_source,
            confident=True, source="operator_override")
    else:
        decision = detect_rotation(isolated_rgb, dpi, dpi_source)

    k = 1 if decision.direction == "CCW" else -1
    rotated = np.rot90(isolated_rgb, k=k, axes=(0, 1))
    return rotated, decision


# ---------------------------------------------------------------------------
# step 3: deskew
# ---------------------------------------------------------------------------


def _deskew_angle(rgb: np.ndarray) -> float:
    """Small-angle search: the correctly-deskewed angle maximizes the
    variance of the horizontal ink-row projection (crisp text lines produce
    sharp peaks/troughs; skew blurs them together)."""
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    thresh = otsu_threshold(gray)
    ink_img = Image.fromarray((gray < thresh).astype(np.uint8) * 255)

    best_angle, best_score = 0.0, -1.0
    angle = -DESKEW_MAX_ANGLE_DEG
    while angle <= DESKEW_MAX_ANGLE_DEG + 1e-9:
        rotated = ink_img.rotate(angle, resample=Image.BILINEAR,
                                 expand=False, fillcolor=0)
        arr = np.asarray(rotated) > 127
        score = float(arr.sum(axis=1).astype(np.float64).var())
        if score > best_score:
            best_score, best_angle = score, angle
        angle += DESKEW_STEP_DEG
    return best_angle


def deskew(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    angle = _deskew_angle(rgb)
    if angle == 0.0:
        return rgb, 0.0
    image = Image.fromarray(rgb).rotate(
        angle, resample=Image.BILINEAR, expand=True, fillcolor=(255, 255, 255))
    return np.asarray(image), angle


# ---------------------------------------------------------------------------
# step 4: normalize to a fixed canvas
# ---------------------------------------------------------------------------


def normalize_canvas(rgb: np.ndarray, size: tuple[int, int] = (CANVAS_WIDTH, CANVAS_HEIGHT)
                     ) -> tuple[np.ndarray, tuple[float, float]]:
    """Scale (independently in x and y) onto a fixed canvas so a downstream
    region mask (e.g. 'signature lives in this rectangle') lands at the same
    normalized coordinates regardless of the input's aspect ratio. This
    deliberately does NOT pad-to-fit: raw components vary from 750x1840 to
    985x2232, and padding would leave the signature zone drifting between
    images instead of landing at a fixed place."""
    h, w = rgb.shape[:2]
    target_w, target_h = size
    image = Image.fromarray(rgb).resize((target_w, target_h), resample=Image.LANCZOS)
    return np.asarray(image), (target_w / w, target_h / h)


def detect_rotation_only(path: str | Path) -> RotationDecision:
    """isolate() -> rotate() only - NOT deskew()/normalize_canvas(). For a
    caller that needs the rotation direction alone to correct a field crop
    taken directly from the ORIGINAL image using Document Intelligence's
    own polygon (which is defined in that original image's pixel space).
    `prepare_cheque_image()`'s full pipeline deliberately isn't reused here:
    deskew's `expand=True` and normalize_canvas's resize both change pixel
    coordinates in ways that would break correspondence with DI's polygon,
    and are unnecessary just to learn CW vs CCW.

    Raises NoChequeFound / OrientationIndeterminate exactly as
    prepare_cheque_image() does - callers must catch these at the pipeline
    boundary, same as always (see this module's docstring).
    """
    path = Path(path)
    image = Image.open(path).convert("RGB")
    dpi, dpi_source = _get_dpi(image)
    rgb = np.asarray(image)
    isolated = isolate(rgb)
    _, decision = rotate(isolated, dpi, dpi_source)
    return decision


# ---------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------


def prepare_cheque_image(path: str | Path, *, debug_dir: Path | None = None,
                         rotation_override: str | None = None) -> PreparedCheque:
    """isolate -> rotate -> deskew -> normalize. Raises ChequeImagePrepError
    subclasses; never returns a best-guess result.

    `debug_dir` (opt-in only) writes one PNG per stage into an
    already-gitignored directory - never enabled by default.

    `rotation_override` (Correction B): bypasses the rotation detector's
    confidence gate for one file with an operator-confirmed direction. See
    scripts/apply_rotation_override.py - this parameter alone doesn't
    constitute "human confirmed it", it's the mechanism that consumes that
    confirmation once recorded.
    """
    path = Path(path)
    image = Image.open(path).convert("RGB")
    dpi, dpi_source = _get_dpi(image)
    rgb = np.asarray(image)

    isolated = isolate(rgb)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(isolated).save(debug_dir / f"{path.stem}_1_isolated.png")

    rotated, rotation = rotate(isolated, dpi, dpi_source, override=rotation_override)
    if debug_dir is not None:
        Image.fromarray(rotated).save(debug_dir / f"{path.stem}_2_rotated.png")

    deskewed, skew_angle = deskew(rotated)
    if debug_dir is not None:
        Image.fromarray(deskewed).save(debug_dir / f"{path.stem}_3_deskewed.png")

    source_crop_size = (deskewed.shape[1], deskewed.shape[0])  # (width, height)
    canvas, scale = normalize_canvas(deskewed)
    if debug_dir is not None:
        Image.fromarray(canvas).save(debug_dir / f"{path.stem}_4_normalized.png")

    return PreparedCheque(
        image=Image.fromarray(canvas),
        rotation=rotation,
        skew_angle_deg=skew_angle,
        source_crop_size=source_crop_size,
        canvas_size=(CANVAS_WIDTH, CANVAS_HEIGHT),
        scale=scale,
    )
