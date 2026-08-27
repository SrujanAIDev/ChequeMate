r"""Report the real separation between signed and (synthetically) unsigned
cheques before chequemate.signature.NOISE_FLOOR becomes load-bearing on a
verdict.

The 22-file 20260820... batch contains 22 positives and zero negatives (all
22 are genuinely signed) - so the "does the detector correctly say ABSENT"
claim cannot be validated on THIS batch's real data at all. Two separate
synthetic negative variants exist per real cheque, both by erasing real
signature ink out of a real, known-signed image:

  - keep_line=True:  the printed PER/signature line is left INTACT, only
                      the handwritten signature is erased. This is the case
                      that actually exercises line removal - if line
                      removal leaves residue, this variant is exactly where
                      it would show up as a false "present".
  - keep_line=False: everything in the zone is erased, including the line.
                      The easier case - there is no line survives.

There IS one real (non-synthetic) negative available in this repo:
cheques/cheque2.png.png, a much older/unrelated test fixture (96 DPI,
365x816 native - NOT part of the 22-file batch or its calibration) with a
visually-confirmed-blank signature line. It is reported here separately and
explicitly, never folded into the synthetic statistics: one sample is not a
distribution, and this script does NOT tune any constant against it. As of
this writing the detector gets it wrong (a confident false positive, not a
refusal) - see tests/test_signature.py's
test_real_unsigned_cheque_is_a_known_false_positive for the locked-in,
documented root cause.

Reported separately, never combined into one number: presence-detection is
validated on real data, absence-detection only on synthetic data plus this
one real (out-of-batch) sample, and those are not equally strong claims.

Run:
    python scripts/calibrate_signature_threshold.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataclasses import replace  # noqa: E402

from chequemate.imageprep import ChequeImagePrepError, prepare_cheque_image  # noqa: E402
from chequemate.signature import (  # noqa: E402
    NOISE_FLOOR,
    SignatureEnvelopeError,
    analyze_signature_zone,
    synthesize_unsigned_variant,
)

CHEQUES_DIR = ROOT / "cheques"
ROTATION_OVERRIDES = {
    "20260820113647241_0001.jpg": "CCW",
    "20260820113647241_0006.jpg": "CCW",
    "20260820113647241_0011.jpg": "CCW",
}
REAL_NEGATIVE_PATH = CHEQUES_DIR / "cheque2.png.png"
REAL_NEGATIVE_ROTATION = "CCW"  # auto-detector refuses (unrelated 96 DPI synthetic-image reason)


def _stats(label: str, values: list[float]) -> None:
    if not values:
        print(f"  {label}: (no data)")
        return
    print(f"  {label}: n={len(values)} min={min(values):.5f} "
         f"max={max(values):.5f} mean={statistics.mean(values):.5f} "
         f"median={statistics.median(values):.5f}")


def main() -> int:
    files = sorted(CHEQUES_DIR.glob("20260820113647241_*.jpg"))

    real_coverage: list[float] = []
    real_ambiguous: list[str] = []
    line_intact_coverage: list[float] = []
    line_intact_ambiguous: list[str] = []
    fully_blank_coverage: list[float] = []
    fully_blank_ambiguous: list[str] = []
    prep_failures: list[str] = []

    for path in files:
        try:
            result = prepare_cheque_image(
                path, rotation_override=ROTATION_OVERRIDES.get(path.name))
        except ChequeImagePrepError as exc:
            prep_failures.append(f"{path.name}: {exc}")
            continue

        real = analyze_signature_zone(result)
        if real.ambiguous:
            real_ambiguous.append(path.name)
        else:
            real_coverage.append(real.ink_coverage)

        try:
            line_intact_img = synthesize_unsigned_variant(result, keep_line=True)
            li = analyze_signature_zone(replace(result, image=line_intact_img))
            if li.ambiguous:
                line_intact_ambiguous.append(path.name)
            else:
                line_intact_coverage.append(li.ink_coverage)

            blank_img = synthesize_unsigned_variant(result, keep_line=False)
            fb = analyze_signature_zone(replace(result, image=blank_img))
            if fb.ambiguous:
                fully_blank_ambiguous.append(path.name)
            else:
                fully_blank_coverage.append(fb.ink_coverage)
        except ValueError:
            # synthesize_unsigned_variant needs a resolvable line - matches
            # real.ambiguous already being True for this file
            pass

    print(f"Prepared: {len(files) - len(prep_failures)}/{len(files)}")
    if prep_failures:
        for line in prep_failures:
            print(f"  FAILED: {line}")
    print()

    print("=== REAL DATA: presence detection (22 real, all known-signed) ===")
    _stats("real signed cheques (post line-removal ink coverage)", real_coverage)
    print(f"  ambiguous: {len(real_ambiguous)} {real_ambiguous}")
    print(f"  called present (coverage > NOISE_FLOOR={NOISE_FLOOR}): "
         f"{sum(1 for c in real_coverage if c > NOISE_FLOOR)}/{len(real_coverage)}")
    print()

    print("=== SYNTHETIC: absence detection, line INTACT (the case that "
         "actually tests line removal) ===")
    _stats("signature erased, printed line kept", line_intact_coverage)
    print(f"  ambiguous: {len(line_intact_ambiguous)} {line_intact_ambiguous}")
    print(f"  falsely called present (coverage > NOISE_FLOOR): "
         f"{sum(1 for c in line_intact_coverage if c > NOISE_FLOOR)}/{len(line_intact_coverage)}")
    print()

    print("=== SYNTHETIC: absence detection, fully blank (easier case) ===")
    _stats("everything in zone erased incl. line", fully_blank_coverage)
    print(f"  ambiguous: {len(fully_blank_ambiguous)} {fully_blank_ambiguous}")
    print(f"  falsely called present (coverage > NOISE_FLOOR): "
         f"{sum(1 for c in fully_blank_coverage if c > NOISE_FLOOR)}/{len(fully_blank_coverage)}")
    print()

    if real_coverage and line_intact_coverage:
        gap = min(real_coverage) - max(line_intact_coverage)
        print(f"=== SEPARATION (real signed floor - line-intact-negative "
             f"ceiling) ===")
        print(f"  {min(real_coverage):.5f} - {max(line_intact_coverage):.5f} "
             f"= {gap:.5f} headroom "
             f"({'positive - clusters separate' if gap > 0 else 'OVERLAP - clusters touch or cross'})")
        print(f"  NOISE_FLOOR={NOISE_FLOOR} sits "
             f"{'inside the gap' if max(line_intact_coverage) < NOISE_FLOOR < min(real_coverage) else 'OUTSIDE the gap - reconsider it'}")
    print()

    print("=== THE ONE REAL (non-synthetic) NEGATIVE - not tuned against ===")
    if REAL_NEGATIVE_PATH.is_file():
        result = prepare_cheque_image(REAL_NEGATIVE_PATH, rotation_override=REAL_NEGATIVE_ROTATION)
        try:
            analysis = analyze_signature_zone(result)
        except SignatureEnvelopeError as exc:
            print(f"  {REAL_NEGATIVE_PATH.name}: REFUSED (outside validated "
                 f"envelope) - {exc}")
        else:
            print(f"  {REAL_NEGATIVE_PATH.name}: ambiguous={analysis.ambiguous} "
                 f"present={analysis.present} coverage={analysis.ink_coverage:.5f} "
                 f"stroke_extent_frac={analysis.stroke_extent_frac:.4f}")
            if analysis.present:
                print(f"  -> FALSE POSITIVE (coverage clears NOISE_FLOOR={NOISE_FLOOR}) - "
                     f"NOT a coincidental refusal. Do not retune constants from this "
                     f"one sample; see CLAUDE.md's signature-detector section.")
    else:
        print(f"  {REAL_NEGATIVE_PATH.name} not found - skipping")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
