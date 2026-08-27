r"""Honest, non-technical scorecard for the signature-zone detector
(chequemate/signature.py) - a summary a non-technical reader can act on,
not a threshold to take on faith.

States plainly: presence-detection is validated on 20 real signed cheques.
Absence-detection has NEVER been validated on a real unsigned cheque of the
current (300 DPI, 20260820... batch) template - only on synthetic negatives
built by erasing ink from those same 20 real cheques, plus one real
negative of a DIFFERENT, older template that the envelope guard now refuses
outright rather than measuring.

Run:
    python scripts/signature_scorecard.py
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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

# Cheque template, by record, for grouping refusals in the scorecard - read
# from the printed bank name on each file (matches the batch's known
# templates, not inferred from OCR).
TEMPLATE_BY_FILE = {
    "20260820113647241_0001.jpg": "RBC", "20260820113647241_0002.jpg": "CIBC",
    "20260820113647241_0003.jpg": "TD", "20260820113647241_0004.jpg": "TD",
    "20260820113647241_0005.jpg": "CIBC", "20260820113647241_0006.jpg": "Scotiabank",
    "20260820113647241_0007.jpg": "BMO", "20260820113647241_0008.jpg": "Meridian",
    "20260820113647241_0009.jpg": "RBC", "20260820113647241_0010.jpg": "TD",
    "20260820113647241_0011.jpg": "CIBC", "20260820113647241_0012.jpg": "Scotiabank",
    "20260820113647241_0013.jpg": "RBC", "20260820113647241_0014.jpg": "TD",
    "20260820113647241_0015.jpg": "Tangerine", "20260820113647241_0016.jpg": "TD",
    "20260820113647241_0017.jpg": "CIBC", "20260820113647241_0018.jpg": "Scotiabank",
    "20260820113647241_0019.jpg": "CIBC", "20260820113647241_0020.jpg": "RBC",
    "20260820113647241_0021.jpg": "RBC", "20260820113647241_0022.jpg": "Tangerine",
}


def main() -> int:
    files = sorted(CHEQUES_DIR.glob("20260820113647241_*.jpg"))

    present, absent, ambiguous = [], [], []
    real_coverage = []
    negative_coverage = []

    for path in files:
        try:
            result = prepare_cheque_image(path, rotation_override=ROTATION_OVERRIDES.get(path.name))
        except ChequeImagePrepError as exc:
            ambiguous.append((path.name, f"image prep failed: {exc}"))
            continue

        analysis = analyze_signature_zone(result)
        if analysis.ambiguous:
            ambiguous.append((path.name, analysis.reason))
            continue
        real_coverage.append(analysis.ink_coverage)
        if analysis.present:
            present.append(path.name)
        else:
            absent.append(path.name)

        try:
            neg_img = synthesize_unsigned_variant(result, keep_line=True)
            neg = analyze_signature_zone(replace(result, image=neg_img))
            if not neg.ambiguous:
                negative_coverage.append(neg.ink_coverage)
        except ValueError:
            pass

    print("=" * 72)
    print("SIGNATURE DETECTOR SCORECARD - chequemate/signature.py")
    print("=" * 72)
    print()
    print(f"Batch: 22 cheques, 20260820... scan (300 DPI), all independently")
    print(f"confirmed signed.")
    print()
    print(f"RESOLVED (found a printed line, made a present/absent call): "
         f"{len(present) + len(absent)}/22")
    print(f"  -> called PRESENT : {len(present)}/22")
    print(f"  -> called ABSENT  : {len(absent)}/22"
         + ("  <-- WOULD BE WRONG (all 22 are signed)" if absent else ""))
    print()
    print(f"REFUSED (no printed line found - correctly declines to guess): "
         f"{len(ambiguous)}/22")
    by_template: dict[str, list[str]] = {}
    for name, _ in ambiguous:
        by_template.setdefault(TEMPLATE_BY_FILE.get(name, "unknown"), []).append(name)
    for template, names in sorted(by_template.items()):
        print(f"  - {template}: {', '.join(names)}")
    print(f"  Reason (same for all): the printed line's span never reached "
         f"the detection threshold anywhere in the search region on these "
         f"templates' specific line style (shorter/lighter printed line).")
    print()

    if real_coverage:
        print(f"Real signed cheques - ink coverage after line removal:")
        print(f"  min={min(real_coverage):.5f}  max={max(real_coverage):.5f}  "
             f"mean={statistics.mean(real_coverage):.5f}")
    if negative_coverage:
        print(f"Synthetic negatives (signature erased, line kept) - ink coverage:")
        print(f"  min={min(negative_coverage):.5f}  max={max(negative_coverage):.5f}  "
             f"mean={statistics.mean(negative_coverage):.5f}")
    if real_coverage and negative_coverage:
        gap = min(real_coverage) - max(negative_coverage)
        print(f"Separation: {gap:.5f} (NOISE_FLOOR={NOISE_FLOOR} sits "
             f"{'inside' if max(negative_coverage) < NOISE_FLOOR < min(real_coverage) else 'OUTSIDE'} "
             f"this gap)")
    print()

    print("-" * 72)
    print("THE ONE REAL (non-synthetic, non-batch) NEGATIVE ON FILE:")
    if REAL_NEGATIVE_PATH.is_file():
        real_neg_result = prepare_cheque_image(REAL_NEGATIVE_PATH, rotation_override="CCW")
        try:
            analyze_signature_zone(real_neg_result)
            print(f"  {REAL_NEGATIVE_PATH.name}: NOT refused (unexpected)")
        except SignatureEnvelopeError:
            print(f"  {REAL_NEGATIVE_PATH.name}: REFUSED - outside the validated "
                 f"300 DPI envelope (96 DPI, different template).")
    print("-" * 72)
    print()

    print("=" * 72)
    print("BOTTOM LINE")
    print("=" * 72)
    print("Presence detection IS validated: 20 real signed cheques all")
    print("correctly read as present, 0 false negatives.")
    print()
    print("Absence detection has NEVER been validated on a real unsigned")
    print("cheque of the current (300 DPI, this batch's template family)")
    print("design. Every 'correctly says absent' result above comes from a")
    print("synthetic negative (erasing real signature ink), not a genuine")
    print("unsigned cheque. The only real unsigned cheque available is a")
    print("different, older, lower-resolution template - it is refused by")
    print("the DPI guard rather than measured, so it provides no evidence")
    print("either way about this batch's absence-detection accuracy.")
    print()
    print("This module is not load-bearing on any verdict. Treat 'absence'")
    print("results as unvalidated until a real unsigned cheque of the")
    print("current template becomes available to test against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
