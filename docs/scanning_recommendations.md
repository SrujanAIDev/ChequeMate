# Scanner-side recommendation: fix the scan, not just the code

**Status:** operational finding from Phase 0b/2b/3 of the image-preparation work
(`chequemate/imageprep.py`). Not implemented in code — this is a process change for
whoever operates the scanner, written down so it doesn't get lost in chat history.

## What was found

Every cheque in the `20260820113647241_*.jpg` batch (22/22) arrived rotated 90 degrees
from upright, with no consistent leading edge — Phase 2b had to build a per-image
rotation detector rather than assume a single fixed direction. On top of that, cheque
content is anchored flush to the scanned page's top edge on 19 of the 22 files (the
other 3 are within 51px of it), which is a scanner registration issue, not a content
problem — cheques that vary in size or placement are being fed with no margin against
the leading edge.

Both are scan-time artifacts, not defects in the cheques themselves.

## Why this is worth fixing at the source

`imageprep.py`'s rotation detector works (see the Phase 2b/3 STOP report), but it isn't
free: it exists only because the scan direction isn't known in advance, and it can't
resolve every file. Under the current MICR-pitch-based detector, calibrated to real
physics (E-13B's fixed 8-characters-per-inch pitch, derived from each image's own DPI
tag) rather than fitted pixel thresholds, **3 of the 22 files (`_0001`, `_0006`,
`_0011`) still can't be resolved with confidence** — their decorative cheque-template
border has an incidental periodicity close enough to the true MICR pitch that the
detector correctly refuses to guess rather than risk a silently wrong crop. Those 3
need a human to confirm orientation by eye before signature detection (Phase 4) can run
on them at all.

None of this class of failure — the rotation-detection code, its edge cases, or the
top-edge registration risk — can exist if the source scan is already upright.

## The recommendation

1. **Feed cheques landscape**, matching the cheque's own natural orientation, so no
   90-degree rotation correction is ever needed.
2. **Position the cheque away from the leading edge** of the scan bed/feeder, so content
   isn't clipped or flush against the registration edge.

This is a scanner operating-procedure change, not a software change — but it is worth
more than any code in `imageprep.py`: it would eliminate the entire rotation-correction
step, the top-edge registration risk, and the `_0001`/`_0006`/`_0011` class of
refuse-rather-than-guess failure, outright, for every future batch.

## If this can't be changed operationally

`imageprep.py`'s rotation detector will keep working as a fallback, but expect a small,
irreducible fraction of files per batch to come back as `OrientationIndeterminate` and
need a manual look — that's a property of some cheque templates' decorative borders,
not something a better detector will fully eliminate.
