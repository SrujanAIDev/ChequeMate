r"""Apply an operator-confirmed rotation for a cheque image the automated
detector correctly refused to guess on (imageprep.OrientationIndeterminate).

Phase 2b/3's MICR-pitch rotation detector is deliberately conservative: it
raises rather than picks a direction when the winning band doesn't clear a
required confidence margin over the losing one (see
chequemate/imageprep.py's ROTATION_MIN_MARGIN). That's correct behaviour —
a silently wrong rotation produces a silently wrong crop, which produces a
silently wrong signature verdict — but a permanent refusal with no
resolution path means that fraction of the batch can never be
signature-checked at all. This script is that resolution path: a human
states the correct rotation for one specific file, and imageprep.py's
`rotation_override` parameter consumes it in place of the detector's
refusal.

This must NEVER become "assume CCW because the rest of the batch is CCW" -
every entry in ROTATION_OVERRIDES below is a specific, individually-checked
file, not a batch-wide default. The detector's own refusal is preserved
(captured as `detector_note` on the record) rather than silenced by the
override - both are visible in the report.

Like apply_visual_verification.py, this never touches the original image
bytes: only reports/cheques.json's rotation provenance fields are updated.

Run:
    python scripts/apply_rotation_override.py

Idempotent: a record already carrying an "operator_rotation_override" review
is skipped on re-run.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chequemate.imageprep import OrientationIndeterminate, prepare_cheque_image  # noqa: E402
from chequemate import report  # noqa: E402

REVIEWER = "Claude (visual review, requested by srujan 2026-08-26)"
CHEQUES_DIR = ROOT / "cheques"

# Each entry is one specific file, individually visually confirmed - NOT a
# batch-wide "assume CCW" default. These three were confirmed during Phase
# 2b's full manual verification of all 22 files (every file in the batch
# was individually opened and read, not sampled): all three are genuinely
# CCW, same as the rest of the batch, but that's a finding about each file,
# not an assumption applied because of the other 19.
ROTATION_OVERRIDES: dict[str, str] = {
    "CHQ-20260824-0001": "CCW",
    "CHQ-20260824-0006": "CCW",
    "CHQ-20260824-0011": "CCW",
}


def _already_reviewed(reviews: list[dict], record_id: str) -> bool:
    prefix = "operator_rotation_override:"
    return any(rv.get("record_id") == record_id
              and str(rv.get("status", "")).startswith(prefix)
              for rv in reviews)


def _backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[backup] {path.name} -> {bak.name}")
    else:
        print(f"[backup] {bak.name} already exists, leaving pre-override "
             f"snapshot untouched")


def main() -> int:
    report.ensure_report_directory()
    _backup(report.CHEQUES_JSON)

    records = report.load_records()
    by_id = {r["record_id"]: i for i, r in enumerate(records)}
    reviews = report.load_reviews()
    applied = skipped = errors = 0

    for record_id, direction in ROTATION_OVERRIDES.items():
        if record_id not in by_id:
            print(f"WARNING: {record_id} not found in cheques.json, skipping")
            errors += 1
            continue
        if _already_reviewed(reviews, record_id):
            skipped += 1
            continue

        record = records[by_id[record_id]]
        image_path = CHEQUES_DIR / record["source_file"]
        if not image_path.is_file():
            print(f"WARNING: source image not found for {record_id}: "
                 f"{image_path}, skipping")
            errors += 1
            continue

        try:
            prepare_cheque_image(image_path)
            print(f"WARNING: {record_id} was NOT refused by the detector — "
                 f"override applied anyway, but this file may not have needed one")
            detector_note = "detector did not refuse this file"
        except OrientationIndeterminate as exc:
            detector_note = str(exc)

        try:
            prepared = prepare_cheque_image(image_path, rotation_override=direction)
        except Exception as exc:  # noqa: BLE001 - report and move to the next file
            print(f"WARNING: {record_id} — override rotation failed: {exc}")
            errors += 1
            continue

        rotation_dict = prepared.rotation.as_dict()
        rotation_dict["detector_note"] = detector_note

        record["rotation"] = rotation_dict
        record["rotation_direction"] = rotation_dict["direction"]
        record["rotation_confident"] = rotation_dict["confident"]
        records[by_id[record_id]] = record
        report.save_records(records)

        reviews.append({
            "record_id": record_id,
            "timestamp": datetime.now().astimezone().isoformat(),
            "reviewer": REVIEWER,
            "status": f"operator_rotation_override:{direction}",
            "note": f"detector refused ({detector_note}); operator confirmed "
                   f"{direction} by visual inspection of the source image",
        })
        report.save_reviews(reviews)
        report.append_audit_event({
            "timestamp": datetime.now().astimezone().isoformat(),
            "event": "operator_rotation_override",
            "record_id": record_id,
            "direction": direction,
            "detector_note": detector_note,
        })

        applied += 1
        print(f"[{record_id}] rotation confirmed {direction} "
             f"(detector had refused: {detector_note})")

    report.regenerate_report()

    print()
    print(f"Applied : {applied}")
    print(f"Already reviewed (skipped) : {skipped}")
    print(f"Errors : {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
