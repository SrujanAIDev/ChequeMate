r"""Run TrOCR field-level secondary verification against every existing
reports/cheques.json record that has a recoverable raw Azure response
(raw/<name>.json, backfilled by scripts/backfill_raw_responses.py for
records that predate --save-raw) and its original source image
(cheques/<source_file>).

Strictly additive: this script ONLY sets a record's `ocr_verifications`
field. It never touches `verdict`, `rules`, `payee`, `amount_words`, or any
other field - TrOCR is a secondary observation, not a re-validation.
Compare with scripts/revalidate.py, which re-derives the deterministic
fields/rules from raw_values and is a separate concern entirely.

A record is skipped (not overwritten) if it already has ocr_verifications,
unless --force is passed. A record with no raw response or no source image
on disk is reported and skipped - never guessed at.

Uses the REAL local TrOCR model (chequemate.trocr_adapter.TransformersTrOCRClient),
loaded once and reused for the whole batch. Requires the model weights to
already be downloaded (see docs/trocr_verification.md's local setup step).

Run:
    python scripts/apply_trocr_verification.py
    python scripts/apply_trocr_verification.py --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

from PIL import Image  # noqa: E402

from chequemate import ocr_verify, report  # noqa: E402
from chequemate.extract import first_document, to_normalized  # noqa: E402
from chequemate.trocr_adapter import TransformersTrOCRClient  # noqa: E402

RAW_DIR = ROOT / "raw"
CHEQUES_DIR = ROOT / "cheques"


def _backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[backup] {path.name} -> {bak.name}")
    else:
        print(f"[backup] {bak.name} already exists, leaving pre-verification "
             f"snapshot untouched")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="re-run TrOCR verification even for records that already "
                         "have ocr_verifications data")
    ap.add_argument("--local-files-only", action="store_true", default=True,
                    help="never attempt a network call for model weights "
                         "(default: on - weights must already be cached locally)")
    args = ap.parse_args(argv)

    report.ensure_report_directory()
    _backup(report.CHEQUES_JSON)

    records = report.load_records()
    by_id = {r["record_id"]: i for i, r in enumerate(records)}

    trocr_cfg = ocr_verify.TrOCRVerificationConfig(
        enabled=True, local_files_only=args.local_files_only)
    print("Loading TrOCR model (once for this whole run)...")
    client = TransformersTrOCRClient(
        model_name_or_path=trocr_cfg.model_name_or_path,
        model_revision=trocr_cfg.model_revision,
        device_preference=trocr_cfg.device_preference,
        max_new_tokens=trocr_cfg.max_new_tokens,
        local_files_only=trocr_cfg.local_files_only,
        inference_timeout_s=trocr_cfg.inference_timeout_s)

    verified = skipped_already_done = skipped_no_raw = skipped_no_image = errors = 0

    for record in records:
        record_id = record["record_id"]
        if record.get("ocr_verifications") and not args.force:
            skipped_already_done += 1
            continue

        stem = Path(record["source_file"]).stem
        raw_path = RAW_DIR / (stem + ".json")
        if not raw_path.is_file():
            print(f"[{record_id}] SKIP - no raw response at {raw_path.relative_to(ROOT)}")
            skipped_no_raw += 1
            continue

        image_path = CHEQUES_DIR / record["source_file"]
        if not image_path.is_file():
            print(f"[{record_id}] SKIP - source image not found: "
                 f"{record['source_file']}")
            skipped_no_image += 1
            continue

        # Reuse a standing operator rotation confirmation (see
        # scripts/apply_rotation_override.py) instead of re-running the
        # detector, which already refused on these specific files once -
        # a human's confirmed direction must never be silently re-refused.
        rotation_meta = record.get("rotation") or {}
        rotation_override = (rotation_meta.get("direction")
                             if rotation_meta.get("source") == "operator_override"
                             else None)

        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            cheque = to_normalized(first_document(raw), source_id=str(image_path))
            with Image.open(image_path) as image:
                fv = ocr_verify.verify_cheque_fields(
                    raw, image, cheque, trocr_cfg, record_id=record_id,
                    trocr_client=client, source_path=image_path,
                    rotation_override=rotation_override)
        except Exception as exc:
            print(f"[{record_id}] ERROR - {type(exc).__name__}: {exc}", file=sys.stderr)
            errors += 1
            continue

        idx = by_id[record_id]
        records[idx]["ocr_verifications"] = report._ocr_verifications_to_dict(fv)
        report.save_records(records)

        summary = {name: v.comparison.status.value for name, v in fv.items()}
        report.append_audit_event({
            "timestamp": datetime.now().astimezone().isoformat(),
            "event": "trocr_verification_backfill",
            "record_id": record_id,
            "source_file": record["source_file"],
            "ocr_verification_summary": summary,
        })
        print(f"[{record_id}] {record['source_file']}: {summary}")
        verified += 1

    report.regenerate_report()
    from chequemate import verification_report
    verification_report.regenerate_verification_report(
        trocr_model_id=trocr_cfg.model_name_or_path,
        trocr_model_revision=trocr_cfg.model_revision)

    print()
    print(f"Verified            : {verified}")
    print(f"Already done (skip) : {skipped_already_done}")
    print(f"No raw response     : {skipped_no_raw}")
    print(f"No source image     : {skipped_no_image}")
    print(f"Errors              : {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
