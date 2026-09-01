r"""One-shot, read-only-w.r.t.-cheques.json backfill: re-run Azure DI on every
already-processed cheque image whose full raw response (boundingRegions +
pages, needed for TrOCR field-region cropping) was never saved to raw/, and
save it there now.

This exists because --save-raw was only wired into run_batch.py recently -
every cheque processed before that has no recoverable polygon data. This
script does NOT touch reports/cheques.json, does NOT create or alter any
record, and does NOT re-run validate() - it exists purely to backfill
raw/<name>.json so ocr_verify.verify_cheque_fields() has something to crop
from. Existing raw/*.json files are left untouched (never re-fetched).

Costs one real Azure Document Intelligence call per backfilled file -
run deliberately, not as part of any routine/automated flow.

A record whose source_file has no corresponding file in cheques/ (e.g. it
was originally ingested from a bare replay .json that no longer exists on
disk) is reported and skipped - there is nothing to re-analyze.

Run:
    python scripts/backfill_raw_responses.py
"""

from __future__ import annotations

import json
import os
import ssl
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import certifi  # noqa: E402

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where())

from chequemate import report  # noqa: E402
from chequemate.extract import analyze_raw  # noqa: E402
from run_batch import load_dotenv  # noqa: E402

RAW_DIR = ROOT / "raw"
CHEQUES_DIR = ROOT / "cheques"


def main() -> int:
    load_dotenv()
    endpoint = os.getenv("AZURE_DI_ENDPOINT")
    key = os.getenv("AZURE_DI_KEY")
    if not endpoint or not key:
        print("Azure credentials not found (AZURE_DI_ENDPOINT / AZURE_DI_KEY).",
             file=sys.stderr)
        return 2

    records = report.load_records()
    existing_raw = {p.stem for p in RAW_DIR.glob("*.json")}

    to_fetch: list[tuple[str, Path]] = []
    no_source_image: list[str] = []
    for r in records:
        source_file = r["source_file"]
        stem = Path(source_file).stem
        if stem in existing_raw:
            continue
        image_path = CHEQUES_DIR / source_file
        if not image_path.is_file():
            no_source_image.append(source_file)
            continue
        to_fetch.append((r["record_id"], image_path))

    print(f"Records: {len(records)}")
    print(f"Already have raw response: {len(records) - len(to_fetch) - len(no_source_image)}")
    print(f"No source image on disk (cannot backfill): {len(no_source_image)}")
    for name in no_source_image:
        print(f"  - {name}")
    print(f"Will call Azure DI for: {len(to_fetch)} file(s)")
    print()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetched = errors = 0
    for record_id, image_path in to_fetch:
        try:
            raw = analyze_raw(str(image_path), endpoint, key)
        except Exception as exc:
            print(f"WARNING: {record_id} ({image_path.name}) - {exc}", file=sys.stderr)
            errors += 1
            continue
        dest = RAW_DIR / (image_path.stem + ".json")
        dest.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
        fetched += 1
        print(f"[{record_id}] {image_path.name} -> {dest.relative_to(ROOT)}")

    print()
    print(f"Fetched: {fetched}")
    print(f"Errors: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
