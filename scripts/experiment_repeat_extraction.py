r"""ONE-OFF EXPERIMENT: call Azure DI 2 more times per batch file (run1 is
already saved in raw/*.json from earlier work) and reconcile all 3 runs via
extract.to_normalized_multi(), to measure how often DI's own run-to-run
instability (confirmed by a separate experiment - see
experiment_imageprep_di_comparison.py's regressions) actually changes a
field's usable status when caught by repeat extraction.

Read-only w.r.t. cheques.json: never written. Extra runs saved to
raw_multi/ (gitignored), never overwriting raw/*.json (run 1).

Costs 2 real Azure Document Intelligence calls per file.

Run:
    python scripts/experiment_repeat_extraction.py
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

from chequemate.extract import (  # noqa: E402
    analyze_raw, first_document, to_normalized, to_normalized_multi,
)
from chequemate.models import ParseStatus  # noqa: E402
from run_batch import load_dotenv  # noqa: E402

CHEQUES_DIR = ROOT / "cheques"
RAW_DIR = ROOT / "raw"                # run 1 - already captured, read-only
RAW_MULTI_DIR = ROOT / "raw_multi"    # runs 2-3 - new, gitignored
BATCH_FILES = [f"20260820113647241_{i:04d}.jpg" for i in range(1, 23)]

FIELD_NAMES = ("payee", "amount_numeric", "amount_words", "cheque_date",
              "signature", "memo")


def fetch_extra_runs() -> None:
    load_dotenv()
    endpoint = os.getenv("AZURE_DI_ENDPOINT")
    key = os.getenv("AZURE_DI_KEY")
    if not endpoint or not key:
        print("Azure credentials not found.", file=sys.stderr)
        raise SystemExit(2)

    RAW_MULTI_DIR.mkdir(exist_ok=True)
    for name in BATCH_FILES:
        stem = Path(name).stem
        path = CHEQUES_DIR / name
        for run in (2, 3):
            dest = RAW_MULTI_DIR / f"{stem}_run{run}.json"
            if dest.is_file():
                print(f"[skip] {name} run{run} (already fetched)")
                continue
            raw = analyze_raw(str(path), endpoint, key)
            dest.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
            print(f"[fetched] {name} run{run}")


def _status_signature(cheque) -> dict:
    return {name: (getattr(cheque, name).parse_status.value,
                  getattr(cheque, name).ok)
           for name in FIELD_NAMES}


def diff() -> None:
    changed = 0
    detail_lines = []
    for name in BATCH_FILES:
        stem = Path(name).stem
        run1_path = RAW_DIR / f"{stem}.json"
        run2_path = RAW_MULTI_DIR / f"{stem}_run2.json"
        run3_path = RAW_MULTI_DIR / f"{stem}_run3.json"
        if not (run1_path.is_file() and run2_path.is_file() and run3_path.is_file()):
            print(f"WARNING: missing a run for {name}", file=sys.stderr)
            continue

        raw1 = json.loads(run1_path.read_text(encoding="utf-8"))
        raw2 = json.loads(run2_path.read_text(encoding="utf-8"))
        raw3 = json.loads(run3_path.read_text(encoding="utf-8"))

        single = to_normalized(first_document(raw1))
        reconciled = to_normalized_multi(
            [first_document(raw1), first_document(raw2), first_document(raw3)])

        single_sig = _status_signature(single)
        recon_sig = _status_signature(reconciled)
        diffs = {f: (single_sig[f], recon_sig[f]) for f in FIELD_NAMES
                if single_sig[f] != recon_sig[f]}
        if diffs:
            changed += 1
            for field_name, (before, after) in diffs.items():
                detail_lines.append(f"  {name}  {field_name}: {before} -> {after}")

    print(f"\nRecords where 3-run reconciliation changed at least one field's "
         f"status: {changed} / {len(BATCH_FILES)}")
    print()
    for line in detail_lines:
        print(line)


if __name__ == "__main__":
    fetch_extra_runs()
    print()
    diff()
