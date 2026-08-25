r"""One-command, incremental batch runner for ChequeMate.

Processes every cheque image (or saved raw-response .json) in a folder, but
skips anything whose file bytes were already recorded in reports/cheques.json
(by source_hash) BEFORE making any Azure call — so a re-run only pays for
genuinely new cheques.

Run:
    python scripts/run_batch.py                    # cheques/ at repo root
    python scripts/run_batch.py --folder path\to\cheques

Needs AZURE_DI_ENDPOINT / AZURE_DI_KEY (env vars or a .env file at repo
root) — only if there's at least one genuinely new image to send to Azure;
a run that only finds already-processed files, or only new .json replay
files, needs no credentials at all.
"""

from __future__ import annotations

import argparse
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

from chequemate import Config, validate  # noqa: E402
from chequemate.extract import analyze, first_document, to_normalized  # noqa: E402
from chequemate import report  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".heif"}
REPLAY_EXTENSIONS = {".json"}


def load_dotenv(path: str = ".env") -> None:
    """Read KEY=value lines from .env. Real env vars always win."""
    p = ROOT / path
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def discover_files(folder: Path) -> list[Path]:
    exts = IMAGE_EXTENSIONS | REPLAY_EXTENSIONS
    return sorted(p for p in folder.iterdir()
                 if p.is_file() and p.suffix.lower() in exts)


def process_one(path: Path, endpoint: str | None, key: str | None,
                cfg: Config, date_convention: str) -> report.ReportOutcome:
    """analyze() for images, offline replay for .json -> validate() -> update_report()."""
    if path.suffix.lower() in REPLAY_EXTENSIONS:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cheque = to_normalized(first_document(raw), source_id=str(path),
                               date_convention=date_convention)
    else:
        cheque = analyze(str(path), endpoint, key, date_convention=date_convention)
    result = validate(cheque, cfg)
    return report.update_report(cheque=cheque, validation=result, source_path=path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default=str(ROOT / "cheques"),
                    help="folder of cheque images / replay JSON (default: cheques/)")
    ap.add_argument("--payee", default="Town of Whitby")
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--payee-tolerance", type=int, default=0)
    ap.add_argument("--date-convention", choices=["DMY", "MDY"], default="DMY")
    ap.add_argument("--verbose", action="store_true",
                    help="print a line per file (skip/new) as it's decided")
    args = ap.parse_args(argv)

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        return 2

    files = discover_files(folder)

    # Incremental / new-only: decide skip-vs-process from already-recorded
    # source_hash values BEFORE touching Azure at all — this is the whole
    # cost-control point of this script.
    already_seen = {r["source_hash"] for r in report.load_records()
                    if r.get("source_hash")}
    to_process: list[Path] = []
    skipped = 0
    for path in files:
        if report.hash_file(path) in already_seen:
            skipped += 1
            if args.verbose:
                print(f"  [skip] {path.name} (already processed)")
        else:
            to_process.append(path)
            if args.verbose:
                print(f"  [new]  {path.name}")

    endpoint = key = None
    if any(p.suffix.lower() in IMAGE_EXTENSIONS for p in to_process):
        load_dotenv()
        endpoint = os.getenv("AZURE_DI_ENDPOINT")
        key = os.getenv("AZURE_DI_KEY")
        if not endpoint or not key:
            print("Azure credentials not found.", file=sys.stderr)
            print(f"  AZURE_DI_ENDPOINT: {'set' if endpoint else 'MISSING'}",
                 file=sys.stderr)
            print(f"  AZURE_DI_KEY:      {'set' if key else 'MISSING'}",
                 file=sys.stderr)
            print(f"  .env file:         "
                 f"{'found' if (ROOT / '.env').is_file() else 'not found'} in {ROOT}",
                 file=sys.stderr)
            print("\nSet AZURE_DI_ENDPOINT / AZURE_DI_KEY (env vars or a .env "
                 "file at repo root) and re-run.", file=sys.stderr)
            return 2

    cfg = Config(expected_payee=args.payee, max_age_months=args.months,
                payee_edit_tolerance=args.payee_tolerance)

    processed: list[tuple[str, str, str]] = []
    errors = 0
    for path in to_process:
        try:
            outcome = process_one(path, endpoint, key, cfg, args.date_convention)
        except Exception as exc:
            print(f"WARNING: {path.name} — {exc}", file=sys.stderr)
            errors += 1
            continue
        if outcome.created:
            processed.append((outcome.record["record_id"], path.name,
                             outcome.record["verdict"]))
        else:
            # Hash matched a record written between our load_records() snapshot
            # and now (e.g. a concurrent run) — genuinely not new after all.
            skipped += 1

    print("Batch complete:")
    print(f"  Found in folder : {len(files)}")
    print(f"  Skipped (already processed) : {skipped}")
    print(f"  Newly processed : {len(processed)}")
    for record_id, name, verdict in processed:
        print(f"    {record_id}  {name}  {verdict}")
    print(f"  Errors : {errors}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
