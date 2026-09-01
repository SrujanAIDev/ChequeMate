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

Every genuinely new image's full Azure response is saved to raw/<name>.json
by default (pass --save-raw "" to disable) - the same capture cli.py's
--save-raw has always done, now on by default here too, so a field-
extraction question ("was X actually absent from Azure's response, or lost
in normalization?") never needs a second Azure call to answer.
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
from chequemate.extract import (  # noqa: E402
    analyze_raw, first_document, to_normalized, to_normalized_multi,
)
from chequemate import report  # noqa: E402
from chequemate import ocr_verify  # noqa: E402

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
                cfg: Config, date_convention: str,
                save_raw_dir: Path | None = None,
                trocr_cfg: "ocr_verify.TrOCRVerificationConfig | None" = None,
                repeat_extraction: int = 1,
                ) -> report.ReportOutcome:
    """Call Azure (or replay a saved response) -> validate() -> update_report().

    `save_raw_dir`, when given, writes the FULL Azure response to
    <dir>/<name>.json before normalizing - the same capture cli.py's
    --save-raw does, so a genuinely new image only ever needs to touch
    Azure once, offline replay/debugging is possible forever after, and a
    field-extraction question (e.g. "was Memo actually absent from Azure's
    response, or lost in normalization?") never again requires a second
    Azure call to answer.

    `trocr_cfg`, when given and enabled, runs field-level TrOCR
    verification (see chequemate.ocr_verify) using this same `raw` Azure
    response - the only place in the pipeline that still has the response's
    boundingRegions/pages available (report.py's persisted records never
    keep raw polygons). Skipped entirely (no error) for replay-JSON-only
    inputs with no source image to crop, and whenever `trocr_cfg` is None
    or disabled - existing behaviour when TrOCR is off is unchanged.

    `repeat_extraction` (default 1 - identical behaviour to before this
    existed): calls Azure this many times on the SAME image and reconciles
    via extract.to_normalized_multi() - see that module's docstring for
    why (a real experiment found DI is not perfectly deterministic
    run-to-run on unchanged input). Ignored for replay-JSON inputs (there
    is nothing to re-call). Each additional run's raw response is saved
    alongside the first as <name>_run<N>.json when save_raw_dir is set.
    """
    is_image = path.suffix.lower() in IMAGE_EXTENSIONS
    if path.suffix.lower() in REPLAY_EXTENSIONS:
        raw_responses = [json.loads(path.read_text(encoding="utf-8"))]
    else:
        raw_responses = [analyze_raw(str(path), endpoint, key)
                         for _ in range(max(1, repeat_extraction))]
        if save_raw_dir is not None:
            save_raw_dir.mkdir(parents=True, exist_ok=True)
            for i, one_raw in enumerate(raw_responses, start=1):
                suffix = "" if i == 1 else f"_run{i}"
                dest = save_raw_dir / (path.stem + suffix + ".json")
                dest.write_text(json.dumps(one_raw, indent=2, default=str),
                                encoding="utf-8")

    documents = [first_document(r) for r in raw_responses]
    if len(documents) == 1:
        cheque = to_normalized(documents[0], source_id=str(path),
                               date_convention=date_convention)
    else:
        cheque = to_normalized_multi(documents, source_id=str(path),
                                     date_convention=date_convention)
    raw = raw_responses[0]  # first run - TrOCR's polygon source below

    result = validate(cheque, cfg)

    ocr_verifications = None
    if trocr_cfg is not None and trocr_cfg.enabled and is_image:
        from PIL import Image
        with Image.open(path) as image:
            ocr_verifications = ocr_verify.verify_cheque_fields(
                raw, image, cheque, trocr_cfg, record_id=path.stem,
                source_path=path)

    return report.update_report(cheque=cheque, validation=result, source_path=path,
                                ocr_verifications=ocr_verifications)


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
    ap.add_argument("--save-raw", metavar="DIR", default=str(ROOT / "raw"),
                    help="write each new full Azure response to DIR/<name>.json "
                         "(default: raw/; pass an empty string to disable)")
    ap.add_argument("--trocr", action="store_true",
                    help="enable field-level TrOCR secondary verification "
                         "(payee/amount_words/memo) on newly-processed images. "
                         "Off by default; equivalent to CHEQUEMATE_TROCR_ENABLED=true. "
                         "See docs/trocr_verification.md.")
    ap.add_argument("--repeat-extraction", type=int, default=1, metavar="N",
                    help="call Azure DI N times per newly-processed image and "
                         "reconcile the results (see extract.to_normalized_multi) "
                         "- fields that disagree across runs become AMBIGUOUS "
                         "and route every rule that reads them to UNABLE rather "
                         "than trusting a single, possibly-inconsistent read. "
                         "Default 1 (off - identical to prior behaviour). "
                         "Each extra call is a real, billed Azure request.")
    args = ap.parse_args(argv)
    save_raw_dir = Path(args.save_raw) if args.save_raw else None

    trocr_cfg = ocr_verify.config_from_env(
        ocr_verify.TrOCRVerificationConfig(enabled=args.trocr))

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
            outcome = process_one(path, endpoint, key, cfg, args.date_convention,
                                  save_raw_dir=save_raw_dir, trocr_cfg=trocr_cfg,
                                  repeat_extraction=args.repeat_extraction)
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
