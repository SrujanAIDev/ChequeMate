"""Validate cheque images, capturing raw Azure responses for offline replay.

First pass over real cheques (one Azure call each, saves the response):

    export AZURE_DI_ENDPOINT=... AZURE_DI_KEY=...
    python cli.py --save-raw raw/ --diagnose --fields cheques/*.png

Every pass after that is free and offline:

    python cli.py --replay raw/*.json --fields

NOTE: saved JSON contains account numbers, addresses and names from real
cheques. Keep raw/ local, gitignored, and delete it when done.
"""

import argparse
import glob
import json
import os
import ssl
import sys
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where())


def load_dotenv(path: str = ".env") -> None:
    """Read KEY=value lines from .env. Real env vars always win.

    Avoids re-exporting credentials every terminal session without ever
    putting them in tracked source. Keep .env gitignored.
    """
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val

from chequemate import Config, Verdict, validate                    # noqa: E402
from chequemate.extract import (analyze_raw, diagnose_signature,    # noqa: E402
                                first_document, to_normalized)
from chequemate import report                                       # noqa: E402


def expand(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        out.extend(hits or [p])
    return out


def run_one(name: str, raw: dict, cfg: Config, args) -> bool:
    print(f"\n{'=' * 64}\n{name}\n{'=' * 64}")
    try:
        doc = first_document(raw)
    except ValueError as exc:
        print(f"EXTRACTION FAILED: {exc}")
        return False

    if args.diagnose:
        print(diagnose_signature(doc))
        print("-" * 64)
    if args.raw:
        print(json.dumps(doc.get("fields", {}), indent=2, default=str))

    cheque = to_normalized(doc, source_id=name,
                           date_convention=args.date_convention)
    result = validate(cheque, cfg)
    print(result.summary())

    if args.fields:
        print("  --- normalised ---")
        for f in (cheque.payee, cheque.amount_numeric, cheque.amount_words,
                  cheque.cheque_date, cheque.signature):
            print(f"  {f.name:<16} {f.parse_status.value:<12} "
                  f"{f.value!r}  <- {f.raw_text!r}")

    try:
        outcome = report.update_report(cheque=cheque, validation=result,
                                       source_path=name)
        if outcome.created:
            print(f"  [report] recorded as {outcome.record['record_id']}")
        else:
            print(f"  [report] Report record already exists: "
                  f"{outcome.record['record_id']}")
    except report.ReportError as exc:
        print(f"  [report] WARNING: {exc}", file=sys.stderr)

    return result.verdict is Verdict.VALID


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="images, or JSON with --replay")
    ap.add_argument("--replay", action="store_true",
                    help="read saved responses instead of calling Azure")
    ap.add_argument("--save-raw", metavar="DIR",
                    help="write each full Azure response to DIR/<name>.json")
    ap.add_argument("--diagnose", action="store_true",
                    help="dump the PayerSignatures field key by key")
    ap.add_argument("--fields", action="store_true",
                    help="show normalised value beside its raw text")
    ap.add_argument("--raw", action="store_true", help="dump all fields as JSON")
    ap.add_argument("--payee", default="Town of Whitby")
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--payee-tolerance", type=int, default=0)
    ap.add_argument("--date-convention", choices=["DMY", "MDY"], default="DMY")
    args = ap.parse_args()

    cfg = Config(expected_payee=args.payee, max_age_months=args.months,
                 payee_edit_tolerance=args.payee_tolerance)
    paths = expand(args.inputs)
    outdir = Path(args.save_raw) if args.save_raw else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    endpoint = key = None
    if not args.replay:
        load_dotenv()
        endpoint = os.getenv("AZURE_DI_ENDPOINT")
        key = os.getenv("AZURE_DI_KEY")
        if not endpoint or not key:
            print("Azure credentials not found.", file=sys.stderr)
            print(f"  AZURE_DI_ENDPOINT: "
                  f"{'set' if endpoint else 'MISSING'}", file=sys.stderr)
            print(f"  AZURE_DI_KEY:      "
                  f"{'set' if key else 'MISSING'}", file=sys.stderr)
            print(f"  .env file:         "
                  f"{'found' if Path('.env').is_file() else 'not found'} "
                  f"in {Path.cwd()}", file=sys.stderr)
            print("\nEither create a .env file here containing:\n"
                  "  AZURE_DI_ENDPOINT=https://<resource>.cognitiveservices"
                  ".azure.com/\n  AZURE_DI_KEY=<your-key>\n"
                  "or set them in THIS terminal with PowerShell syntax:\n"
                  '  $env:AZURE_DI_ENDPOINT = "https://..."\n'
                  '  $env:AZURE_DI_KEY = "..."\n'
                  "(note: `set VAR=value` does NOT work in PowerShell)",
                  file=sys.stderr)
            return 2

    valid = 0
    for path in paths:
        try:
            if args.replay:
                raw = json.loads(Path(path).read_text())
            else:
                raw = analyze_raw(path, endpoint, key)
                if outdir:
                    dest = outdir / (Path(path).stem + ".json")
                    dest.write_text(json.dumps(raw, indent=2, default=str))
                    print(f"[saved {dest}]")
        except Exception as exc:
            print(f"\n{path}: FAILED — {exc}")
            continue
        valid += run_one(path, raw, cfg, args)

    print(f"\n{'=' * 64}\n{valid}/{len(paths)} VALID")
    return 0 if valid == len(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())