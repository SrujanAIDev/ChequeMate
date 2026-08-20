r"""Retroactive re-flag: apply ruleset 1.1.0 (mandatory memo) to existing
reports/cheques.json records.

Standalone and idempotent — run it as many times as you like:

    python scripts/migrate_memo.py

For each record still on an older ruleset_version:
  - If a source cheque image can be matched (by source_hash) to a file under
    cheques/, this calls Azure Document Intelligence (using AZURE_DI_ENDPOINT
    / AZURE_DI_KEY from .env) to re-extract the real Memo value and re-runs
    validate() — an accurate re-flag.
  - Otherwise it prints a WARNING and blanket-fails the memo rule in place,
    since no source exists to check what the real memo was.

Records already on the current ruleset_version are left untouched, so a
second run is a no-op (no duplicate rules, no duplicate audit lines).

Backs up reports/cheques.json to reports/cheques.json.bak before writing —
only on the first run (an existing .bak is never overwritten, so it always
holds the true pre-migration snapshot).
"""

from __future__ import annotations

import os
import shutil
import ssl
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import certifi  # noqa: E402

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where())

from chequemate import Config, validate  # noqa: E402
from chequemate.extract import analyze  # noqa: E402
from chequemate import report  # noqa: E402

TARGET_VERSION = "1.1.0"
CHEQUES_DIR = ROOT / "cheques"


def load_dotenv(path: str = ".env") -> None:
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


def _backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[backup] {path.name} -> {bak.name}")
    else:
        print(f"[backup] {bak.name} already exists, leaving pre-migration "
              f"snapshot untouched")


def _index_local_images_by_hash() -> dict[str, list[Path]]:
    if not CHEQUES_DIR.is_dir():
        return {}
    index: dict[str, list[Path]] = {}
    for p in sorted(CHEQUES_DIR.iterdir()):
        if p.is_file():
            index.setdefault(report.hash_file(p), []).append(p)
    return index


def _pick_source_image(record: dict, by_hash: dict[str, list[Path]]) -> Path | None:
    """Prefer the file whose name matches the record's own source_file —
    two different files can share identical bytes (e.g. a saved copy), and
    picking the wrong one would silently rewrite source_file on migration."""
    candidates = by_hash.get(record.get("source_hash")) or []
    if not candidates:
        return None
    for c in candidates:
        if c.name == record.get("source_file"):
            return c
    return candidates[0]


def _recompute_verdict(rules: dict) -> str:
    return "INVALID" if any(r.get("status") == "FAIL" for r in rules.values()) \
        else "VALID"


def _reprocess_from_source(record: dict, image_path: Path,
                           endpoint: str, key: str) -> dict:
    """Accurate path: call Azure on the real image, re-run validate().

    `source_file` is preserved from the original record — image_path is only
    the byte-identical local file used to make the call, and may not be the
    same filename the record was originally created under.
    """
    cfg = Config()
    processed_time = datetime.fromisoformat(record["processed_time"])
    cheque = analyze(str(image_path), endpoint, key)
    result = validate(cheque, cfg)
    rebuilt = report.create_report_record(
        cheque, result, source_file=record["source_file"],
        source_hash=record["source_hash"], processed_time=processed_time)
    rebuilt["record_id"] = record["record_id"]
    return rebuilt


def _fallback_blanket_fail(record: dict) -> dict:
    """No source available: append a hard-FAIL memo rule in place."""
    updated = dict(record)
    rules = dict(updated.get("rules") or {})
    rules["memo"] = {
        "status": "FAIL",
        "message": "memo not captured under ruleset 1.0.0 — re-flagged on "
                   "migration",
        "confidence": None,
    }
    updated["rules"] = rules
    updated["memo"] = updated.get("memo")  # schema parity with new records
    updated["confidence"] = dict(updated.get("confidence") or {}, memo=None)
    updated["raw_values"] = dict(updated.get("raw_values") or {}, memo=None)
    updated["ruleset_version"] = TARGET_VERSION
    updated["verdict"] = _recompute_verdict(rules)
    return updated


def main() -> int:
    load_dotenv()
    endpoint = os.getenv("AZURE_DI_ENDPOINT")
    key = os.getenv("AZURE_DI_KEY")

    report.ensure_report_directory()
    _backup(report.CHEQUES_JSON)

    records = report.load_records()
    by_hash = _index_local_images_by_hash()

    before_valid = sum(1 for r in records if r.get("verdict") == "VALID")
    before_invalid = len(records) - before_valid

    reprocessed = fallback = skipped = 0
    for i, record in enumerate(records):
        if record.get("ruleset_version") == TARGET_VERSION:
            skipped += 1
            continue

        image_path = _pick_source_image(record, by_hash)
        if image_path and endpoint and key:
            print(f"[reprocess] {record['record_id']} <- {image_path.name} "
                 f"(live Azure call)")
            new_record = _reprocess_from_source(record, image_path, endpoint, key)
            method = "reprocessed"
            reprocessed += 1
        else:
            reason = "no matching source image found" if not image_path else \
                "AZURE_DI_ENDPOINT / AZURE_DI_KEY not set"
            print(f"WARNING: {record['record_id']} ({record.get('source_file')}) "
                 f"— {reason}. Blanket-failing memo under ruleset "
                 f"{TARGET_VERSION}. This assumes the memo was blank, which "
                 f"is NOT verified — re-process from source for an accurate "
                 f"re-flag once the original is available.")
            new_record = _fallback_blanket_fail(record)
            method = "fallback_blanket_fail"
            fallback += 1

        # Persist this one record's change (data + audit) before moving on,
        # so a crash mid-run never leaves audit.jsonl ahead of cheques.json —
        # a re-run would otherwise re-migrate (and double-log) this record.
        records[i] = new_record
        report.save_records(records)
        report.append_audit_event({
            "timestamp": datetime.now().astimezone().isoformat(),
            "event": "ruleset_migration",
            "record_id": new_record["record_id"],
            "from_ruleset": record.get("ruleset_version"),
            "to_ruleset": TARGET_VERSION,
            "verdict": new_record["verdict"],
            "method": method,
        })

    report.regenerate_report()

    after_valid = sum(1 for r in records if r.get("verdict") == "VALID")
    after_invalid = len(records) - after_valid

    print()
    print(f"Reprocessed from source : {reprocessed}")
    print(f"Fallback blanket-failed : {fallback}")
    print(f"Already on {TARGET_VERSION} (skipped) : {skipped}")
    print(f"Valid   before -> after : {before_valid} -> {after_valid}")
    print(f"Invalid before -> after : {before_invalid} -> {after_invalid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
