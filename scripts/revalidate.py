r"""Re-validate every existing reports/cheques.json record against the
CURRENT normalize/rules logic — entirely offline.

Every record already stores raw_values (the original extractor text for
payee, amount_numeric, amount_words, date, memo) plus signature_detected
and per-field confidence. That's everything normalize_*() needs to rebuild
a NormalizedCheque and re-run validate() — no Azure call, no image, no
re-extraction required. Use this whenever rules.py/normalize.py logic
changes and you want history re-scored against it (as opposed to
migrate_memo.py, which is for when the SCHEMA gained a field extraction
never captured before).

Run:
    python scripts/revalidate.py

Idempotent: a record already on the current ruleset_version is skipped, so
re-running is a no-op. Backs up cheques.json once (never overwrites an
existing .bak). Persists + audits one record at a time so a crash mid-run
can never leave audit.jsonl ahead of cheques.json.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import shutil  # noqa: E402

from chequemate import Config, validate  # noqa: E402
from chequemate.extract import normalize_memo  # noqa: E402
from chequemate.models import NormalizedCheque  # noqa: E402
from chequemate.normalize import (  # noqa: E402
    normalize_amount_numeric,
    normalize_amount_words,
    normalize_date,
    normalize_payee,
    normalize_signature,
)
from chequemate.validate import RULE_SET_VERSION as TARGET_VERSION  # noqa: E402
from chequemate import report  # noqa: E402


def _rebuild_cheque(record: dict, date_convention: str = "DMY") -> NormalizedCheque:
    raw = record.get("raw_values") or {}
    conf = record.get("confidence") or {}
    return NormalizedCheque(
        payee=normalize_payee(raw.get("payee"), confidence=conf.get("payee")),
        amount_numeric=normalize_amount_numeric(
            raw.get("amount_numeric"), confidence=conf.get("amount_numeric")),
        amount_words=normalize_amount_words(
            raw.get("amount_words"), confidence=conf.get("amount_words")),
        cheque_date=normalize_date(
            raw.get("date"), prefer=date_convention, confidence=conf.get("date")),
        signature=normalize_signature(
            None, detected=record.get("signature_detected"),
            confidence=conf.get("signature")),
        memo=normalize_memo(raw.get("memo"), confidence=conf.get("memo")),
        source_id=record.get("source_file"),
    )


def _backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[backup] {path.name} -> {bak.name}")
    else:
        print(f"[backup] {bak.name} already exists, leaving pre-revalidation "
             f"snapshot untouched")


def main() -> int:
    report.ensure_report_directory()
    _backup(report.CHEQUES_JSON)

    records = report.load_records()
    cfg = Config()

    before_valid = sum(1 for r in records if r.get("verdict") == "VALID")
    changed = flipped_to_valid = flipped_to_invalid = skipped = 0

    for i, record in enumerate(records):
        if record.get("ruleset_version") == TARGET_VERSION:
            skipped += 1
            continue

        cheque = _rebuild_cheque(record)
        processed_time = datetime.fromisoformat(record["processed_time"])
        result = validate(cheque, cfg)

        rebuilt = report.create_report_record(
            cheque, result, source_file=record["source_file"],
            source_hash=record["source_hash"], processed_time=processed_time)
        rebuilt["record_id"] = record["record_id"]

        old_verdict, new_verdict = record.get("verdict"), rebuilt["verdict"]
        if old_verdict != new_verdict:
            if new_verdict == "VALID":
                flipped_to_valid += 1
            else:
                flipped_to_invalid += 1
            print(f"[{record['record_id']}] {record.get('source_file')}: "
                 f"{old_verdict} -> {new_verdict}")

        records[i] = rebuilt
        changed += 1
        report.save_records(records)
        report.append_audit_event({
            "timestamp": datetime.now().astimezone().isoformat(),
            "event": "ruleset_revalidation",
            "record_id": rebuilt["record_id"],
            "from_ruleset": record.get("ruleset_version"),
            "to_ruleset": TARGET_VERSION,
            "old_verdict": old_verdict,
            "new_verdict": new_verdict,
        })

    report.regenerate_report()

    after_valid = sum(1 for r in records if r.get("verdict") == "VALID")

    print()
    print(f"Re-validated : {changed}")
    print(f"Already on {TARGET_VERSION} (skipped) : {skipped}")
    print(f"Flipped INVALID -> VALID : {flipped_to_valid}")
    print(f"Flipped VALID -> INVALID : {flipped_to_invalid}")
    print(f"Valid before -> after : {before_valid} -> {after_valid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
