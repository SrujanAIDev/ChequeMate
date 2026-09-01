r"""Apply manually-verified corrections for fields the automated pipeline
couldn't confidently resolve on its own, after a human (or a human-directed
visual read of the source image) confirms the true value.

This is NOT model training — chequemate has no trainable component:
prebuilt-check.us is Azure's fixed cloud model (not retrainable from here),
and normalize.py/rules.py are deterministic code, not a learned classifier.
This script is the honest alternative: a small, explicit, auditable list of
"here is the verified truth for this specific record" corrections, applied
on top of the untouched algorithmic record — never silently overwriting the
original OCR read.

For each correction:
  - the underlying Field (signature) or the specific RuleResult (payee) is
    updated to the verified value,
  - raw_values / raw OCR text is left completely untouched, so what the
    extractor actually said stays visible in the report's detail panel,
  - a reviews.json entry is added explaining the override,
  - a ruleset_migration-style audit.jsonl event is appended.

Run:
    python scripts/apply_visual_verification.py

Idempotent: a record already carrying a "manual_visual_verification" review
for a given field is skipped on re-run.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chequemate import Config, validate  # noqa: E402
from chequemate.extract import amount_numeric_cents, normalize_memo  # noqa: E402
from chequemate.models import NormalizedCheque, RuleResult, RuleStatus, Verdict  # noqa: E402
from chequemate.normalize import (  # noqa: E402
    normalize_amount_numeric,
    normalize_amount_words,
    normalize_date,
    normalize_payee,
    normalize_signature,
)
from chequemate import report  # noqa: E402

REVIEWER = "Claude (visual review, requested by srujan 2026-08-24)"

# Cleared for the 1.7.0 re-score (explicit instruction: nothing should be
# masked by a stale human correction while re-scoring against the new
# ruleset - see CLAUDE.md/the 1.7.0 changelog). The 8 confirmations that
# stood here (6 signature, 2 payee, all against CHQ-20260824-000x records)
# are preserved in git history and in reports/reviews.json.bak (the
# pre-1.7.0 snapshot) if they ever need restoring - this is not a claim
# that the underlying human visual confirmations were wrong, only that
# this re-score reports the algorithm's own current output unmasked.
SIGNATURE_CONFIRMATIONS: list[str] = []

PAYEE_CONFIRMATIONS: dict[str, tuple[str, str]] = {}


def _rebuild_cheque(record: dict, date_convention: str = "DMY") -> NormalizedCheque:
    raw = record.get("raw_values") or {}
    conf = record.get("confidence") or {}
    amount_numeric = normalize_amount_numeric(
        raw.get("amount_numeric"), confidence=conf.get("amount_numeric"))
    return NormalizedCheque(
        payee=normalize_payee(raw.get("payee"), confidence=conf.get("payee")),
        amount_numeric=amount_numeric,
        amount_words=normalize_amount_words(
            raw.get("amount_words"), numeric_cents=amount_numeric_cents(amount_numeric),
            confidence=conf.get("amount_words")),
        cheque_date=normalize_date(
            raw.get("date"), prefer=date_convention, confidence=conf.get("date")),
        signature=normalize_signature(
            None, detected=record.get("signature_detected"),
            confidence=conf.get("signature")),
        memo=normalize_memo(raw.get("memo"), confidence=conf.get("memo")),
        source_id=record.get("source_file"),
    )


def _recompute_verdict(rules: list[RuleResult]) -> Verdict:
    """Mirrors validate.py's three-way collapse (FAIL beats UNABLE beats clean)."""
    if any(r.status is RuleStatus.FAIL for r in rules):
        return Verdict.INVALID
    if any(r.status is RuleStatus.UNABLE for r in rules):
        return Verdict.REVIEW
    return Verdict.VALID


def _already_reviewed(reviews: list[dict], record_id: str, field: str) -> bool:
    return any(rv.get("record_id") == record_id
              and rv.get("status") == f"manual_visual_verification:{field}"
              for rv in reviews)


def _backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[backup] {path.name} -> {bak.name}")
    else:
        print(f"[backup] {bak.name} already exists, leaving pre-correction "
             f"snapshot untouched")


def main() -> int:
    report.ensure_report_directory()
    _backup(report.CHEQUES_JSON)

    records = report.load_records()
    by_id = {r["record_id"]: i for i, r in enumerate(records)}
    reviews = report.load_reviews()
    cfg = Config()
    applied = skipped = 0

    def save_correction(record_id: str, field: str, note: str, new_record: dict):
        nonlocal applied
        idx = by_id[record_id]
        records[idx] = new_record
        report.save_records(records)
        reviews.append({
            "record_id": record_id,
            "timestamp": datetime.now().astimezone().isoformat(),
            "reviewer": REVIEWER,
            "status": f"manual_visual_verification:{field}",
            "note": note,
        })
        report.save_reviews(reviews)
        report.append_audit_event({
            "timestamp": datetime.now().astimezone().isoformat(),
            "event": "manual_visual_verification",
            "record_id": record_id,
            "field": field,
            "note": note,
        })
        applied += 1
        print(f"[{record_id}] {field}: {note}")

    # --- signature: field-level correction (no raw text ever existed to lose) ---
    for record_id in SIGNATURE_CONFIRMATIONS:
        if record_id not in by_id:
            print(f"WARNING: {record_id} not found in cheques.json, skipping")
            continue
        if _already_reviewed(reviews, record_id, "signature"):
            skipped += 1
            continue
        record = records[by_id[record_id]]
        processed_time = datetime.fromisoformat(record["processed_time"])

        cheque = _rebuild_cheque(record)
        cheque.signature = normalize_signature(None, detected=True,
                                               confidence=cheque.signature.confidence)
        result = validate(cheque, cfg)
        new_record = report.create_report_record(
            cheque, result, source_file=record["source_file"],
            source_hash=record["source_hash"], processed_time=processed_time)
        new_record["record_id"] = record_id

        save_correction(record_id, "signature",
                        "Extractor returned no signature verdict at all "
                        "('field not returned by extractor'); visually "
                        "confirmed a handwritten signature is present on "
                        "the source image.", new_record)

    # --- payee: rule-level override (raw OCR text preserved for audit) ---
    for record_id, (confirmed, why) in PAYEE_CONFIRMATIONS.items():
        if record_id not in by_id:
            print(f"WARNING: {record_id} not found in cheques.json, skipping")
            continue
        if _already_reviewed(reviews, record_id, "payee"):
            skipped += 1
            continue
        record = records[by_id[record_id]]
        processed_time = datetime.fromisoformat(record["processed_time"])

        cheque = _rebuild_cheque(record)
        result = validate(cheque, cfg)
        corrected_rules = [
            RuleResult("payee", RuleStatus.PASS,
                      f"{cheque.payee.raw_text!r} (OCR) visually confirmed as "
                      f"{confirmed!r} against the source image — manual "
                      f"review override; {why}", cheque.payee.confidence)
            if r.rule_id == "payee" else r
            for r in result.rules
        ]
        result.rules = corrected_rules
        result.verdict = _recompute_verdict(corrected_rules)

        new_record = report.create_report_record(
            cheque, result, source_file=record["source_file"],
            source_hash=record["source_hash"], processed_time=processed_time)
        new_record["record_id"] = record_id

        save_correction(record_id, "payee", why, new_record)

    report.regenerate_report()

    print()
    print(f"Applied : {applied}")
    print(f"Already verified (skipped) : {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
