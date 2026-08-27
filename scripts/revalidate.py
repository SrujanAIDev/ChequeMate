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

Standing scripts/apply_visual_verification.py overrides are a LAYER on top
of this, not a reason to skip a record. Every record is rebuilt fresh from
raw_values exactly as before (so a genuine rules/normalize improvement
reaches every record, including previously-verified ones), and only THEN
does any standing human-confirmed value get re-applied on top, with the
verdict recomputed after the overlay. Skipping verified records entirely
was tried first and rejected: it freezes all four of a record's OTHER
fields against every future improvement just because one field was once
manually confirmed, and its ruleset_version silently drifts from the rest
of the corpus. See CLAUDE.md's "revalidate.py must never touch a record
with a standing visual-verification override" section for the incident
that motivated this (the ORIGINAL version of this script, used for the
1.3.0 -> 1.4.0 bump, discarded 8 such overrides outright).

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import shutil  # noqa: E402

from chequemate import Config, validate  # noqa: E402
from chequemate.extract import normalize_memo  # noqa: E402
from chequemate.models import NormalizedCheque, RuleResult, RuleStatus  # noqa: E402
from chequemate.normalize import (  # noqa: E402
    normalize_amount_numeric,
    normalize_amount_words,
    normalize_date,
    normalize_payee,
    normalize_signature,
)
from chequemate.validate import RULE_SET_VERSION as TARGET_VERSION  # noqa: E402
from chequemate import report  # noqa: E402

# The canonical source of "what value a human confirmed" for a given
# record/field - same dicts apply_visual_verification.py itself applies.
# Reusing them (rather than re-deriving from reviews.json's free-text note)
# is exactly what "layer, don't skip" requires: the note is prose for a
# person, these dicts are the machine-readable confirmed value.
from apply_visual_verification import (  # noqa: E402
    PAYEE_CONFIRMATIONS,
    SIGNATURE_CONFIRMATIONS,
    _recompute_verdict,
)


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


def _apply_standing_overrides(record_id: str, cheque: NormalizedCheque,
                              result) -> bool:
    """Re-apply any standing human-confirmed value on top of a freshly
    rebuilt cheque/result, mutating both in place. Returns whether anything
    was applied. Mirrors apply_visual_verification.py's own two shapes:
    signature is a field-level correction, payee is a rule-level override."""
    applied = False

    if record_id in SIGNATURE_CONFIRMATIONS:
        cheque.signature = normalize_signature(
            None, detected=True, confidence=cheque.signature.confidence)
        result.rules = [
            RuleResult("signature", RuleStatus.PASS,
                      "signature detected in signature region",
                      cheque.signature.confidence)
            if r.rule_id == "signature" else r
            for r in result.rules
        ]
        applied = True

    if record_id in PAYEE_CONFIRMATIONS:
        confirmed, why = PAYEE_CONFIRMATIONS[record_id]
        result.rules = [
            RuleResult("payee", RuleStatus.PASS,
                      f"{cheque.payee.raw_text!r} (OCR) visually confirmed as "
                      f"{confirmed!r} against the source image — manual "
                      f"review override; {why}", cheque.payee.confidence)
            if r.rule_id == "payee" else r
            for r in result.rules
        ]
        applied = True

    if applied:
        result.verdict = _recompute_verdict(result.rules)
    return applied


def main() -> int:
    report.ensure_report_directory()
    _backup(report.CHEQUES_JSON)

    records = report.load_records()
    cfg = Config()

    before_valid = sum(1 for r in records if r.get("verdict") == "VALID")
    before_review = sum(1 for r in records if r.get("verdict") == "REVIEW")
    before_invalid = sum(1 for r in records if r.get("verdict") == "INVALID")
    changed = skipped = 0
    overlaid = 0
    verdict_transitions: dict[tuple[str, str], int] = {}

    for i, record in enumerate(records):
        if record.get("ruleset_version") == TARGET_VERSION:
            skipped += 1
            continue

        cheque = _rebuild_cheque(record)
        processed_time = datetime.fromisoformat(record["processed_time"])
        result = validate(cheque, cfg)

        record_id = record["record_id"]
        overlay_applied = _apply_standing_overrides(record_id, cheque, result)
        if overlay_applied:
            overlaid += 1

        rebuilt = report.create_report_record(
            cheque, result, source_file=record["source_file"],
            source_hash=record["source_hash"], processed_time=processed_time,
            rotation=record.get("rotation"))
        rebuilt["record_id"] = record_id

        old_verdict, new_verdict = record.get("verdict"), rebuilt["verdict"]
        if old_verdict != new_verdict:
            key = (old_verdict, new_verdict)
            verdict_transitions[key] = verdict_transitions.get(key, 0) + 1
            print(f"[{record_id}] {record.get('source_file')}: "
                 f"{old_verdict} -> {new_verdict}"
                 f"{' (overlay re-applied)' if overlay_applied else ''}")
        elif overlay_applied:
            print(f"[{record_id}] {record.get('source_file')}: "
                 f"{new_verdict} (overlay re-applied on top of re-derived fields)")

        records[i] = rebuilt
        changed += 1
        report.save_records(records)
        report.append_audit_event({
            "timestamp": datetime.now().astimezone().isoformat(),
            "event": "ruleset_revalidation",
            "record_id": record_id,
            "from_ruleset": record.get("ruleset_version"),
            "to_ruleset": TARGET_VERSION,
            "old_verdict": old_verdict,
            "new_verdict": new_verdict,
            "standing_override_reapplied": overlay_applied,
        })

    # --- explicit override-integrity check -------------------------------
    # The 1.3.0 -> 1.4.0 revalidation once discarded 8 standing overrides
    # outright. Don't just claim the layering fix works this time - check
    # every override this script knows how to re-apply actually landed.
    by_id = {r["record_id"]: r for r in records}
    override_failures: list[str] = []
    for record_id in SIGNATURE_CONFIRMATIONS:
        rec = by_id.get(record_id)
        if rec is None:
            override_failures.append(f"{record_id}: record missing entirely")
            continue
        sig_rule = (rec.get("rules") or {}).get("signature")
        if sig_rule is None or sig_rule.get("status") != "PASS":
            override_failures.append(
                f"{record_id}: signature override lost "
                f"(rule status = {sig_rule.get('status') if sig_rule else 'MISSING'})")
    for record_id in PAYEE_CONFIRMATIONS:
        rec = by_id.get(record_id)
        if rec is None:
            override_failures.append(f"{record_id}: record missing entirely")
            continue
        payee_rule = (rec.get("rules") or {}).get("payee")
        if payee_rule is None or payee_rule.get("status") != "PASS":
            override_failures.append(
                f"{record_id}: payee override lost "
                f"(rule status = {payee_rule.get('status') if payee_rule else 'MISSING'})")
    # Rotation overrides live in the `rotation` field this script always
    # copies forward from the pre-revalidation record (never regenerates
    # it) - confirm that provenance is still marked as an operator override.
    for rec in records:
        rotation = rec.get("rotation") or {}
        if rotation.get("source") == "operator_override" and not rotation.get("detector_note"):
            override_failures.append(
                f"{rec['record_id']}: rotation override present but lost its "
                f"detector_note provenance")

    report.regenerate_report()

    after_valid = sum(1 for r in records if r.get("verdict") == "VALID")
    after_review = sum(1 for r in records if r.get("verdict") == "REVIEW")
    after_invalid = sum(1 for r in records if r.get("verdict") == "INVALID")

    print()
    print(f"Re-validated : {changed}")
    print(f"Already on {TARGET_VERSION} (skipped) : {skipped}")
    print(f"Standing overrides re-applied on top : {overlaid}")
    for (old_v, new_v), count in sorted(verdict_transitions.items()):
        print(f"Flipped {old_v} -> {new_v} : {count}")
    print(f"Valid   before -> after : {before_valid} -> {after_valid}")
    print(f"Review  before -> after : {before_review} -> {after_review}")
    print(f"Invalid before -> after : {before_invalid} -> {after_invalid}")
    print()
    if override_failures:
        print(f"OVERRIDE INTEGRITY CHECK: FAILED ({len(override_failures)} issue(s))")
        for msg in override_failures:
            print(f"  - {msg}")
    else:
        print(f"OVERRIDE INTEGRITY CHECK: PASSED "
             f"({len(SIGNATURE_CONFIRMATIONS)} signature + "
             f"{len(PAYEE_CONFIRMATIONS)} payee standing overrides intact)")
    return 1 if override_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
