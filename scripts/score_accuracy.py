r"""Score reports/cheques.json against tests/ground_truth.json.

Read-only: never mutates cheques.json or ground_truth.json. For every field
in a ground-truth record that isn't null, compares the pipeline's extracted
value (after normalize.py) to the hand-keyed true value and reports:

  - per-field accuracy across the batch
  - a per-record grid (Y = match, N = mismatch, - = skipped/null truth)

A null ground-truth value is skipped, not counted as a failure - the harness
only scores what a human has actually hand-keyed so far.

Run:
    python scripts/score_accuracy.py
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chequemate.normalize import canonical_payee  # noqa: E402
from chequemate import report  # noqa: E402

GROUND_TRUTH_JSON = ROOT / "tests" / "ground_truth.json"

FIELDS = ["payee", "amount_numeric", "amount_words", "cheque_date",
         "signature", "memo"]


def _norm_text(text: str | None) -> str:
    return " ".join((text or "").split()).lower()


def _decimal_or_none(text: str | None) -> Decimal | None:
    if text is None:
        return None
    try:
        return Decimal(str(text))
    except InvalidOperation:
        return None


def _compare_payee(truth: str, record: dict) -> bool:
    return canonical_payee(truth) == (record.get("payee_normalized") or "")


def _compare_amount(field: str, truth: str, record: dict) -> bool:
    return _decimal_or_none(truth) == _decimal_or_none(record.get(field))


def _compare_date(truth: str, record: dict) -> bool:
    return truth == record.get("cheque_date")


def _compare_signature(truth: bool, record: dict) -> bool:
    return bool(truth) == bool(record.get("signature_detected"))


def _compare_memo(truth: str, record: dict) -> bool:
    return _norm_text(truth) == _norm_text(record.get("memo"))


_COMPARATORS = {
    "payee": _compare_payee,
    "amount_numeric": lambda t, r: _compare_amount("amount_numeric", t, r),
    "amount_words": lambda t, r: _compare_amount("amount_words", t, r),
    "cheque_date": _compare_date,
    "signature": _compare_signature,
    "memo": _compare_memo,
}


def main() -> int:
    ground_truth = json.loads(GROUND_TRUTH_JSON.read_text(encoding="utf-8"))
    records_by_id = {r["record_id"]: r for r in report.load_records()}

    field_matches = {f: 0 for f in FIELDS}
    field_scored = {f: 0 for f in FIELDS}
    grid_rows: list[tuple[str, dict[str, str]]] = []
    missing_records: list[str] = []

    for truth_rec in ground_truth["records"]:
        record_id = truth_rec["record_id"]
        record = records_by_id.get(record_id)
        if record is None:
            missing_records.append(record_id)
            continue

        row: dict[str, str] = {}
        for field in FIELDS:
            truth_value = truth_rec.get(field)
            if truth_value is None:
                row[field] = "-"
                continue
            field_scored[field] += 1
            is_match = _COMPARATORS[field](truth_value, record)
            if is_match:
                field_matches[field] += 1
            row[field] = "Y" if is_match else "N"
        grid_rows.append((record_id, row))

    print("Per-field accuracy (of hand-keyed, non-null ground truth):")
    for field in FIELDS:
        scored = field_scored[field]
        matches = field_matches[field]
        pct = f"{100 * matches / scored:5.1f}%" if scored else "  n/a "
        print(f"  {field:<15} {matches:>3}/{scored:<3} {pct}")

    total_scored = sum(field_scored.values())
    total_matches = sum(field_matches.values())
    overall = f"{100 * total_matches / total_scored:.1f}%" if total_scored else "n/a"
    print(f"  {'OVERALL':<15} {total_matches:>3}/{total_scored:<3} {overall:>6}")

    print()
    header = f"{'record_id':<20}" + "".join(f"{f:<16}" for f in FIELDS)
    print(header)
    for record_id, row in grid_rows:
        line = f"{record_id:<20}" + "".join(f"{row[f]:<16}" for f in FIELDS)
        print(line)

    if missing_records:
        print()
        print(f"WARNING: {len(missing_records)} ground-truth record_id(s) not "
             f"found in reports/cheques.json: {', '.join(missing_records)}")

    if total_scored == 0:
        print()
        print("No ground truth has been hand-keyed yet (all fields null) - "
             "nothing to score.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
