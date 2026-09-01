r"""Evaluate TrOCR field-level secondary verification against
reports/cheques.json's own ocr_verifications data, plus (where hand-keyed)
tests/ground_truth.json - same ground-truth file and read-only conventions
as scripts/score_accuracy.py, extended with the OCR-verification-specific
metrics this feature needs.

Reports per eligible field (payee / amount_words / memo), never collapsed
into one "overall accuracy" number:

  - number eligible (TrOCR verification ran for this field on this record)
  - number attempted (TrOCR actually executed, whether or not it succeeded)
  - crop acceptance rate
  - TrOCR completion rate
  - exact / normalized / allowlist agreement rates (reported separately,
    not summed into one "agreement" figure)
  - disagreement rate
  - DI-only count, both-undetermined count
  - where tests/ground_truth.json has a hand-keyed value: DI exact match,
    TrOCR exact match, selected-output exact match (three separate figures)

If no ground truth has been hand-keyed for a field, its "vs. ground truth"
numbers are reported as "n/a - agreement analysis only", never silently
treated as 0% or skipped without saying so - this run is agreement
analysis (DI vs. TrOCR), not accuracy analysis, until real ground truth
exists (same honesty rule score_accuracy.py already follows).

Run:
    python scripts/evaluate_trocr.py
    python scripts/evaluate_trocr.py --save-baseline   # record this run as
                                                        # the regression baseline
    python scripts/evaluate_trocr.py --json out.json --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chequemate import report  # noqa: E402
from chequemate.ocr_verify import (  # noqa: E402
    normalize_amount_words_for_comparison, normalize_memo_for_comparison,
    normalize_payee_for_comparison,
)

GROUND_TRUTH_JSON = ROOT / "tests" / "ground_truth.json"
BASELINE_JSON = ROOT / "reports" / "trocr_evaluation_baseline.json"

FIELDS = ("payee", "amount_words", "memo")
AGREEMENT_STATUSES = ("AGREE_EXACT", "AGREE_NORMALIZED", "AGREE_ALLOWLIST")

_TRUTH_COMPARATORS = {
    "payee": lambda truth, value: (normalize_payee_for_comparison(truth)
                                   == normalize_payee_for_comparison(value)),
    "amount_words": lambda truth, value: (
        normalize_amount_words_for_comparison(truth)[0] is not None
        and normalize_amount_words_for_comparison(truth)[0]
        == normalize_amount_words_for_comparison(value)[0]),
    "memo": lambda truth, value: (normalize_memo_for_comparison(truth)
                                  == normalize_memo_for_comparison(value)),
}

# ground_truth.json's field names differ slightly from ocr_verify's
# (amount_words vs the JSON key "amount_words" - same; "memo" - same;
# "payee" - same) - kept as an explicit map anyway so a future rename on
# either side fails loudly here rather than silently mis-scoring.
_GROUND_TRUTH_KEY = {"payee": "payee", "amount_words": "amount_words", "memo": "memo"}


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def evaluate(records: list[dict], ground_truth: dict | None) -> dict:
    truth_by_id = {}
    if ground_truth:
        truth_by_id = {r["record_id"]: r for r in ground_truth.get("records", [])}

    per_field = {f: {
        "eligible": 0, "attempted": 0, "crop_accepted": 0,
        "trocr_completed": 0, "agree_exact": 0, "agree_normalized": 0,
        "agree_allowlist": 0, "disagree": 0, "di_only": 0,
        "both_undetermined": 0, "crop_rejected": 0, "trocr_not_run": 0,
        "trocr_failed": 0,
        "truth_scored": 0, "di_exact_matches": 0, "trocr_exact_matches": 0,
        "selected_exact_matches": 0,
    } for f in FIELDS}

    for record in records:
        ocr = record.get("ocr_verifications") or {}
        for field_name in FIELDS:
            fv = ocr.get(field_name)
            if fv is None:
                continue  # TrOCR verification did not run at all for this record
            stats = per_field[field_name]
            stats["eligible"] += 1

            crop_status = fv["crop"]["status"]
            secondary_status = fv["secondary"]["status"]
            comparison_status = fv["comparison"]["status"]

            if crop_status in ("accepted", "adjusted"):
                stats["crop_accepted"] += 1
            if crop_status == "rejected":
                stats["crop_rejected"] += 1
            if secondary_status in ("completed", "failed", "undetermined"):
                stats["attempted"] += 1
            if secondary_status == "completed":
                stats["trocr_completed"] += 1
            if secondary_status == "failed":
                stats["trocr_failed"] += 1
            if crop_status == "not_run" and secondary_status == "not_run":
                stats["trocr_not_run"] += 1

            if comparison_status == "AGREE_EXACT":
                stats["agree_exact"] += 1
            elif comparison_status == "AGREE_NORMALIZED":
                stats["agree_normalized"] += 1
            elif comparison_status == "AGREE_ALLOWLIST":
                stats["agree_allowlist"] += 1
            elif comparison_status == "DISAGREE":
                stats["disagree"] += 1
            elif comparison_status == "DI_ONLY":
                stats["di_only"] += 1
            elif comparison_status == "BOTH_UNDETERMINED":
                stats["both_undetermined"] += 1

            truth_rec = truth_by_id.get(record["record_id"])
            truth_value = (truth_rec or {}).get(_GROUND_TRUTH_KEY[field_name])
            if truth_value is not None:
                stats["truth_scored"] += 1
                comparator = _TRUTH_COMPARATORS[field_name]
                if comparator(truth_value, fv["primary"]["raw_value"]):
                    stats["di_exact_matches"] += 1
                if fv["secondary"]["raw_value"] is not None and \
                        comparator(truth_value, fv["secondary"]["raw_value"]):
                    stats["trocr_exact_matches"] += 1
                selected = fv["comparison"]["selected_display_value"]
                if selected is not None and comparator(truth_value, selected):
                    stats["selected_exact_matches"] += 1

    results = {}
    for field_name, s in per_field.items():
        eligible = s["eligible"]
        results[field_name] = {
            "number_eligible": eligible,
            "number_attempted": s["attempted"],
            "crop_acceptance_rate": _rate(s["crop_accepted"], eligible),
            "trocr_completion_rate": _rate(s["trocr_completed"], s["attempted"]),
            "exact_agreement_rate": _rate(s["agree_exact"], eligible),
            "normalized_agreement_rate": _rate(s["agree_normalized"], eligible),
            "allowlist_agreement_rate": _rate(s["agree_allowlist"], eligible),
            "disagreement_rate": _rate(s["disagree"], eligible),
            "di_only_count": s["di_only"],
            "both_undetermined_count": s["both_undetermined"],
            "crop_rejected_count": s["crop_rejected"],
            "trocr_not_run_count": s["trocr_not_run"],
            "trocr_failed_count": s["trocr_failed"],
            "ground_truth_scored": s["truth_scored"],
            "analysis_kind": ("accuracy" if s["truth_scored"] else
                             "agreement (no ground truth hand-keyed for this field yet)"),
            "di_exact_match_rate": (_rate(s["di_exact_matches"], s["truth_scored"])
                                    if s["truth_scored"] else None),
            "trocr_exact_match_rate": (_rate(s["trocr_exact_matches"], s["truth_scored"])
                                       if s["truth_scored"] else None),
            "selected_output_exact_match_rate": (
                _rate(s["selected_exact_matches"], s["truth_scored"])
                if s["truth_scored"] else None),
        }
    return results


def _print_report(results: dict) -> None:
    for field_name, r in results.items():
        print(f"\n{field_name}")
        print(f"  eligible={r['number_eligible']}  attempted={r['number_attempted']}")
        print(f"  crop_acceptance_rate={r['crop_acceptance_rate']}  "
             f"trocr_completion_rate={r['trocr_completion_rate']}")
        print(f"  exact_agreement={r['exact_agreement_rate']}  "
             f"normalized_agreement={r['normalized_agreement_rate']}  "
             f"allowlist_agreement={r['allowlist_agreement_rate']}  "
             f"disagreement={r['disagreement_rate']}")
        print(f"  di_only={r['di_only_count']}  both_undetermined={r['both_undetermined_count']}  "
             f"crop_rejected={r['crop_rejected_count']}  trocr_not_run={r['trocr_not_run_count']}  "
             f"trocr_failed={r['trocr_failed_count']}")
        print(f"  [{r['analysis_kind']}] ground_truth_scored={r['ground_truth_scored']}  "
             f"di_exact_match={r['di_exact_match_rate']}  "
             f"trocr_exact_match={r['trocr_exact_match_rate']}  "
             f"selected_output_exact_match={r['selected_output_exact_match_rate']}")


def _print_regression(current: dict, baseline: dict) -> None:
    print("\nRegression vs. baseline "
         f"({BASELINE_JSON.relative_to(ROOT)}):")
    for field_name in FIELDS:
        cur, base = current.get(field_name, {}), baseline.get(field_name, {})
        for metric in ("exact_agreement_rate", "normalized_agreement_rate",
                      "disagreement_rate", "selected_output_exact_match_rate"):
            c, b = cur.get(metric), base.get(metric)
            if c is None or b is None:
                continue
            delta = round(c - b, 4)
            flag = " <-- REGRESSION" if metric != "disagreement_rate" and delta < -0.001 else \
                  " <-- REGRESSION" if metric == "disagreement_rate" and delta > 0.001 else ""
            print(f"  {field_name}.{metric}: {b} -> {c} (delta {delta:+.4f}){flag}")


def _write_csv(results: dict, path: Path) -> None:
    fieldnames = ["field_name"] + list(next(iter(results.values())).keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for field_name, r in results.items():
            writer.writerow({"field_name": field_name, **r})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", metavar="PATH",
                    default=str(ROOT / "reports" / "trocr_evaluation.json"))
    ap.add_argument("--csv", metavar="PATH",
                    default=str(ROOT / "reports" / "trocr_evaluation.csv"))
    ap.add_argument("--save-baseline", action="store_true",
                    help="record this run's results as the regression baseline "
                         "for future runs (does not overwrite silently on later "
                         "runs unless passed again explicitly)")
    args = ap.parse_args(argv)

    records = report.load_records()
    ground_truth = (json.loads(GROUND_TRUTH_JSON.read_text(encoding="utf-8"))
                    if GROUND_TRUTH_JSON.is_file() else None)

    results = evaluate(records, ground_truth)
    _print_report(results)

    if BASELINE_JSON.is_file() and not args.save_baseline:
        baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
        _print_regression(results, baseline)
    elif not BASELINE_JSON.is_file():
        print("\nNo regression baseline yet - run with --save-baseline to "
             "record one.")

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_csv(results, Path(args.csv))
    print(f"\nWrote {args.json}")
    print(f"Wrote {args.csv}")

    if args.save_baseline:
        BASELINE_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved regression baseline to {BASELINE_JSON.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
