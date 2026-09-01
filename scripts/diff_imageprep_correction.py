r"""Diff, per field per record, DI's extraction from the STORED (sideways)
input against DI's extraction from the CORRECTED (imageprep-rotated,
deskewed) input produced by experiment_imageprep_di_comparison.py.

Read-only: never touches reports/cheques.json, never recomputes a verdict,
never changes a rule. Reads raw/*.json (stored) and raw_corrected/*.json
(new) side by side and reports what changed.

Run (after experiment_imageprep_di_comparison.py has populated
raw_corrected/):
    python scripts/diff_imageprep_correction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chequemate.extract import first_document, to_normalized  # noqa: E402
from chequemate.models import ParseStatus, RuleStatus  # noqa: E402
from chequemate.rules import Config, check_payee  # noqa: E402

RAW_DIR = ROOT / "raw"
RAW_CORRECTED_DIR = ROOT / "raw_corrected"
BATCH_FILES = [f"20260820113647241_{i:04d}.jpg" for i in range(1, 23)]

CFG = Config()


def _load(stem: str, directory: Path):
    path = directory / f"{stem}.json"
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    cheque = to_normalized(first_document(raw))
    return raw, cheque


def main() -> int:
    rows = []
    missing_corrected = []

    for name in BATCH_FILES:
        stem = Path(name).stem
        old = _load(stem, RAW_DIR)
        new = _load(stem, RAW_CORRECTED_DIR)
        if old is None:
            print(f"WARNING: no stored raw response for {name}", file=sys.stderr)
            continue
        if new is None:
            missing_corrected.append(name)
            continue
        rows.append((name, old[1], new[1]))

    print(f"Records compared: {len(rows)} / {len(BATCH_FILES)}")
    if missing_corrected:
        print(f"Missing corrected response (experiment didn't run/failed prep "
             f"for these): {missing_corrected}")
    print()

    # -------------------------------------------------------------------
    # WordAmount: parseability transition
    # -------------------------------------------------------------------
    print("=" * 78)
    print("WordAmount: garbled -> parseable?")
    print("=" * 78)
    aw_improved = aw_unchanged = aw_regressed = 0
    for name, old_cheque, new_cheque in rows:
        old_ok = old_cheque.amount_words.parse_status is ParseStatus.OK
        new_ok = new_cheque.amount_words.parse_status is ParseStatus.OK
        if not old_ok and new_ok:
            aw_improved += 1
            print(f"  IMPROVED  {name}")
            print(f"      old ({old_cheque.amount_words.parse_status.value}): "
                 f"{old_cheque.amount_words.raw_text!r}")
            print(f"      new (OK -> {new_cheque.amount_words.value}): "
                 f"{new_cheque.amount_words.raw_text!r}")
        elif old_ok and not new_ok:
            aw_regressed += 1
            print(f"  REGRESSED {name}")
            print(f"      old (OK -> {old_cheque.amount_words.value}): "
                 f"{old_cheque.amount_words.raw_text!r}")
            print(f"      new ({new_cheque.amount_words.parse_status.value}): "
                 f"{new_cheque.amount_words.raw_text!r}")
        else:
            aw_unchanged += 1

    # -------------------------------------------------------------------
    # PayTo: check_payee rule status transition + named cases
    # -------------------------------------------------------------------
    print()
    print("=" * 78)
    print("PayTo: check_payee() outcome")
    print("=" * 78)
    pt_improved = pt_unchanged = pt_regressed = 0
    for name, old_cheque, new_cheque in rows:
        old_status = check_payee(old_cheque, CFG).status
        new_status = check_payee(new_cheque, CFG).status
        old_pass = old_status is RuleStatus.PASS
        new_pass = new_status is RuleStatus.PASS
        tag = ""
        if not old_pass and new_pass:
            pt_improved += 1
            tag = "IMPROVED "
        elif old_pass and not new_pass:
            pt_regressed += 1
            tag = "REGRESSED"
        else:
            pt_unchanged += 1
        if name in ("20260820113647241_0001.jpg", "20260820113647241_0016.jpg") or tag:
            print(f"  {tag or '         '} {name}  {old_status.value} -> {new_status.value}")
            print(f"      old: {old_cheque.payee.raw_text!r}")
            print(f"      new: {new_cheque.payee.raw_text!r}")

    # -------------------------------------------------------------------
    # Memo: is the field returned at all?
    # -------------------------------------------------------------------
    print()
    print("=" * 78)
    print("Memo: field returned at all?")
    print("=" * 78)
    memo_improved = memo_unchanged = memo_regressed = 0
    for name, old_cheque, new_cheque in rows:
        old_present = old_cheque.memo.parse_status is ParseStatus.OK
        new_present = new_cheque.memo.parse_status is ParseStatus.OK
        if not old_present and new_present:
            memo_improved += 1
            print(f"  IMPROVED  {name}: (absent) -> {new_cheque.memo.raw_text!r}")
        elif old_present and not new_present:
            memo_regressed += 1
            print(f"  REGRESSED {name}: {old_cheque.memo.raw_text!r} -> (absent)")
        else:
            memo_unchanged += 1

    # -------------------------------------------------------------------
    # PayerSignatures: does a verdict now exist at all?
    # -------------------------------------------------------------------
    print()
    print("=" * 78)
    print("PayerSignatures: any verdict returned at all?")
    print("=" * 78)
    sig_improved = sig_unchanged = sig_regressed = 0
    for name, old_cheque, new_cheque in rows:
        old_has_verdict = old_cheque.signature.parse_status is ParseStatus.OK
        new_has_verdict = new_cheque.signature.parse_status is ParseStatus.OK
        if not old_has_verdict and new_has_verdict:
            sig_improved += 1
            print(f"  IMPROVED  {name}: (no verdict) -> {new_cheque.signature.value}")
        elif old_has_verdict and not new_has_verdict:
            sig_regressed += 1
            print(f"  REGRESSED {name}: {old_cheque.signature.value} -> (no verdict)")
        else:
            sig_unchanged += 1
            if old_has_verdict and new_has_verdict and old_cheque.signature.value != new_cheque.signature.value:
                print(f"  FLIPPED (both have a verdict, but it changed) {name}: "
                     f"{old_cheque.signature.value} -> {new_cheque.signature.value}")

    # -------------------------------------------------------------------
    # Confidence: up / down / flat, across all fields
    # -------------------------------------------------------------------
    conf_up = conf_down = conf_flat = conf_skipped = 0
    for name, old_cheque, new_cheque in rows:
        for field_name in ("payee", "amount_numeric", "amount_words", "cheque_date",
                           "memo", "signature"):
            old_conf = getattr(old_cheque, field_name).confidence
            new_conf = getattr(new_cheque, field_name).confidence
            if old_conf is None or new_conf is None:
                conf_skipped += 1
                continue
            if new_conf > old_conf + 1e-9:
                conf_up += 1
            elif new_conf < old_conf - 1e-9:
                conf_down += 1
            else:
                conf_flat += 1

    print()
    print("=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)
    print(f"{'field':<18}{'improved':<10}{'unchanged':<11}{'regressed':<10}")
    print(f"{'WordAmount':<18}{aw_improved:<10}{aw_unchanged:<11}{aw_regressed:<10}"
         f"{'  <-- REGRESSION' if aw_regressed else ''}")
    print(f"{'PayTo':<18}{pt_improved:<10}{pt_unchanged:<11}{pt_regressed:<10}"
         f"{'  <-- REGRESSION' if pt_regressed else ''}")
    print(f"{'Memo':<18}{memo_improved:<10}{memo_unchanged:<11}{memo_regressed:<10}"
         f"{'  <-- REGRESSION' if memo_regressed else ''}")
    print(f"{'PayerSignatures':<18}{sig_improved:<10}{sig_unchanged:<11}{sig_regressed:<10}"
         f"{'  <-- REGRESSION' if sig_regressed else ''}")
    print()
    print(f"Confidence (all fields, both sides had a value): "
         f"up={conf_up}  down={conf_down}  flat={conf_flat}  "
         f"(skipped, one side missing a value)={conf_skipped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
