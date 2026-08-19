"""Persistent, cumulative reporting: JSON records -> HTML dashboard.

Every validated cheque becomes one privacy-safe record in reports/cheques.json,
one line in reports/audit.jsonl, and a row in the regenerated
reports/chequemate_report.html. JSON is the source of truth; the HTML is
always fully rebuilt from it, never patched in place.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from .models import NormalizedCheque, RuleResult, ValidationResult
from .validate import RULE_SET_VERSION

MODEL_ID = "prebuilt-check.us"

SCHEMA_VERSION = "1.0"

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
CHEQUES_JSON = REPORTS_DIR / "cheques.json"
REVIEWS_JSON = REPORTS_DIR / "reviews.json"
AUDIT_LOG = REPORTS_DIR / "audit.jsonl"
HTML_REPORT = REPORTS_DIR / "chequemate_report.html"

_RECORD_ID_RE = re.compile(r"^CHQ-(\d{8})-(\d{4})$")


class ReportError(RuntimeError):
    """A report data file exists but cannot be safely read or written."""


@dataclass
class ReportOutcome:
    record: dict
    created: bool


# ---------------------------------------------------------------------------
# filesystem: directory setup + atomic, safe JSON I/O
# ---------------------------------------------------------------------------

def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` without ever leaving a partially-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


def ensure_report_directory() -> Path:
    """Create reports/ and its data files if this is the first run."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not CHEQUES_JSON.exists():
        _atomic_write_text(CHEQUES_JSON, "[]")
    if not REVIEWS_JSON.exists():
        _atomic_write_text(REVIEWS_JSON, "[]")
    if not AUDIT_LOG.exists():
        AUDIT_LOG.touch()
    return REPORTS_DIR


def _load_json_array(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReportError(
            f"{path} contains invalid JSON and was left untouched so no "
            f"records are lost. Fix or move the file, then retry: {exc}"
        ) from exc
    if not isinstance(data, list):
        raise ReportError(f"{path} must contain a JSON array, found "
                          f"{type(data).__name__}")
    return data


def load_records() -> list[dict]:
    ensure_report_directory()
    return _load_json_array(CHEQUES_JSON)


def save_records(records: list[dict]) -> None:
    ensure_report_directory()
    _atomic_write_text(CHEQUES_JSON, json.dumps(records, indent=2, ensure_ascii=False))


def load_reviews() -> list[dict]:
    ensure_report_directory()
    return _load_json_array(REVIEWS_JSON)


def save_reviews(reviews: list[dict]) -> None:
    ensure_report_directory()
    _atomic_write_text(REVIEWS_JSON, json.dumps(reviews, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# hashing (duplicate-ingestion protection)
# ---------------------------------------------------------------------------

def hash_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes. Works for cheque images and replay JSON alike."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# record IDs
# ---------------------------------------------------------------------------

def _next_record_id(records: list[dict], today: date) -> str:
    day = today.strftime("%Y%m%d")
    prefix = f"CHQ-{day}-"
    existing_ids = {r.get("record_id") for r in records}
    max_seq = 0
    for r in records:
        m = _RECORD_ID_RE.match(r.get("record_id") or "")
        if m and m.group(1) == day:
            max_seq = max(max_seq, int(m.group(2)))

    seq = max_seq + 1
    candidate = f"{prefix}{seq:04d}"
    while candidate in existing_ids:
        seq += 1
        candidate = f"{prefix}{seq:04d}"
    return candidate


# ---------------------------------------------------------------------------
# JSON-safe conversion
# ---------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return str(value)


def _field_value(field) -> Any:
    """A field's typed value, JSON-safe, or None if it never parsed."""
    return _json_safe(field.value) if field.ok else None


# ---------------------------------------------------------------------------
# record creation
# ---------------------------------------------------------------------------

def create_report_record(cheque: NormalizedCheque, validation: ValidationResult,
                          source_file: str | Path, source_hash: str,
                          processed_time: datetime | None = None) -> dict:
    """Build one privacy-safe, JSON-safe record. Does not assign record_id."""
    processed_time = processed_time or datetime.now().astimezone()
    rules_by_id: dict[str, RuleResult] = {r.rule_id: r for r in validation.rules}

    def rule_conf(rule_id: str) -> float | None:
        r = rules_by_id.get(rule_id)
        return r.confidence if r else None

    rules = {
        rule_id: {
            "status": r.status.value,
            "message": r.evidence,
            "confidence": r.confidence,
        }
        for rule_id, r in rules_by_id.items()
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": None,
        "processed_time": processed_time.isoformat(),
        "source_file": Path(source_file).name,
        "source_hash": source_hash,
        "model": MODEL_ID,
        "ruleset_version": RULE_SET_VERSION,
        "verdict": validation.verdict.value,
        "payee": cheque.payee.raw_text,
        "payee_normalized": _field_value(cheque.payee),
        "amount_numeric": _field_value(cheque.amount_numeric),
        "amount_words": _field_value(cheque.amount_words),
        "cheque_date": _field_value(cheque.cheque_date),
        "signature_detected": _field_value(cheque.signature),
        "confidence": {
            "payee": rule_conf("payee"),
            "amount_numeric": rule_conf("amount_match"),
            "amount_words": rule_conf("amount_match"),
            "date": rule_conf("date"),
            "signature": rule_conf("signature"),
        },
        "raw_values": {
            "payee": cheque.payee.raw_text,
            "amount_numeric": cheque.amount_numeric.raw_text,
            "amount_words": cheque.amount_words.raw_text,
            "date": cheque.cheque_date.raw_text,
        },
        "rules": rules,
    }


def append_record(record: dict, today: date | None = None) -> ReportOutcome:
    """Append `record` unless its source_hash is already on file.

    Returns the stored record (existing one, on a duplicate) plus whether a
    new record was actually created.
    """
    ensure_report_directory()
    records = load_records()

    for existing in records:
        if existing.get("source_hash") and \
                existing.get("source_hash") == record.get("source_hash"):
            return ReportOutcome(record=existing, created=False)

    today = today or datetime.now().date()
    stored = dict(record)
    stored["record_id"] = _next_record_id(records, today)
    records.append(stored)
    save_records(records)
    return ReportOutcome(record=stored, created=True)


# ---------------------------------------------------------------------------
# audit trail
# ---------------------------------------------------------------------------

def append_audit_event(event: dict) -> None:
    """Append one immutable, newline-delimited JSON event. Never rewrites history."""
    ensure_report_directory()
    line = json.dumps(event, ensure_ascii=False)
    with open(AUDIT_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

_STATUS_COLOR = {
    "VALID": "#0ca30c",
    "INVALID": "#d03b3b",
    "PASS": "#0ca30c",
    "FAIL": "#d03b3b",
    "UNABLE": "#c98500",
}
_STATUS_ICON = {
    "VALID": "✓", "PASS": "✓",
    "INVALID": "✗", "FAIL": "✗",
    "UNABLE": "?",
}

_CSS = """
:root {
  color-scheme: light;
  --surface-1: #fcfcfb;
  --page: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #898781;
  --gridline: #e1e0d9;
  --border: rgba(11,11,11,0.10);
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --gridline: #2c2c2a;
    --border: rgba(255,255,255,0.10);
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--text-primary);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  padding: 32px;
}
h1 { font-size: 22px; margin: 0 0 4px; }
.subtitle { color: var(--text-secondary); margin: 0 0 28px; font-size: 13px; }
.tiles { display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }
.tile {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 24px;
  min-width: 140px;
}
.tile .label { color: var(--text-secondary); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.tile .value { font-size: 32px; font-weight: 600; font-variant-numeric: proportional-nums; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--gridline); font-variant-numeric: tabular-nums; vertical-align: top; }
th { color: var(--text-secondary); font-size: 12px; text-transform: uppercase; letter-spacing: .03em; font-weight: 600; }
tr:last-child td { border-bottom: none; }
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.rule-line { display: flex; gap: 6px; align-items: baseline; margin: 2px 0; }
.rule-icon { font-weight: 700; width: 1em; display: inline-block; }
.rule-id { color: var(--text-secondary); min-width: 92px; display: inline-block; }
.mono { font-variant-numeric: tabular-nums; }
.empty { color: var(--text-secondary); padding: 24px; text-align: center; }
footer { color: var(--muted); font-size: 12px; margin-top: 20px; }
details > summary { cursor: pointer; color: var(--text-secondary); }
"""


def _badge(status: str) -> str:
    color = _STATUS_COLOR.get(status, "#898781")
    icon = _STATUS_ICON.get(status, "")
    return (f'<span class="badge" style="background:{color}1a;color:{color}">'
           f'{icon} {html.escape(status)}</span>')


def _rule_rows(rules: dict) -> str:
    order = ["payee", "amount_match", "signature", "date"]
    ids = [r for r in order if r in rules] + \
        [r for r in rules if r not in order]
    parts = []
    for rid in ids:
        r = rules[rid]
        status = r.get("status", "")
        color = _STATUS_COLOR.get(status, "#898781")
        icon = _STATUS_ICON.get(status, "")
        conf = r.get("confidence")
        conf_txt = f" (conf {conf:.2f})" if isinstance(conf, (int, float)) else ""
        parts.append(
            '<div class="rule-line">'
            f'<span class="rule-icon" style="color:{color}">{icon}</span>'
            f'<span class="rule-id">{html.escape(rid)}</span>'
            f'<span>{html.escape(r.get("message", ""))}{html.escape(conf_txt)}</span>'
            '</div>')
    return "".join(parts)


def _fmt(value: Any) -> str:
    return html.escape(str(value)) if value is not None else \
        '<span class="mono" style="color:var(--muted)">&mdash;</span>'


def generate_html(records: list[dict], reviews: list[dict] | None = None) -> str:
    reviews = reviews or []
    reviews_by_record = {}
    for rev in reviews:
        rid = rev.get("record_id")
        if rid:
            reviews_by_record.setdefault(rid, []).append(rev)

    total = len(records)
    valid = sum(1 for r in records if r.get("verdict") == "VALID")
    invalid = total - valid

    tiles = f"""
    <div class="tiles">
      <div class="tile"><div class="label">Total Processed</div><div class="value">{total}</div></div>
      <div class="tile"><div class="label">Valid</div><div class="value" style="color:{_STATUS_COLOR['VALID']}">{valid}</div></div>
      <div class="tile"><div class="label">Invalid</div><div class="value" style="color:{_STATUS_COLOR['INVALID']}">{invalid}</div></div>
    </div>
    """

    if not records:
        body = '<div class="empty">No cheques processed yet.</div>'
    else:
        ordered = sorted(records, key=lambda r: r.get("processed_time", ""),
                         reverse=True)
        rows = []
        for rec in ordered:
            review_note = ""
            revs = reviews_by_record.get(rec.get("record_id"))
            if revs:
                notes = "; ".join(html.escape(str(v.get("note", ""))) for v in revs)
                review_note = f'<div class="rule-line" style="color:var(--muted)">reviewed: {notes}</div>'
            rows.append(f"""
            <tr>
              <td class="mono">{html.escape(str(rec.get("record_id", "")))}</td>
              <td class="mono">{html.escape(str(rec.get("processed_time", "")))}</td>
              <td>{html.escape(str(rec.get("source_file", "")))}</td>
              <td>{_badge(rec.get("verdict", ""))}</td>
              <td>{_fmt(rec.get("payee"))}</td>
              <td class="mono">{_fmt(rec.get("amount_numeric"))}</td>
              <td class="mono">{_fmt(rec.get("cheque_date"))}</td>
              <td>{"yes" if rec.get("signature_detected") else "no" if rec.get("signature_detected") is False else _fmt(None)}</td>
              <td>
                <details>
                  <summary>rules</summary>
                  {_rule_rows(rec.get("rules", {}))}
                  {review_note}
                </details>
              </td>
            </tr>""")
        body = f"""
        <table>
          <thead>
            <tr>
              <th>Record ID</th><th>Processed</th><th>Source File</th><th>Verdict</th>
              <th>Payee</th><th>Amount</th><th>Cheque Date</th><th>Signature</th><th>Rules</th>
            </tr>
          </thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
        """

    generated = datetime.now().astimezone().isoformat()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChequeMate Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>ChequeMate Validation Report</h1>
<p class="subtitle">Cumulative results across every cheque processed by this pipeline.</p>
{tiles}
{body}
<footer>Generated {html.escape(generated)} &middot; model {html.escape(MODEL_ID)} &middot; ruleset {html.escape(RULE_SET_VERSION)}</footer>
</body>
</html>
"""


def regenerate_report() -> Path:
    """Rebuild the HTML dashboard in full from the current JSON records."""
    records = load_records()
    reviews = load_reviews()
    _atomic_write_text(HTML_REPORT, generate_html(records, reviews))
    return HTML_REPORT


# ---------------------------------------------------------------------------
# high-level entry point for the CLI
# ---------------------------------------------------------------------------

def update_report(cheque: NormalizedCheque, validation: ValidationResult,
                  source_path: str | Path) -> ReportOutcome:
    """Record one validated cheque and regenerate the dashboard.

    Safe to call once per successfully-analyzed cheque; duplicates (same
    source file contents) are detected and skipped without altering the
    cheque's own validation result.
    """
    ensure_report_directory()
    source_hash = hash_file(source_path)
    record = create_report_record(cheque, validation, source_path, source_hash)
    outcome = append_record(record)

    if outcome.created:
        append_audit_event({
            "timestamp": outcome.record["processed_time"],
            "event": "cheque_processed",
            "record_id": outcome.record["record_id"],
            "source_file": outcome.record["source_file"],
            "verdict": outcome.record["verdict"],
            "ruleset_version": outcome.record["ruleset_version"],
            "model": outcome.record["model"],
        })
    else:
        append_audit_event({
            "timestamp": datetime.now().astimezone().isoformat(),
            "event": "duplicate_ingestion_skipped",
            "record_id": outcome.record["record_id"],
            "source_file": Path(source_path).name,
        })

    regenerate_report()
    return outcome
