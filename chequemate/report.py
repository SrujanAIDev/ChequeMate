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
# HTML dashboard — glassmorphic, client-rendered from embedded JSON
# ---------------------------------------------------------------------------

_CSS = """
:root{
  color-scheme: dark;
  --bg-a:#0b0f1e;
  --bg-b:#111827;
  --bg-c:#0d1b1a;
  --glass:rgba(255,255,255,0.055);
  --glass-strong:rgba(255,255,255,0.09);
  --glass-border:rgba(255,255,255,0.14);
  --text-primary:#f3f5f9;
  --text-secondary:#aab2c5;
  --muted:#707c93;
  --brand-a:#a78bfa;
  --brand-b:#22d3ee;
  --ok:#34d399;
  --ok-bg:rgba(52,211,153,0.16);
  --bad:#fb7185;
  --bad-bg:rgba(251,113,133,0.16);
  --warn:#fbbf24;
  --shadow:0 8px 32px rgba(0,0,0,0.35);
  --radius:16px;
}
*,*::before,*::after{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0;
  min-height:100vh;
  font:14px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif;
  color:var(--text-primary);
  background:
    radial-gradient(1100px 620px at 8% -10%, rgba(167,139,250,0.28), transparent 60%),
    radial-gradient(900px 560px at 100% 0%, rgba(34,211,238,0.22), transparent 55%),
    radial-gradient(1000px 700px at 50% 120%, rgba(52,211,153,0.14), transparent 60%),
    linear-gradient(160deg, var(--bg-a), var(--bg-b) 55%, var(--bg-c));
  background-attachment: fixed;
  padding: 28px clamp(16px, 4vw, 40px) 48px;
}
.glass{
  background: var(--glass);
  backdrop-filter: blur(20px) saturate(160%);
  -webkit-backdrop-filter: blur(20px) saturate(160%);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}
.appbar{
  display:flex; align-items:center; gap:18px; flex-wrap:wrap;
  padding:18px 24px; margin-bottom:18px;
}
.brand{display:flex; align-items:center; gap:10px}
.brand .dot{
  width:10px; height:10px; border-radius:50%;
  background: linear-gradient(135deg, var(--brand-a), var(--brand-b));
  box-shadow: 0 0 14px rgba(167,139,250,0.7);
}
.brand .word{font-size:20px; font-weight:700; letter-spacing:.2px}
.brand .word b{
  background: linear-gradient(135deg, var(--brand-a), var(--brand-b));
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.subtitle{color:var(--text-secondary); font-size:12.5px; padding-left:12px; border-left:1px solid var(--glass-border)}
.chips{display:flex; gap:8px; flex-wrap:wrap; margin-left:auto}
.chip{
  background: var(--glass-strong); border:1px solid var(--glass-border);
  border-radius:999px; padding:6px 13px; font-size:11.5px; color:var(--text-secondary);
  white-space:nowrap;
}
.chip b{color:var(--text-primary); font-weight:600}
.stats{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:16px}
@media (max-width:900px){.stats{grid-template-columns:repeat(2,1fr)}}
.tile{padding:16px 20px; position:relative; overflow:hidden}
.tile::before{
  content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
  background:var(--bar,var(--brand-a));
}
.tile.total{--bar:var(--brand-b)}
.tile.ok{--bar:var(--ok)}
.tile.bad{--bar:var(--bad)}
.tile .label{font-size:10.5px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted)}
.tile .value{font-size:30px; font-weight:700; margin-top:6px; font-variant-numeric:tabular-nums}
.tile.ok .value{color:var(--ok)}
.tile.bad .value{color:var(--bad)}
.tile.donut{display:flex; align-items:center; gap:14px}
.donut-svg{width:64px; height:64px; transform:rotate(-90deg); flex:none}
.donut-svg circle{fill:none; stroke-width:7}
.donut-svg .track{stroke:rgba(255,255,255,0.08)}
.donut-legend{font-size:11px; color:var(--text-secondary); line-height:1.7}
.donut-legend b{color:var(--text-primary)}
.dk{display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:6px}
.controls{
  display:flex; align-items:center; gap:10px; flex-wrap:wrap;
  padding:12px 18px; margin-bottom:16px;
}
.search{
  display:flex; align-items:center; gap:8px; flex:1; min-width:200px;
  background:var(--glass-strong); border:1px solid var(--glass-border);
  border-radius:10px; padding:8px 13px;
}
.search input{
  border:none; outline:none; background:transparent; color:var(--text-primary);
  font-size:12.5px; width:100%;
}
.search input::placeholder{color:var(--muted)}
select.ctl{
  background:var(--glass-strong); border:1px solid var(--glass-border);
  color:var(--text-primary); border-radius:10px; padding:8px 12px;
  font-size:12.5px; cursor:pointer;
}
.btn{
  border:none; border-radius:10px; padding:9px 16px; font-size:12.5px; font-weight:600;
  cursor:pointer; display:inline-flex; align-items:center; gap:6px; color:#0b0f1e;
  background:linear-gradient(135deg, var(--brand-a), var(--brand-b));
  transition:filter .15s, transform .15s;
}
.btn:hover{filter:brightness(1.08); transform:translateY(-1px)}
.resultcount{font-size:11.5px; color:var(--muted); margin-left:auto}
.table-card{padding:0; overflow:hidden}
.table-scroll{overflow-x:auto}
table{width:100%; border-collapse:collapse; min-width:920px}
thead th{
  position:sticky; top:0; background:rgba(15,20,35,0.86);
  backdrop-filter:blur(12px);
  color:var(--text-secondary); font-size:10.5px; text-transform:uppercase;
  letter-spacing:.06em; text-align:left; padding:12px 14px; cursor:pointer;
  border-bottom:1px solid var(--glass-border); white-space:nowrap; user-select:none;
}
thead th .si{opacity:.5; margin-left:3px}
tbody td{padding:11px 14px; border-bottom:1px solid rgba(255,255,255,0.06); font-size:12.5px; vertical-align:middle}
tbody tr.row{cursor:pointer; transition:background .15s}
tbody tr.row:hover{background:rgba(255,255,255,0.045)}
tbody tr.row.expanded{background:rgba(255,255,255,0.06)}
.cell-id{font-family:Consolas,monospace; font-size:11px; color:var(--text-secondary)}
.cell-payee b{display:block; color:var(--text-primary); font-size:12.5px; font-weight:600}
.mono{font-variant-numeric:tabular-nums}
.dash{color:var(--muted)}
.badge{
  display:inline-flex; align-items:center; gap:5px; padding:3px 10px;
  border-radius:999px; font-size:11px; font-weight:700;
}
.badge.ok{background:var(--ok-bg); color:var(--ok)}
.badge.bad{background:var(--bad-bg); color:var(--bad)}
.rule-dots{display:flex; gap:4px}
.rd{width:9px; height:9px; border-radius:2px; display:inline-block; flex:none}
.rd.PASS{background:var(--ok)}
.rd.FAIL{background:var(--bad)}
.rd.UNABLE{background:var(--warn)}
tr.detail-row{display:none}
tr.detail-row.open{display:table-row}
tr.detail-row td{
  background:rgba(0,0,0,0.18); padding:0; border-bottom:1px solid rgba(255,255,255,0.08);
}
.detail-inner{padding:16px 22px; display:grid; grid-template-columns:1.1fr 0.9fr; gap:22px}
@media (max-width:760px){.detail-inner{grid-template-columns:1fr}}
.detail-group h4{
  font-size:10.5px; text-transform:uppercase; letter-spacing:.07em;
  color:var(--muted); margin:0 0 8px;
}
.rule-line{display:flex; gap:8px; align-items:baseline; margin:5px 0; font-size:12px}
.rule-id{color:var(--text-secondary); min-width:100px; display:inline-block}
.conf-tag{color:var(--muted); font-size:10.5px}
.review-item{
  border-left:3px solid var(--brand-a); background:rgba(167,139,250,0.08);
  border-radius:0 8px 8px 0; padding:7px 11px; margin-bottom:7px; font-size:11.5px;
}
.review-item .m{color:var(--muted); font-size:10px; margin-top:2px}
.raw-line{font-size:11px; color:var(--text-secondary); margin:3px 0; font-family:Consolas,monospace}
.empty-row td{text-align:center; color:var(--muted); padding:40px 0}
footer{
  margin-top:18px; padding:14px 20px; font-size:11px; color:var(--muted);
  display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;
}
::-webkit-scrollbar{width:10px; height:10px}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.16); border-radius:6px}
::-webkit-scrollbar-track{background:transparent}
"""

_JS_TEMPLATE = r"""
const RECORDS = __RECORDS_JSON__;
const REVIEWS = __REVIEWS_JSON__;
const RULE_COUNTS = __RULE_COUNTS_JSON__;

const RULE_ORDER = ['payee', 'amount_match', 'date', 'signature'];
const RULE_LABEL = {payee:'Payee', amount_match:'Amount', date:'Date', signature:'Signature'};

const reviewsByRecord = {};
REVIEWS.forEach(function(rev){
  if(!rev.record_id) return;
  (reviewsByRecord[rev.record_id] = reviewsByRecord[rev.record_id] || []).push(rev);
});

let sortKey = 'processed_time';
let sortAsc = false;
let expandedId = null;
let view = RECORDS.slice();

function esc(s){
  if(s === null || s === undefined) return '';
  return String(s).replace(/[&<>"']/g, function(c){
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}
function fmt(v){
  return (v === null || v === undefined || v === '') ? '<span class="dash">&mdash;</span>' : esc(v);
}
function ruleDots(rules){
  rules = rules || {};
  return RULE_ORDER.filter(function(k){ return rules[k]; }).map(function(k){
    var st = rules[k].status || 'UNABLE';
    return '<span class="rd ' + st + '" title="' + RULE_LABEL[k] + ': ' + st + '"></span>';
  }).join('');
}
function rowHTML(rec, idx){
  var verdictCls = rec.verdict === 'VALID' ? 'ok' : 'bad';
  var verdictIcon = rec.verdict === 'VALID' ? '✓' : '✗';
  var sig = rec.signature_detected === true ? 'Yes'
    : rec.signature_detected === false ? 'No'
    : '<span class="dash">&mdash;</span>';
  return ''
    + '<td class="mono">' + (idx + 1) + '</td>'
    + '<td class="cell-id">' + esc(rec.record_id) + '</td>'
    + '<td>' + esc(String(rec.processed_time || '').replace('T', ' ').slice(0, 19)) + '</td>'
    + '<td>' + esc(rec.source_file) + '</td>'
    + '<td><span class="badge ' + verdictCls + '">' + verdictIcon + ' ' + esc(rec.verdict) + '</span></td>'
    + '<td class="cell-payee"><b>' + fmt(rec.payee) + '</b></td>'
    + '<td class="mono">' + fmt(rec.amount_numeric) + '</td>'
    + '<td class="mono">' + fmt(rec.cheque_date) + '</td>'
    + '<td>' + sig + '</td>'
    + '<td class="rule-dots">' + ruleDots(rec.rules) + '</td>';
}
function detailHTML(rec){
  var rules = rec.rules || {};
  var ruleHtml = RULE_ORDER.filter(function(k){ return rules[k]; }).map(function(k){
    var r = rules[k];
    var conf = (typeof r.confidence === 'number')
      ? ' <span class="conf-tag">(conf ' + r.confidence.toFixed(2) + ')</span>' : '';
    return '<div class="rule-line"><span class="rd ' + r.status + '"></span>'
      + '<span class="rule-id">' + RULE_LABEL[k] + '</span>'
      + '<span>' + esc(r.message || '') + conf + '</span></div>';
  }).join('') || '<div class="rule-line dash">No rule data.</div>';

  var raw = rec.raw_values || {};
  var rawHtml = Object.keys(raw).map(function(k){
    return '<div class="raw-line">' + esc(k) + ': ' + esc(raw[k]) + '</div>';
  }).join('') || '<div class="raw-line dash">No raw values captured.</div>';

  var revs = reviewsByRecord[rec.record_id] || [];
  var revHtml = revs.length ? revs.map(function(rv){
    return '<div class="review-item">' + esc(rv.note || rv.status || '')
      + '<div class="m">' + esc(rv.reviewer || '')
      + (rv.status ? ' · ' + esc(rv.status) : '')
      + (rv.timestamp ? ' · ' + esc(rv.timestamp) : '') + '</div></div>';
  }).join('') : '<div class="raw-line dash">No review history.</div>';

  return '<div class="detail-inner">'
    + '<div class="detail-group"><h4>Rule Results</h4>' + ruleHtml + '</div>'
    + '<div class="detail-group"><h4>Raw Extracted Text</h4>' + rawHtml
    + '<h4 style="margin-top:14px">Review History</h4>' + revHtml + '</div>'
    + '</div>';
}
function render(){
  var tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  if(view.length === 0){
    tbody.innerHTML = '<tr class="empty-row"><td colspan="10">No cheques processed yet.</td></tr>';
    document.getElementById('rc').textContent = 'Showing 0 of ' + RECORDS.length + ' records';
    return;
  }
  view.forEach(function(rec, i){
    var tr = document.createElement('tr');
    tr.className = 'row' + (rec.record_id === expandedId ? ' expanded' : '');
    tr.innerHTML = rowHTML(rec, i);
    tr.addEventListener('click', function(){ toggleExpand(rec.record_id); });
    tbody.appendChild(tr);

    var dtr = document.createElement('tr');
    dtr.className = 'detail-row' + (rec.record_id === expandedId ? ' open' : '');
    var dtd = document.createElement('td');
    dtd.colSpan = 10;
    dtd.innerHTML = (rec.record_id === expandedId) ? detailHTML(rec) : '';
    dtr.appendChild(dtd);
    tbody.appendChild(dtr);
  });
  document.getElementById('rc').textContent = 'Showing ' + view.length + ' of ' + RECORDS.length + ' records';
}
function toggleExpand(id){
  expandedId = (expandedId === id) ? null : id;
  render();
}
function applyFilters(){
  var q = document.getElementById('q').value.trim().toLowerCase();
  var fv = document.getElementById('f-verdict').value;
  var fy = document.getElementById('f-year').value;
  view = RECORDS.filter(function(r){
    if(fv && r.verdict !== fv) return false;
    if(fy && String(r.cheque_date || '').slice(0, 4) !== fy) return false;
    if(q){
      var hay = [r.record_id, r.payee, r.source_file, r.payee_normalized].join(' ').toLowerCase();
      if(hay.indexOf(q) === -1) return false;
    }
    return true;
  });
  sortView();
}
function sortView(){
  view.sort(function(a, b){
    var va = a[sortKey], vb = b[sortKey];
    if(va === undefined || va === null) va = '';
    if(vb === undefined || vb === null) vb = '';
    if(va < vb) return sortAsc ? -1 : 1;
    if(va > vb) return sortAsc ? 1 : -1;
    return 0;
  });
  render();
}
function setSort(key){
  if(sortKey === key){ sortAsc = !sortAsc; } else { sortKey = key; sortAsc = true; }
  sortView();
}
function populateYearFilter(){
  var years = {};
  RECORDS.forEach(function(r){
    var y = String(r.cheque_date || '').slice(0, 4);
    if(y) years[y] = true;
  });
  var sel = document.getElementById('f-year');
  Object.keys(years).sort().reverse().forEach(function(y){
    var opt = document.createElement('option');
    opt.value = y; opt.textContent = y;
    sel.appendChild(opt);
  });
}
function exportCSV(){
  var cols = ['record_id', 'processed_time', 'source_file', 'verdict', 'payee',
              'amount_numeric', 'cheque_date', 'signature_detected'];
  var lines = [cols.join(',')];
  view.forEach(function(r){
    lines.push(cols.map(function(c){
      var v = r[c];
      if(v === null || v === undefined) v = '';
      v = String(v).replace(/"/g, '""');
      return '"' + v + '"';
    }).join(','));
  });
  var blob = new Blob([lines.join('\n')], {type: 'text/csv'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'chequemate_export.csv';
  a.click();
}
function renderDonut(){
  var total = RULE_COUNTS.PASS + RULE_COUNTS.FAIL + RULE_COUNTS.UNABLE;
  var svg = document.getElementById('donut');
  var colors = {PASS: '#34d399', FAIL: '#fb7185', UNABLE: '#fbbf24'};
  var C = 2 * Math.PI * 15.9;
  var off = 0;
  if(total > 0){
    ['PASS', 'FAIL', 'UNABLE'].forEach(function(k){
      var frac = RULE_COUNTS[k] / total;
      if(frac <= 0) return;
      var el = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      el.setAttribute('cx', 21); el.setAttribute('cy', 21); el.setAttribute('r', 15.9);
      el.style.stroke = colors[k];
      el.setAttribute('stroke-dasharray', (frac * C) + ' ' + (C - frac * C));
      el.setAttribute('stroke-dashoffset', -off * C);
      svg.appendChild(el);
      off += frac;
    });
  }
  document.getElementById('donut-legend').innerHTML =
    '<span class="dk" style="background:#34d399"></span><b>' + RULE_COUNTS.PASS + '</b> Pass<br>'
    + '<span class="dk" style="background:#fb7185"></span><b>' + RULE_COUNTS.FAIL + '</b> Fail<br>'
    + '<span class="dk" style="background:#fbbf24"></span><b>' + RULE_COUNTS.UNABLE + '</b> Unable';
}

document.getElementById('q').addEventListener('input', applyFilters);
document.getElementById('f-verdict').addEventListener('change', applyFilters);
document.getElementById('f-year').addEventListener('change', applyFilters);
document.getElementById('export-btn').addEventListener('click', exportCSV);
document.querySelectorAll('th[data-key]').forEach(function(th){
  th.addEventListener('click', function(){ setSort(th.getAttribute('data-key')); });
});

populateYearFilter();
renderDonut();
applyFilters();
"""

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChequeMate Report</title>
<style>__CSS__</style>
</head>
<body>

<header class="appbar glass">
  <div class="brand"><span class="dot"></span><span class="word"><b>Cheque</b>Mate</span></div>
  <div class="subtitle">Cumulative cheque validation report</div>
  <div class="chips">
    <span class="chip">&#128197; <b>__DATE_RANGE__</b></span>
    <span class="chip">Ruleset <b>__RULESET__</b></span>
    <span class="chip">Model <b>__MODEL__</b></span>
    <span class="chip">Generated <b>__GENERATED__</b></span>
  </div>
</header>

<section class="stats">
  <div class="tile glass total"><div class="label">Total Processed</div><div class="value">__TOTAL__</div></div>
  <div class="tile glass ok"><div class="label">Valid</div><div class="value">__VALID__</div></div>
  <div class="tile glass bad"><div class="label">Invalid</div><div class="value">__INVALID__</div></div>
  <div class="tile glass donut">
    <svg class="donut-svg" viewBox="0 0 42 42" id="donut"><circle class="track" cx="21" cy="21" r="15.9"></circle></svg>
    <div class="donut-legend" id="donut-legend"></div>
  </div>
</section>

<section class="controls glass">
  <div class="search">&#128269; <input id="q" placeholder="Search payee, record ID, source file..."></div>
  <select class="ctl" id="f-verdict">
    <option value="">All verdicts</option>
    <option value="VALID">Valid</option>
    <option value="INVALID">Invalid</option>
  </select>
  <select class="ctl" id="f-year">
    <option value="">All years</option>
  </select>
  <button class="btn" id="export-btn">&#8595; Export CSV</button>
  <span class="resultcount" id="rc"></span>
</section>

<section class="table-card glass">
  <div class="table-scroll">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th data-key="record_id">Record ID <span class="si">&#8645;</span></th>
          <th data-key="processed_time">Processed <span class="si">&#8645;</span></th>
          <th data-key="source_file">Source File <span class="si">&#8645;</span></th>
          <th data-key="verdict">Verdict <span class="si">&#8645;</span></th>
          <th data-key="payee">Payee <span class="si">&#8645;</span></th>
          <th data-key="amount_numeric">Amount <span class="si">&#8645;</span></th>
          <th data-key="cheque_date">Cheque Date <span class="si">&#8645;</span></th>
          <th>Signature</th>
          <th>Rules</th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</section>

<footer>
  <span>ChequeMate Validation Report &middot; click a row to expand rule detail</span>
  <span>Model __MODEL__ &middot; Ruleset __RULESET__</span>
</footer>

<script>__JS__</script>
</body>
</html>
"""


def _parse_processed_time(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _date_range_label(records: list[dict]) -> str:
    dates = [d for d in (_parse_processed_time(r.get("processed_time"))
                         for r in records) if d]
    if not dates:
        return "—"
    lo, hi = min(dates), max(dates)
    if lo.date() == hi.date():
        return lo.strftime("%Y-%m-%d")
    return f"{lo.strftime('%Y-%m-%d')} → {hi.strftime('%Y-%m-%d')}"


def _rule_status_counts(records: list[dict]) -> dict:
    counts = {"PASS": 0, "FAIL": 0, "UNABLE": 0}
    for rec in records:
        for rule in (rec.get("rules") or {}).values():
            status = rule.get("status")
            if status in counts:
                counts[status] += 1
    return counts


def _json_for_script(value: Any) -> str:
    """JSON for embedding inside a <script> tag, safe against premature close."""
    return json.dumps(value, ensure_ascii=False, default=str).replace("</", "<\\/")


def generate_html(records: list[dict], reviews: list[dict] | None = None) -> str:
    reviews = reviews or []
    total = len(records)
    valid = sum(1 for r in records if r.get("verdict") == "VALID")
    invalid = total - valid
    rule_counts = _rule_status_counts(records)
    date_range = _date_range_label(records)
    generated = datetime.now().astimezone().isoformat(timespec="seconds")

    js = (_JS_TEMPLATE
          .replace("__RECORDS_JSON__", _json_for_script(records))
          .replace("__REVIEWS_JSON__", _json_for_script(reviews))
          .replace("__RULE_COUNTS_JSON__", _json_for_script(rule_counts)))

    return (_HTML_TEMPLATE
            .replace("__CSS__", _CSS)
            .replace("__JS__", js)
            .replace("__TOTAL__", str(total))
            .replace("__VALID__", str(valid))
            .replace("__INVALID__", str(invalid))
            .replace("__DATE_RANGE__", html.escape(date_range))
            .replace("__RULESET__", html.escape(RULE_SET_VERSION))
            .replace("__MODEL__", html.escape(MODEL_ID))
            .replace("__GENERATED__", html.escape(generated)))


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
