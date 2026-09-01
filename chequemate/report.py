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

def _field_verification_to_dict(fv) -> dict:
    return {
        "field_name": fv.field_name,
        "primary": {
            "engine": fv.primary.engine,
            "raw_value": fv.primary.raw_value,
            "normalized_value": fv.primary.normalized_value,
            "confidence": fv.primary.confidence,
            "polygon": fv.primary.polygon,
            "page_number": fv.primary.page_number,
            "model_id": fv.primary.model_id,
            "model_version": fv.primary.model_version,
        },
        "crop": {
            "status": fv.crop.status.value,
            "page_number": fv.crop.page_number,
            "pixel_bbox": list(fv.crop.pixel_bbox) if fv.crop.pixel_bbox else None,
            "padding_pixels": fv.crop.padding_pixels,
            "validation_reasons": fv.crop.validation_reasons,
            "image_reference": fv.crop.image_reference,
        },
        "secondary": {
            "engine": fv.secondary.engine,
            "status": fv.secondary.status.value,
            "raw_value": fv.secondary.raw_value,
            "normalized_value": fv.secondary.normalized_value,
            "score": fv.secondary.score,
            "score_type": fv.secondary.score_type,
            "model_id": fv.secondary.model_id,
            "model_version": fv.secondary.model_version,
            "preprocessing_variant": fv.secondary.preprocessing_variant,
            "latency_ms": fv.secondary.latency_ms,
            "error_code": fv.secondary.error_code,
        },
        "comparison": {
            "status": fv.comparison.status.value,
            "similarity": fv.comparison.similarity,
            "selected_display_value": fv.comparison.selected_display_value,
            "selection_source": fv.comparison.selection_source,
            "manual_review_required": fv.comparison.manual_review_required,
            "reason_codes": fv.comparison.reason_codes,
        },
    }


def _ocr_verifications_to_dict(ocr_verifications: dict | None) -> dict | None:
    """JSON-safe serialization of {field_name: FieldVerification}. Returns
    None (not {}) when verification never ran, so the report/JS layer can
    cleanly distinguish "TrOCR disabled for this record" from "ran and
    found nothing eligible" - see report.py's REVIEW-verdict handling for
    the same never-collapse-distinct-states principle."""
    if not ocr_verifications:
        return None
    return {name: _field_verification_to_dict(fv)
           for name, fv in ocr_verifications.items()}


def create_report_record(cheque: NormalizedCheque, validation: ValidationResult,
                          source_file: str | Path, source_hash: str,
                          processed_time: datetime | None = None,
                          rotation: dict | None = None,
                          ocr_verifications: dict | None = None) -> dict:
    """Build one privacy-safe, JSON-safe record. Does not assign record_id.

    `rotation` is imageprep.RotationDecision.as_dict() when the image went
    through that pipeline stage, else None (no imageprep call site exists
    yet - this field is forward-compatible schema, same pattern as memo's
    addition: existing callers keep working, new ones can start passing it).
    Kept both nested (full detail for the report's detail panel) and as flat
    top-level fields (rotation_direction / rotation_confident) so the
    dashboard's existing flat sort/filter machinery can reach it without
    special-casing nested objects.

    `ocr_verifications` is {field_name: FieldVerification} from
    ocr_verify.verify_cheque_fields() - None (the default) when TrOCR
    verification is disabled or was never run for this record, so every
    existing caller and every existing record already in cheques.json
    keeps working unchanged.
    """
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
        "memo": _field_value(cheque.memo),
        "rotation": rotation,
        "rotation_direction": rotation["direction"] if rotation else None,
        "rotation_confident": rotation["confident"] if rotation else None,
        "ocr_verifications": _ocr_verifications_to_dict(ocr_verifications),
        "confidence": {
            "payee": rule_conf("payee"),
            "amount_numeric": rule_conf("amount_match"),
            "amount_words": rule_conf("amount_match"),
            "date": rule_conf("date"),
            "signature": rule_conf("signature"),
            "memo": rule_conf("memo"),
        },
        "raw_values": {
            "payee": cheque.payee.raw_text,
            "amount_numeric": cheque.amount_numeric.raw_text,
            "amount_words": cheque.amount_words.raw_text,
            "date": cheque.cheque_date.raw_text,
            "memo": cheque.memo.raw_text,
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
  --warn-bg:rgba(251,191,36,0.16);
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
.stats{display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-bottom:16px}
@media (max-width:1100px){.stats{grid-template-columns:repeat(3,1fr)}}
@media (max-width:640px){.stats{grid-template-columns:repeat(2,1fr)}}
.tile{padding:16px 20px; position:relative; overflow:hidden}
.tile::before{
  content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
  background:var(--bar,var(--brand-a));
}
.tile.total{--bar:var(--brand-b)}
.tile.ok{--bar:var(--ok)}
.tile.review{--bar:var(--warn)}
.tile.bad{--bar:var(--bad)}
.tile .label{font-size:10.5px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted)}
.tile .value{font-size:30px; font-weight:700; margin-top:6px; font-variant-numeric:tabular-nums}
.tile.ok .value{color:var(--ok)}
.tile.review .value{color:var(--warn)}
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
.badge.review{background:var(--warn-bg); color:var(--warn)}
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
.plain-line{display:flex; gap:9px; align-items:flex-start; margin:8px 0; font-size:13px; line-height:1.5; color:var(--text-primary)}
.plain-line .rd{margin-top:6px}
.ready-line{font-size:13px; line-height:1.5; color:var(--ok); font-weight:600}
.sop-block{margin-bottom:16px}
.sop-block:last-child{margin-bottom:0}
.sop-block h5{font-size:12px; font-weight:700; color:var(--text-primary); margin:0 0 6px}
.sop-steps{margin:0; padding-left:18px; font-size:12px; color:var(--text-secondary); line-height:1.65}
.sop-steps li{margin:4px 0}
.detail-meta{grid-column:1 / -1; margin-top:4px; padding-top:14px; border-top:1px solid rgba(255,255,255,0.08)}
.detail-meta summary{cursor:pointer; font-size:10.5px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted)}
.detail-meta summary:hover{color:var(--text-secondary)}
.detail-meta .meta-body{display:grid; grid-template-columns:1fr 1fr 1fr; gap:22px; margin-top:12px}
.detail-meta h5{font-size:10.5px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:0 0 8px}
@media (max-width:760px){.detail-meta .meta-body{grid-template-columns:1fr}}
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

/* ── view-cheque button + full-screen lightbox ── */
.view-btn{
  border:1px solid var(--glass-border); background:var(--glass-strong);
  color:var(--text-secondary); border-radius:8px; padding:5px 10px;
  font-size:11px; font-weight:600; cursor:pointer; white-space:nowrap;
  transition:background .15s, color .15s;
}
.view-btn:hover{background:rgba(255,255,255,0.14); color:var(--text-primary)}
.lightbox{
  position:fixed; inset:0; z-index:500; display:none;
  align-items:center; justify-content:center; flex-direction:column;
  background:rgba(6,8,14,0.94); backdrop-filter:blur(6px);
}
.lightbox.open{display:flex}
.lightbox .lb-frame{
  position:relative; display:flex; align-items:center; justify-content:center;
  width:100%; height:100%; padding:60px 90px; overflow:auto;
}
.lightbox img{
  max-width:100%; max-height:100%; object-fit:contain; border-radius:10px;
  box-shadow:0 20px 60px rgba(0,0,0,0.55); background:#fff;
  transition:transform .15s ease; transform-origin:center center;
}
.lightbox .missing{
  color:var(--muted); font-size:14px; text-align:center; max-width:320px;
}
.lightbox .close{
  position:absolute; top:18px; right:22px; background:var(--glass-strong);
  border:1px solid var(--glass-border); color:var(--text-primary);
  width:38px; height:38px; border-radius:50%; font-size:18px; cursor:pointer;
  line-height:1; z-index:2;
}
.lightbox .close:hover{background:rgba(255,255,255,0.16)}
.lightbox .zoom-controls{
  position:absolute; top:18px; left:22px; z-index:2; display:flex; align-items:center;
  gap:2px; background:var(--glass-strong); border:1px solid var(--glass-border);
  border-radius:999px; padding:4px;
}
.lightbox .zoom-controls button{
  width:34px; height:34px; border-radius:50%; border:none; background:transparent;
  color:var(--text-primary); font-size:18px; cursor:pointer; line-height:1;
}
.lightbox .zoom-controls button:hover:not(:disabled){background:rgba(255,255,255,0.16)}
.lightbox .zoom-controls button:disabled{opacity:.35; cursor:default}
.lightbox .zoom-level{
  min-width:44px; text-align:center; font-size:11px; color:var(--text-secondary);
  font-variant-numeric:tabular-nums;
}
.lightbox .nav{
  position:absolute; top:50%; transform:translateY(-50%);
  background:var(--glass-strong); border:1px solid var(--glass-border);
  color:var(--text-primary); width:52px; height:52px; border-radius:50%;
  font-size:24px; cursor:pointer; line-height:1;
}
.lightbox .nav:hover{background:rgba(255,255,255,0.16)}
.lightbox .nav.prev{left:16px}
.lightbox .nav.next{right:16px}
.lightbox .nav.hidden{display:none}
.lightbox .lb-cap{
  color:var(--text-secondary); font-size:12.5px; text-align:center;
  margin-top:4px; padding:0 20px;
}
.lightbox .lb-counter{
  color:var(--muted); font-size:11px; margin-top:4px;
}
"""

_JS_TEMPLATE = r"""
const RECORDS = __RECORDS_JSON__;
const REVIEWS = __REVIEWS_JSON__;
const RULE_COUNTS = __RULE_COUNTS_JSON__;

const RULE_ORDER = ['payee', 'amount_match', 'date', 'signature', 'memo'];
const RULE_LABEL = {payee:'Payee', amount_match:'Amount', date:'Date', signature:'Signature', memo:'Memo'};

/* ── plain-language + Service Whitby SOP guidance for reception staff ──
   Static text only — driven purely by each rule's status/message already
   in the record. No LLM calls, no network, no cheque image involved. */
const RULE_GUIDANCE = {
  payee: {
    pass: "✓ Made payable to the Town of Whitby.",
    fail: function(msg){
      return msg.indexOf('misspelled') !== -1
        ? "The payee name looks misspelled — it doesn't exactly match 'Town of Whitby'."
        : "This cheque isn't made out to the Town of Whitby — it may be meant for someone else.";
    },
    steps: [
      "This fails the completeness check — the cheque must be payable to the Town of Whitby.",
      "Place the cheque in the 'Problem Cheques' folder in the filing cabinet behind the Service Desk.",
      "Look up the customer's contact info in Vailtech (Journals).",
      "If contact info exists, call the customer to correct it or send a new cheque.",
      "Create a Cheque Return Letter, photocopy the letter + cheque into the 'Cheque Returned to Sender Requesting More Information' folder, and mail the original cheque + letter to the address on the Roll number."
    ]
  },
  amount_match: {
    /* Ruleset 1.8.0/1.9.0: a PASS can come from two clean readings
       agreeing, from corroboration (a degraded reading - the written
       cents suffix needed repair - still converging with the numeral),
       or from token-match (the exact written-amount parse failed, but
       every word the numeral implies was found in the text with no
       conflicting scale word). Surface which one happened rather than
       showing the same generic line either way. */
    pass: function(msg){
      if(msg.indexOf('resolved by token-match') !== -1)
        return "✓ The written amount and the number amount agree — the written amount couldn't be read as a clean phrase, but every word the numeral implies is present in it, with no conflicting scale word (e.g. no stray \"thousand\") stated.";
      if(msg.indexOf('resolved by corroboration') !== -1)
        return "✓ The written amount and the number amount agree — the written amount's cents suffix couldn't be read as printed, but the dollar figure written out in words still matches the numeral, which is real corroboration.";
      return "✓ The written amount and the number amount agree.";
    },
    fail: function(msg){
      return msg.indexOf('no valid reading') !== -1
        ? "The written amount states a figure (e.g. a \"thousand\" or a different hundred) that the number amount rules out entirely — this looks like a genuine alteration or a completely different cheque, not just messy handwriting."
        : "The amount in words doesn't match the number amount. By law the written words are the amount that counts.";
    },
    steps: [
      "This fails the completeness check — written and numeric amounts must match. Do not enter the payment.",
      "If the customer is at the counter, have them correct the cheque and initial the change.",
      "Otherwise place it in the 'Problem Cheques' folder and contact the customer for a corrected or replacement cheque (Vailtech → Journals for contact info)."
    ]
  },
  date: {
    /* Ruleset 1.10.0: a 6-digit date read resolved via converging
       interpretations (see normalize._resolve_six_digit_date) is a
       repaired reading, not a clean one - say so, same principle as
       amount_match's corroboration/token-match notes. */
    pass: function(msg){
      if(msg.indexOf('resolved via') !== -1)
        return "✓ The cheque date is valid — the printed date box was short a digit or two, but every plausible way of reading it agreed on this date.";
      return msg.indexOf('post-dated') !== -1
        ? "✓ Date accepted — this cheque is post-dated (future-dated) and should go to the post-dated batch process."
        : "✓ The cheque date is valid and current.";
    },
    fail: function(msg){
      if(msg.indexOf('stale') !== -1) return "This cheque is stale-dated — it's more than 6 months old and the bank won't accept it.";
      if(msg.indexOf('post-dated') !== -1) return "This cheque is post-dated — it's dated in the future and can't be cashed yet.";
      return "This cheque's date did not pass validation.";
    },
    steps: function(msg){
      if(msg.indexOf('stale') !== -1) return [
        "Do not deposit a stale-dated cheque.",
        "Place it in the 'Problem Cheques' folder and contact the customer for a replacement cheque; use the Cheque Return Letter process if there's no contact info."
      ];
      if(msg.indexOf('post-dated') !== -1) return [
        "Date-stamp it on the day received.",
        "If it's a tax payment, check whether the post-dated date is an instalment due date.",
        "If it IS an instalment due date, file it in the correct due-date box in the bottom drawer of the Service Whitby filing cabinet.",
        "If it is NOT, file it in the 'Post Dated Cheques' folder, then enter it via Vailtech → Journals → Post Dated Cheques → Enter Cheques when batching."
      ];
      return ["Manually review this item before processing."];
    },
    unable: "We couldn't read the cheque date — please check it by hand.",
    unableSteps: ["Manually confirm the date before processing."]
  },
  signature: {
    pass: "✓ A signature was detected.",
    fail: "No signature was found — the cheque appears to be unsigned.",
    steps: [
      "An unsigned cheque is incomplete and cannot be processed.",
      "Place it in the 'Problem Cheques' folder and contact the customer to sign or replace it (Cheque Return Letter process if there's no contact info)."
    ],
    unable: "Could not determine whether a signature is present — this is not the same as a signature being present but hard to read; the cheque needs a manual look.",
    unableSteps: ["Visually confirm whether a signature is present before processing — do not assume one exists."]
  },
  memo: {
    /* Ruleset 1.6.0: check_memo can no longer return FAIL — a blank/absent
       memo routes to UNABLE (see rules.py's check_memo docstring for why:
       this pipeline cannot reliably tell a genuinely-blank memo line apart
       from Azure simply failing to extract one). Only pass/unable are
       reachable states now. */
    pass: "✓ A memo/note is present, showing what the payment is for.",
    unable: "We couldn't confirm a memo/note is present — this may be a data-extraction gap rather than a genuinely missing memo, so it needs a manual look at the actual cheque before treating it as missing.",
    unableSteps: [
      "Visually check the cheque itself for a memo/note before treating it as missing.",
      "If a memo genuinely isn't present, place the cheque in the 'Cheque Missing Information' folder in the filing cabinet.",
      "Scan a copy of the cheque.",
      "Email pmt-investigation@whitby.ca using the email template and attach the scan.",
      "If a department claims it, route it to them; if not, continue the missing-information exception process."
    ]
  }
};

function guidanceText(ruleId, status, message){
  var g = RULE_GUIDANCE[ruleId];
  var msg = (message || '').toLowerCase();
  if(g){
    if(status === 'PASS') return typeof g.pass === 'function' ? g.pass(msg) : g.pass;
    if(status === 'UNABLE' && g.unable) return g.unable;
    if(status === 'FAIL' && g.fail) return typeof g.fail === 'function' ? g.fail(msg) : g.fail;
  }
  // No guidance was specified for this rule/status combination (e.g. an
  // UNABLE on payee/amount_match/memo) — surface the rule's own message
  // rather than fabricating text that wasn't given.
  return message || ((RULE_LABEL[ruleId] || ruleId) + ' could not be evaluated.');
}
function guidanceSteps(ruleId, status, message){
  var g = RULE_GUIDANCE[ruleId];
  var msg = (message || '').toLowerCase();
  if(!g || status === 'PASS') return [];
  if(status === 'UNABLE' && g.unableSteps) return g.unableSteps;
  if(status === 'FAIL' && g.steps) return typeof g.steps === 'function' ? g.steps(msg) : g.steps;
  return ['Manually review this item before processing.'];
}

const reviewsByRecord = {};
REVIEWS.forEach(function(rev){
  if(!rev.record_id) return;
  (reviewsByRecord[rev.record_id] = reviewsByRecord[rev.record_id] || []).push(rev);
});

let sortKey = 'processed_time';
let sortAsc = false;
let expandedId = null;
let view = RECORDS.slice();

// Cheque images are never embedded in this file (privacy: nothing copies
// the image bytes anywhere) — this is a relative path to the source
// images folder, resolved live by the browser, and it is topology-
// dependent (see CLAUDE.md's "IMAGE_DIR relative-path trap" section):
// regenerate_report()'s default writes to reports/chequemate_report.html
// with images at repo-root cheques/ (a sibling of reports/, not of this
// file), needing '../cheques/' - but at least one real deployment
// (the \\Th240netsrv\chequemate jump-server share) puts the HTML and
// cheques/ as SIBLINGS in the same folder instead, needing 'cheques/'
// with no '../'. generate_html()'s `image_dir` parameter controls this -
// never hand-edit the value baked into a deployed copy (it is silently
// discarded the next time anyone regenerates from that same call site);
// pass the right `image_dir` for the target layout instead.
// MUST NOT start with '/': a leading slash makes the browser treat it as
// an absolute path from the site/host root, discarding whatever directory
// this report itself was deployed into (e.g. file://host/chequemate/... ->
// file://host/cheques/... , silently dropping /chequemate).
const IMAGE_DIR = '__IMAGE_DIR__';
let lightboxIndex = 0;

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
  var verdictCls = rec.verdict === 'VALID' ? 'ok' : rec.verdict === 'REVIEW' ? 'review' : 'bad';
  var verdictIcon = rec.verdict === 'VALID' ? '✓' : rec.verdict === 'REVIEW' ? '?' : '✗';
  var sig = rec.signature_detected === true ? 'Yes'
    : rec.signature_detected === false ? 'No'
    : '<span class="dash">&mdash;</span>';
  var rot = '<span class="dash">&mdash;</span>';
  if(rec.rotation){
    var isOverride = rec.rotation.source === 'operator_override';
    var rotCls = rec.rotation_confident ? 'ok' : 'bad';
    var rotIcon = rec.rotation_confident ? '' : ' ⚠';
    var rotTitle = 'fundamental ' + rec.rotation.fundamental_score.toFixed(2)
      + ' / harmonic ' + rec.rotation.harmonic_score.toFixed(2) + ' @ '
      + rec.rotation.dpi + ' dpi (' + rec.rotation.dpi_source + ')'
      + (isOverride ? ' — operator override; detector had refused' : '');
    rot = '<span class="badge ' + rotCls + '" title="' + esc(rotTitle) + '">'
      + esc(rec.rotation_direction) + rotIcon
      + (isOverride ? ' \u{1F464}' : '') + '</span>';
  }
  return ''
    + '<td class="mono">' + (idx + 1) + '</td>'
    + '<td class="cell-id">' + esc(rec.record_id) + '</td>'
    + '<td>' + esc(String(rec.processed_time || '').replace('T', ' ').slice(0, 19)) + '</td>'
    + '<td>' + esc(rec.source_file) + '</td>'
    + '<td><button class="view-btn" data-id="' + esc(rec.record_id) + '" '
      + 'onclick="event.stopPropagation(); openLightbox(this.dataset.id);">'
      + '&#128247; View Cheque</button></td>'
    + '<td><span class="badge ' + verdictCls + '">' + verdictIcon + ' ' + esc(rec.verdict) + '</span></td>'
    + '<td class="cell-payee"><b>' + fmt(rec.payee) + '</b></td>'
    + '<td class="mono">' + fmt(rec.amount_numeric) + '</td>'
    + '<td class="mono">' + fmt(rec.cheque_date) + '</td>'
    + '<td>' + fmt(rec.memo) + '</td>'
    + '<td>' + sig + '</td>'
    + '<td>' + rot + '</td>'
    + '<td class="rule-dots">' + ruleDots(rec.rules) + '</td>';
}
function detailHTML(rec){
  var rules = rec.rules || {};
  var ids = RULE_ORDER.filter(function(k){ return rules[k]; });

  var meaningHtml = ids.map(function(k){
    var r = rules[k];
    return '<div class="plain-line"><span class="rd ' + r.status + '"></span>'
      + '<span>' + esc(guidanceText(k, r.status, r.message)) + '</span></div>';
  }).join('') || '<div class="plain-line dash">No rule data.</div>';

  var failing = ids.filter(function(k){ return rules[k].status !== 'PASS'; });
  var actionHtml;
  if(!ids.length){
    actionHtml = '<div class="plain-line dash">No rule data.</div>';
  } else if(!failing.length){
    actionHtml = '<div class="ready-line">✓ Cheque is complete — date-stamp and classify it '
      + 'by payment type (tax, Town invoice, tax certificate, or miscellaneous).</div>';
  } else {
    actionHtml = failing.map(function(k){
      var r = rules[k];
      var steps = guidanceSteps(k, r.status, r.message).map(function(s){
        return '<li>' + esc(s) + '</li>';
      }).join('');
      return '<div class="sop-block"><h5>' + esc(RULE_LABEL[k] || k) + '</h5>'
        + '<ul class="sop-steps">' + steps + '</ul></div>';
    }).join('');
  }

  var raw = rec.raw_values || {};
  var rawHtml = Object.keys(raw).map(function(k){
    return '<div class="raw-line">' + esc(k) + ': ' + esc(raw[k]) + '</div>';
  }).join('') || '<div class="raw-line dash">No raw values captured.</div>';

  var rot = rec.rotation;
  var rotHtml = rot
    ? '<div class="raw-line">direction: ' + esc(rot.direction) + '</div>'
      + '<div class="raw-line">source: ' + esc(rot.source || 'detector') + '</div>'
      + '<div class="raw-line">fundamental score: ' + esc(rot.fundamental_score.toFixed(3)) + '</div>'
      + '<div class="raw-line">harmonic score: ' + esc(rot.harmonic_score.toFixed(3)) + '</div>'
      + '<div class="raw-line">MICR pitch lock: ' + esc(rot.fundamental_lag_px) + 'px</div>'
      + '<div class="raw-line">dpi: ' + esc(rot.dpi) + ' (' + esc(rot.dpi_source) + ')</div>'
      + '<div class="raw-line">confident: ' + (rot.confident
          ? 'yes' : '<b>no — orientation was marginal, verify by hand</b>') + '</div>'
      + (rot.detector_note
          ? '<div class="raw-line"><b>detector originally refused:</b> ' + esc(rot.detector_note) + '</div>'
          : '')
    : '<div class="raw-line dash">No image-preparation data on this record.</div>';

  var revs = reviewsByRecord[rec.record_id] || [];
  var revHtml = revs.length ? revs.map(function(rv){
    return '<div class="review-item">' + esc(rv.note || rv.status || '')
      + '<div class="m">' + esc(rv.reviewer || '')
      + (rv.status ? ' · ' + esc(rv.status) : '')
      + (rv.timestamp ? ' · ' + esc(rv.timestamp) : '') + '</div></div>';
  }).join('') : '<div class="raw-line dash">No review history.</div>';

  return '<div class="detail-inner">'
    + '<div class="detail-group"><h4>What this means</h4>' + meaningHtml + '</div>'
    + '<div class="detail-group"><h4>What to do next</h4>' + actionHtml + '</div>'
    + '<details class="detail-meta">'
    + '<summary>Raw extracted text &amp; review history</summary>'
    + '<div class="meta-body">'
    + '<div><h5>Raw extracted text</h5>' + rawHtml + '</div>'
    + '<div><h5>Image preparation</h5>' + rotHtml + '</div>'
    + '<div><h5>Review history</h5>' + revHtml + '</div>'
    + '</div>'
    + '</details>'
    + '</div>';
}
function render(){
  var tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  if(view.length === 0){
    tbody.innerHTML = '<tr class="empty-row"><td colspan="13">No cheques processed yet.</td></tr>';
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
    dtd.colSpan = 13;
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
  var fr = document.getElementById('f-rotation').value;
  view = RECORDS.filter(function(r){
    if(fv && r.verdict !== fv) return false;
    if(fy && String(r.cheque_date || '').slice(0, 4) !== fy) return false;
    if(fr === 'low' && r.rotation_confident !== false) return false;
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
              'amount_numeric', 'cheque_date', 'memo', 'signature_detected'];
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

/* ── minimal dependency-free .xlsx (OOXML) writer ──
   No SheetJS, no CDN, nothing leaves this file. Cells are written as
   inline strings (no shared-strings table) so values round-trip exactly
   as shown, with no numeric/locale coercion. The zip itself is store-only
   (no compression) — a hand-rolled ZIP writer with correct CRC32, local
   headers, central directory and EOCD is enough for Excel/LibreOffice to
   open it with no format-mismatch warning. */
var CRC_TABLE = (function(){
  var table = new Uint32Array(256);
  for(var n = 0; n < 256; n++){
    var c = n;
    for(var k = 0; k < 8; k++){
      c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    }
    table[n] = c >>> 0;
  }
  return table;
})();
function crc32(bytes){
  var crc = 0xFFFFFFFF;
  for(var i = 0; i < bytes.length; i++){
    crc = CRC_TABLE[(crc ^ bytes[i]) & 0xFF] ^ (crc >>> 8);
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}
function pushU16(arr, n){ arr.push(n & 0xFF, (n >>> 8) & 0xFF); }
function pushU32(arr, n){
  arr.push(n & 0xFF, (n >>> 8) & 0xFF, (n >>> 16) & 0xFF, (n >>> 24) & 0xFF);
}
function pushBytes(arr, bytes){ for(var i = 0; i < bytes.length; i++) arr.push(bytes[i]); }

function zipFiles(files){
  var encoder = new TextEncoder();
  var localChunks = [];
  var central = [];
  var offset = 0;

  files.forEach(function(f){
    var nameBytes = encoder.encode(f.name);
    var dataBytes = encoder.encode(f.data);
    var crc = crc32(dataBytes);
    var size = dataBytes.length;

    var lh = [];
    pushU32(lh, 0x04034b50);
    pushU16(lh, 20);    // version needed to extract
    pushU16(lh, 0);     // general purpose flag
    pushU16(lh, 0);     // method: 0 = store (no compression)
    pushU16(lh, 0);     // mod time
    pushU16(lh, 0x21);  // mod date: 1980-01-01
    pushU32(lh, crc);
    pushU32(lh, size);  // compressed size == uncompressed (store-only)
    pushU32(lh, size);
    pushU16(lh, nameBytes.length);
    pushU16(lh, 0);     // extra field length
    pushBytes(lh, nameBytes);
    var lhBytes = new Uint8Array(lh);
    localChunks.push(lhBytes, dataBytes);

    var ch = [];
    pushU32(ch, 0x02014b50);
    pushU16(ch, 20);    // version made by
    pushU16(ch, 20);    // version needed to extract
    pushU16(ch, 0);     // general purpose flag
    pushU16(ch, 0);     // method
    pushU16(ch, 0);     // mod time
    pushU16(ch, 0x21);  // mod date
    pushU32(ch, crc);
    pushU32(ch, size);
    pushU32(ch, size);
    pushU16(ch, nameBytes.length);
    pushU16(ch, 0);     // extra field length
    pushU16(ch, 0);     // comment length
    pushU16(ch, 0);     // disk number start
    pushU16(ch, 0);     // internal file attributes
    pushU32(ch, 0);     // external file attributes
    pushU32(ch, offset); // offset of local header
    pushBytes(ch, nameBytes);
    central.push(new Uint8Array(ch));

    offset += lhBytes.length + dataBytes.length;
  });

  var centralOffset = offset;
  var centralSize = central.reduce(function(sum, c){ return sum + c.length; }, 0);

  var eocd = [];
  pushU32(eocd, 0x06054b50);
  pushU16(eocd, 0);            // disk number
  pushU16(eocd, 0);            // disk with central directory
  pushU16(eocd, files.length); // entries on this disk
  pushU16(eocd, files.length); // entries total
  pushU32(eocd, centralSize);
  pushU32(eocd, centralOffset);
  pushU16(eocd, 0);            // comment length

  var parts = localChunks.concat(central, [new Uint8Array(eocd)]);
  var total = parts.reduce(function(sum, p){ return sum + p.length; }, 0);
  var out = new Uint8Array(total);
  var pos = 0;
  parts.forEach(function(p){ out.set(p, pos); pos += p.length; });
  return out;
}

function colLetter(n){
  var s = '';
  while(n > 0){
    var rem = (n - 1) % 26;
    s = String.fromCharCode(65 + rem) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}
function xlsxCell(ref, value, bold){
  var v = (value === null || value === undefined) ? '' : String(value);
  return '<c r="' + ref + '" t="inlineStr"' + (bold ? ' s="1"' : '') + '>'
    + '<is><t xml:space="preserve">' + esc(v) + '</t></is></c>';
}
function exportXLSX(){
  var cols = ['record_id', 'processed_time', 'source_file', 'verdict', 'payee',
              'amount_numeric', 'cheque_date', 'memo', 'signature_detected'];
  var headers = ['Record ID', 'Processed', 'Source File', 'Verdict', 'Payee',
                 'Amount', 'Cheque Date', 'Memo', 'Signature'];

  var rowsXml = '<row r="1">' + headers.map(function(h, i){
    return xlsxCell(colLetter(i + 1) + '1', h, true);
  }).join('') + '</row>';

  view.forEach(function(r, ri){
    var rowNum = ri + 2;
    rowsXml += '<row r="' + rowNum + '">' + cols.map(function(c, ci){
      return xlsxCell(colLetter(ci + 1) + rowNum, r[c], false);
    }).join('') + '</row>';
  });

  var sheetXml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    + '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    + '<dimension ref="A1:' + colLetter(cols.length) + (view.length + 1) + '"/>'
    + '<sheetData>' + rowsXml + '</sheetData>'
    + '</worksheet>';

  var contentTypes = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    + '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    + '<Default Extension="xml" ContentType="application/xml"/>'
    + '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    + '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
    + '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
    + '</Types>';

  var rootRels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    + '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
    + '</Relationships>';

  var workbookXml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    + '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
    + 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
    + '<sheets><sheet name="Cheques" sheetId="1" r:id="rId1"/></sheets>'
    + '</workbook>';

  var workbookRels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    + '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
    + '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    + '</Relationships>';

  var stylesXml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    + '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    + '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
    + '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
    + '<fills count="2"><fill><patternFill patternType="none"/></fill>'
    + '<fill><patternFill patternType="gray125"/></fill></fills>'
    + '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
    + '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
    + '<cellXfs count="2">'
    + '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
    + '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
    + '</cellXfs>'
    + '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
    + '</styleSheet>';

  var files = [
    {name: '[Content_Types].xml', data: contentTypes},
    {name: '_rels/.rels', data: rootRels},
    {name: 'xl/workbook.xml', data: workbookXml},
    {name: 'xl/_rels/workbook.xml.rels', data: workbookRels},
    {name: 'xl/styles.xml', data: stylesXml},
    {name: 'xl/worksheets/sheet1.xml', data: sheetXml}
  ];

  var zipBytes = zipFiles(files);
  var blob = new Blob([zipBytes], {
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
  });
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'chequemate_export.xlsx';
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

/* ── full-screen cheque-image viewer ──
   Navigates the CURRENT filtered/sorted view (not the full record set) —
   arrows walk through whatever the table is showing right now. The prev
   arrow hides at the first record, the next arrow hides at the last. */
function openLightbox(recordId){
  var idx = view.findIndex(function(r){ return r.record_id === recordId; });
  lightboxIndex = idx === -1 ? 0 : idx;
  renderLightbox();
  document.getElementById('lightbox').classList.add('open');
}
function closeLightbox(){
  document.getElementById('lightbox').classList.remove('open');
}
function lightboxPrev(){
  if(lightboxIndex > 0){ lightboxIndex--; renderLightbox(); }
}
function lightboxNext(){
  if(lightboxIndex < view.length - 1){ lightboxIndex++; renderLightbox(); }
}
function lightboxImgError(){
  document.getElementById('lb-img').style.display = 'none';
  document.getElementById('lb-missing').style.display = '';
}

var ZOOM_MIN = 0.5, ZOOM_MAX = 3, ZOOM_STEP = 0.25;
var lightboxZoom = 1;
function applyZoom(){
  document.getElementById('lb-img').style.transform = 'scale(' + lightboxZoom + ')';
  document.getElementById('lb-zoom-level').textContent = Math.round(lightboxZoom * 100) + '%';
  document.getElementById('lb-zoom-in').disabled = lightboxZoom >= ZOOM_MAX;
  document.getElementById('lb-zoom-out').disabled = lightboxZoom <= ZOOM_MIN;
}
function zoomIn(){
  lightboxZoom = Math.min(ZOOM_MAX, Math.round((lightboxZoom + ZOOM_STEP) * 100) / 100);
  applyZoom();
}
function zoomOut(){
  lightboxZoom = Math.max(ZOOM_MIN, Math.round((lightboxZoom - ZOOM_STEP) * 100) / 100);
  applyZoom();
}

function renderLightbox(){
  var rec = view[lightboxIndex];
  if(!rec) return;
  var img = document.getElementById('lb-img');
  img.style.display = '';
  document.getElementById('lb-missing').style.display = 'none';
  img.src = rec.source_file ? IMAGE_DIR + encodeURIComponent(rec.source_file) : '';
  document.getElementById('lb-cap').textContent =
    rec.record_id + '  ·  ' + (rec.source_file || 'no source file on record')
    + '  ·  ' + (rec.payee || '—') + '  ·  ' + rec.verdict;
  document.getElementById('lb-counter').textContent =
    (lightboxIndex + 1) + ' of ' + view.length;
  document.getElementById('lb-prev').classList.toggle('hidden', lightboxIndex === 0);
  document.getElementById('lb-next').classList.toggle('hidden', lightboxIndex === view.length - 1);
  lightboxZoom = 1;
  applyZoom();
}
document.getElementById('lb-prev').addEventListener('click', lightboxPrev);
document.getElementById('lb-next').addEventListener('click', lightboxNext);
document.getElementById('lb-close').addEventListener('click', closeLightbox);
document.getElementById('lb-zoom-in').addEventListener('click', zoomIn);
document.getElementById('lb-zoom-out').addEventListener('click', zoomOut);
document.getElementById('lightbox').addEventListener('click', function(e){
  if(e.target.id === 'lightbox') closeLightbox();
});
document.addEventListener('keydown', function(e){
  if(!document.getElementById('lightbox').classList.contains('open')) return;
  if(e.key === 'ArrowLeft') lightboxPrev();
  else if(e.key === 'ArrowRight') lightboxNext();
  else if(e.key === 'Escape') closeLightbox();
  else if(e.key === '+' || e.key === '=') zoomIn();
  else if(e.key === '-' || e.key === '_') zoomOut();
});

document.getElementById('q').addEventListener('input', applyFilters);
document.getElementById('f-verdict').addEventListener('change', applyFilters);
document.getElementById('f-year').addEventListener('change', applyFilters);
document.getElementById('f-rotation').addEventListener('change', applyFilters);
document.getElementById('export-btn').addEventListener('click', exportCSV);
document.getElementById('export-xlsx-btn').addEventListener('click', exportXLSX);
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
  <div class="tile glass review"><div class="label">Review</div><div class="value">__REVIEW__</div></div>
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
    <option value="REVIEW">Review</option>
    <option value="INVALID">Invalid</option>
  </select>
  <select class="ctl" id="f-year">
    <option value="">All years</option>
  </select>
  <select class="ctl" id="f-rotation">
    <option value="">All rotations</option>
    <option value="low">Low-confidence rotation</option>
  </select>
  <button class="btn" id="export-btn">&#8595; Export CSV</button>
  <button class="btn" id="export-xlsx-btn">&#8595; Export Excel</button>
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
          <th>Cheque</th>
          <th data-key="verdict">Verdict <span class="si">&#8645;</span></th>
          <th data-key="payee">Payee <span class="si">&#8645;</span></th>
          <th data-key="amount_numeric">Amount <span class="si">&#8645;</span></th>
          <th data-key="cheque_date">Cheque Date <span class="si">&#8645;</span></th>
          <th data-key="memo">Memo <span class="si">&#8645;</span></th>
          <th>Signature</th>
          <th data-key="rotation_confident">Rotation <span class="si">&#8645;</span></th>
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

<div class="lightbox" id="lightbox">
  <div class="lb-frame">
    <div class="zoom-controls">
      <button id="lb-zoom-out" aria-label="Zoom out">&#8722;</button>
      <span class="zoom-level" id="lb-zoom-level">100%</span>
      <button id="lb-zoom-in" aria-label="Zoom in">&#43;</button>
    </div>
    <button class="nav prev" id="lb-prev" aria-label="Previous cheque">&#8249;</button>
    <img id="lb-img" alt="Cheque image" onerror="lightboxImgError()">
    <div class="missing" id="lb-missing" style="display:none">
      No cheque image found at <code>../cheques/</code> for this record.<br>
      It may have been sourced from a saved raw response instead of an image.
    </div>
    <button class="nav next" id="lb-next" aria-label="Next cheque">&#8250;</button>
    <button class="close" id="lb-close" aria-label="Close">&#10005;</button>
  </div>
  <div class="lb-cap" id="lb-cap"></div>
  <div class="lb-counter" id="lb-counter"></div>
</div>

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


def generate_html(records: list[dict], reviews: list[dict] | None = None,
                  image_dir: str = "../cheques/") -> str:
    """`image_dir` is the browser-relative path from wherever the returned
    HTML ends up living to the folder holding the source cheque images -
    see the "IMAGE_DIR relative-path trap" note in the JS template and in
    CLAUDE.md. The default ('../cheques/') is correct ONLY for
    regenerate_report()'s own layout (reports/chequemate_report.html next
    to a repo-root cheques/ sibling of reports/) - a different deployment
    layout (e.g. the HTML and cheques/ as siblings of each other) needs a
    different value passed in here, never hand-edited into the output
    afterward.
    """
    reviews = reviews or []
    total = len(records)
    valid = sum(1 for r in records if r.get("verdict") == "VALID")
    review = sum(1 for r in records if r.get("verdict") == "REVIEW")
    invalid = sum(1 for r in records if r.get("verdict") == "INVALID")
    rule_counts = _rule_status_counts(records)
    date_range = _date_range_label(records)
    generated = datetime.now().astimezone().isoformat(timespec="seconds")

    js = (_JS_TEMPLATE
          .replace("__RECORDS_JSON__", _json_for_script(records))
          .replace("__REVIEWS_JSON__", _json_for_script(reviews))
          .replace("__RULE_COUNTS_JSON__", _json_for_script(rule_counts))
          .replace("__IMAGE_DIR__", image_dir))

    return (_HTML_TEMPLATE
            .replace("__CSS__", _CSS)
            .replace("__JS__", js)
            .replace("__TOTAL__", str(total))
            .replace("__VALID__", str(valid))
            .replace("__REVIEW__", str(review))
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
                  source_path: str | Path,
                  rotation: dict | None = None,
                  ocr_verifications: dict | None = None) -> ReportOutcome:
    """Record one validated cheque and regenerate the dashboard.

    Safe to call once per successfully-analyzed cheque; duplicates (same
    source file contents) are detected and skipped without altering the
    cheque's own validation result. `rotation` is optional imageprep
    provenance (see create_report_record) - omit until a caller wires
    imageprep.py into the live extraction path. `ocr_verifications` is
    optional TrOCR field-verification provenance (see create_report_record)
    - omit (the default) when TrOCR verification is disabled; every
    existing call site keeps working unchanged.
    """
    ensure_report_directory()
    source_hash = hash_file(source_path)
    record = create_report_record(cheque, validation, source_path, source_hash,
                                  rotation=rotation,
                                  ocr_verifications=ocr_verifications)
    outcome = append_record(record)

    if outcome.created:
        ocr_summary = None
        if ocr_verifications:
            ocr_summary = {
                name: fv.comparison.status.value
                for name, fv in ocr_verifications.items()
            }
        audit_event = {
            "timestamp": outcome.record["processed_time"],
            "event": "cheque_processed",
            "record_id": outcome.record["record_id"],
            "source_file": outcome.record["source_file"],
            "verdict": outcome.record["verdict"],
            "ruleset_version": outcome.record["ruleset_version"],
            "model": outcome.record["model"],
        }
        if ocr_summary:
            audit_event["ocr_verification_summary"] = ocr_summary
        append_audit_event(audit_event)
    else:
        append_audit_event({
            "timestamp": datetime.now().astimezone().isoformat(),
            "event": "duplicate_ingestion_skipped",
            "record_id": outcome.record["record_id"],
            "source_file": Path(source_path).name,
        })

    regenerate_report()
    # Local import: verification_report.py imports this module (for
    # load_records/load_reviews/REPORTS_DIR/_atomic_write_text) - a
    # module-level import here would be circular. This is purely additive:
    # the original chequemate_report.html above is already written and
    # never touched by this - a failure building the new report must not
    # prevent the existing, already-working report from being available.
    try:
        from . import verification_report
        verification_report.regenerate_verification_report()
    except Exception as exc:  # noqa: BLE001 - the new report is additive;
        # it must never take down the primary reporting path it rides
        # alongside.
        print(f"WARNING: verification report generation failed: "
             f"{type(exc).__name__}: {exc}")
    return outcome
