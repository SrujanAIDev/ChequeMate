import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from chequemate import validate
from chequemate.extract import to_normalized
from chequemate import report

TODAY = date(2026, 8, 17)


def azure_doc(**overrides):
    fields = {
        "PayTo": {"valueString": "The Town of Whitby", "confidence": 0.97},
        "NumberAmount": {"content": "$\n125.50", "confidence": 0.95},
        "WordAmount": {"content": "One Hundred and twenty five dollars and 50\n"
                                  "/100 DOLLARS", "confidence": 0.91},
        "CheckDate": {"content": "17 08 2026", "confidence": 0.93},
        "PayerSignatures": {"valueSignature": "signed", "confidence": 0.88},
        "Memo": {"valueString": "April Rent Payment", "confidence": 0.90},
    }
    for k, v in overrides.items():
        if v is None:
            fields.pop(k, None)
        else:
            fields[k] = v
    return {"fields": fields}


@pytest.fixture(autouse=True)
def isolated_reports(tmp_path, monkeypatch):
    """Point every report.* path constant at a scratch directory."""
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(report, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(report, "CHEQUES_JSON", reports_dir / "cheques.json")
    monkeypatch.setattr(report, "REVIEWS_JSON", reports_dir / "reviews.json")
    monkeypatch.setattr(report, "AUDIT_LOG", reports_dir / "audit.jsonl")
    monkeypatch.setattr(report, "HTML_REPORT", reports_dir / "chequemate_report.html")
    return reports_dir


@pytest.fixture
def source_file(tmp_path):
    p = tmp_path / "cheque1.png.png"
    p.write_bytes(b"fake image bytes")
    return p


# --- filesystem setup -------------------------------------------------------

def test_ensure_report_directory_creates_files(isolated_reports):
    report.ensure_report_directory()
    assert isolated_reports.is_dir()
    assert json.loads(report.CHEQUES_JSON.read_text()) == []
    assert json.loads(report.REVIEWS_JSON.read_text()) == []
    assert report.AUDIT_LOG.is_file()


def test_load_records_empty_when_missing(isolated_reports):
    assert report.load_records() == []


def test_save_and_load_records_roundtrip():
    records = [{"record_id": "CHQ-20260818-0001", "verdict": "VALID"}]
    report.save_records(records)
    assert report.load_records() == records
    # pretty-printed for readability
    assert report.CHEQUES_JSON.read_text().count("\n") > 1


def test_empty_file_treated_as_empty_array():
    report.ensure_report_directory()
    report.CHEQUES_JSON.write_text("")
    assert report.load_records() == []


def test_corrupted_json_raises_and_preserves_file():
    report.ensure_report_directory()
    report.CHEQUES_JSON.write_text("{not valid json")
    with pytest.raises(report.ReportError):
        report.load_records()
    assert report.CHEQUES_JSON.read_text() == "{not valid json"


# --- reviews.json: same array-of-events schema as cheques.json --------------
#
# A review is an event (status change, note) over time, not a single mutable
# row — the same record_id can appear in more than one entry to preserve
# history. That only works as a top-level array, so reviews.json shares the
# exact same schema, loader, and corruption contract as cheques.json rather
# than an object keyed by record_id.

def test_reviews_json_initializes_as_array(isolated_reports):
    report.ensure_report_directory()
    assert json.loads(report.REVIEWS_JSON.read_text()) == []


def test_save_and_load_reviews_roundtrip():
    reviews = [
        {"record_id": "CHQ-20260818-0001", "timestamp": "2026-08-18T10:00:00",
         "reviewer": "srujan", "status": "Under Review", "note": "checking payee"},
        {"record_id": "CHQ-20260818-0001", "timestamp": "2026-08-19T09:00:00",
         "reviewer": "srujan", "status": "Confirmed Valid", "note": None},
    ]
    report.save_reviews(reviews)
    assert report.load_reviews() == reviews
    # two entries share one record_id — only an array preserves that history
    assert len([r for r in report.load_reviews()
               if r["record_id"] == "CHQ-20260818-0001"]) == 2


def test_reviews_json_rejects_object_schema():
    """An object keyed by record_id can't hold review history; only a list is valid."""
    report.ensure_report_directory()
    report.REVIEWS_JSON.write_text(
        '{"CHQ-20260818-0001": {"status": "Under Review"}}')
    with pytest.raises(report.ReportError):
        report.load_reviews()
    # corrupt-relative-to-schema file is preserved, not silently overwritten
    assert "CHQ-20260818-0001" in report.REVIEWS_JSON.read_text()


def test_reviews_json_empty_file_treated_as_empty_array():
    report.ensure_report_directory()
    report.REVIEWS_JSON.write_text("")
    assert report.load_reviews() == []


# --- record creation ---------------------------------------------------------

def test_create_report_record_shape(source_file):
    cheque = to_normalized(azure_doc())
    result = validate(cheque, today=TODAY)
    rec = report.create_report_record(
        cheque, result, source_file, report.hash_file(source_file))

    assert rec["schema_version"] == "1.0"
    assert rec["record_id"] is None
    assert rec["source_file"] == "cheque1.png.png"
    assert rec["model"] == "prebuilt-check.us"
    assert rec["verdict"] == "VALID"
    assert rec["amount_numeric"] == "125.50"          # Decimal -> str
    assert rec["amount_words"] == "125.50"
    assert rec["cheque_date"] == "2026-08-17"          # date -> ISO string
    assert rec["signature_detected"] is True
    assert rec["confidence"]["amount_numeric"] is None  # amount_match carries no confidence
    assert rec["rules"]["payee"]["status"] == "PASS"
    assert rec["raw_values"]["amount_numeric"] == "$ 125.50"
    json.dumps(rec)  # must be fully JSON-serialisable


def test_missing_field_is_null_not_fabricated(source_file):
    doc = azure_doc(PayerSignatures=None)
    cheque = to_normalized(doc)
    result = validate(cheque, today=TODAY)
    rec = report.create_report_record(
        cheque, result, source_file, report.hash_file(source_file))
    assert rec["signature_detected"] is None
    assert rec["rules"]["signature"]["status"] == "UNABLE"


# --- record IDs & append -----------------------------------------------------

def test_append_record_assigns_sequential_ids(source_file, tmp_path):
    cheque = to_normalized(azure_doc())
    result = validate(cheque, today=TODAY)

    rec1 = report.create_report_record(cheque, result, source_file, "hash-one")
    out1 = report.append_record(rec1, today=TODAY)
    assert out1.created
    assert out1.record["record_id"] == "CHQ-20260817-0001"

    rec2 = report.create_report_record(cheque, result, source_file, "hash-two")
    out2 = report.append_record(rec2, today=TODAY)
    assert out2.created
    assert out2.record["record_id"] == "CHQ-20260817-0002"

    assert len(report.load_records()) == 2


def test_append_record_dedupes_by_hash(source_file):
    cheque = to_normalized(azure_doc())
    result = validate(cheque, today=TODAY)
    rec = report.create_report_record(cheque, result, source_file, "same-hash")

    out1 = report.append_record(rec, today=TODAY)
    out2 = report.append_record(rec, today=TODAY)

    assert out1.created is True
    assert out2.created is False
    assert out2.record["record_id"] == out1.record["record_id"]
    assert len(report.load_records()) == 1


def test_next_record_id_skips_existing_even_out_of_order():
    report.save_records([
        {"record_id": "CHQ-20260817-0001"},
        {"record_id": "CHQ-20260817-0003"},
    ])
    next_id = report._next_record_id(report.load_records(), TODAY)
    assert next_id == "CHQ-20260817-0004"


# --- hashing -----------------------------------------------------------------

def test_hash_file_deterministic(tmp_path):
    p = tmp_path / "a.json"
    p.write_text('{"a": 1}')
    assert report.hash_file(p) == report.hash_file(p)

    q = tmp_path / "b.json"
    q.write_text('{"a": 2}')
    assert report.hash_file(p) != report.hash_file(q)


# --- audit trail ---------------------------------------------------------

def test_append_audit_event_is_one_line_per_event():
    report.append_audit_event({"event": "cheque_processed", "record_id": "X"})
    report.append_audit_event({"event": "cheque_processed", "record_id": "Y"})
    lines = report.AUDIT_LOG.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["record_id"] == "X"
    assert json.loads(lines[1])["record_id"] == "Y"


# --- HTML dashboard -----------------------------------------------------

def test_generate_html_reports_totals():
    records = [
        {"record_id": "CHQ-1", "verdict": "VALID", "processed_time": "t1",
         "rules": {}},
        {"record_id": "CHQ-2", "verdict": "INVALID", "processed_time": "t2",
         "rules": {}},
    ]
    html_out = report.generate_html(records)
    assert "Total Processed" in html_out
    assert ">2<" in html_out  # total
    assert "CHQ-1" in html_out and "CHQ-2" in html_out


def test_generate_html_renders_full_review_history_not_just_latest():
    """Two review events for one record_id must both appear — proves the
    array schema's history is actually surfaced, not collapsed to one row."""
    records = [{"record_id": "CHQ-1", "verdict": "VALID", "processed_time": "t1",
               "rules": {}}]
    reviews = [
        {"record_id": "CHQ-1", "note": "first pass looks ok"},
        {"record_id": "CHQ-1", "note": "second pass, confirmed"},
    ]
    html_out = report.generate_html(records, reviews)
    assert "first pass looks ok" in html_out
    assert "second pass, confirmed" in html_out


def test_generate_html_empty_state():
    html_out = report.generate_html([])
    assert "No cheques processed yet" in html_out


# --- end-to-end via update_report --------------------------------------

def test_update_report_end_to_end_cumulative(tmp_path):
    img1 = tmp_path / "cheque1.png.png"
    img1.write_bytes(b"cheque-one-bytes")
    img2 = tmp_path / "cheque2.png.png"
    img2.write_bytes(b"cheque-two-bytes")

    cheque1 = to_normalized(azure_doc())
    result1 = validate(cheque1, today=TODAY)
    out1 = report.update_report(cheque1, result1, img1)
    assert out1.created

    doc2 = azure_doc(PayerSignatures=None,
                     CheckDate={"content": "15062020\nDDM MY YYY"})
    cheque2 = to_normalized(doc2)
    result2 = validate(cheque2, today=TODAY)
    out2 = report.update_report(cheque2, result2, img2)
    assert out2.created

    records = report.load_records()
    assert len(records) == 2
    assert {r["record_id"] for r in records} == {
        out1.record["record_id"], out2.record["record_id"]}

    html_text = report.HTML_REPORT.read_text(encoding="utf-8")
    assert ">2<" in html_text  # total processed
    assert ">1<" in html_text  # valid count present somewhere

    # Re-processing the same file (e.g. re-run of the CLI) must not duplicate.
    out1_again = report.update_report(cheque1, result1, img1)
    assert out1_again.created is False
    assert len(report.load_records()) == 2

    audit_lines = report.AUDIT_LOG.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(l)["event"] for l in audit_lines]
    assert events.count("cheque_processed") == 2
    assert events.count("duplicate_ingestion_skipped") == 1
