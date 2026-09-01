"""End-to-end integration test: a synthetic cheque image + a mocked
Document Intelligence response (with field polygons) + a fake TrOCR
adapter, run through run_batch.process_one() exactly as production would,
verifying the persisted record schema, the audit trail, and that the new
verification report gets created - all through report.py's real file I/O,
isolated to a tmp_path (same isolation pattern as tests/test_report.py).

No real cheque data, no Azure call, no torch/transformers model load.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chequemate import ocr_verify, report  # noqa: E402
from chequemate.trocr_adapter import TrOCRRawResult  # noqa: E402
from chequemate.models import TrOCRStatus  # noqa: E402
import run_batch  # noqa: E402


class _FakeClient:
    """Records every image it was asked to read, for the "never sends the
    full cheque" regression check, and returns a fixed reading per field."""

    def __init__(self, text_by_field: dict):
        self.text_by_field = text_by_field
        self.seen_image_sizes: list[tuple[int, int]] = []

    def generate(self, image, *, field_name):
        self.seen_image_sizes.append(image.size)
        text = self.text_by_field.get(field_name)
        return TrOCRRawResult(
            status=TrOCRStatus.COMPLETED if text else TrOCRStatus.UNDETERMINED,
            raw_text=text, sequence_score=None, score_type="unavailable",
            error_code=None, latency_ms=2.0, model_id="fake-trocr",
            model_version="test", device="cpu")


IMAGE_W, IMAGE_H = 1200, 500


def _synthetic_raw_response():
    return {
        "documents": [{
            "fields": {
                "PayTo": {"valueString": "Town of Whitby", "confidence": 0.55,
                         "boundingRegions": [{"pageNumber": 1,
                                             "polygon": [80, 200, 600, 200,
                                                       600, 260, 80, 260]}]},
                "NumberAmount": {"content": "$200.00", "confidence": 0.9},
                "WordAmount": {"content": "Two hundred dollars and 00/100",
                              "confidence": 0.55,
                              "boundingRegions": [{"pageNumber": 1,
                                                  "polygon": [80, 300, 900, 300,
                                                            900, 350, 80, 350]}]},
                "CheckDate": {"content": "17 08 2026", "confidence": 0.9},
                "PayerSignatures": {"valueSignature": "signed", "confidence": 0.8},
                "Memo": {"valueString": "Roll number 998877", "confidence": 0.85,
                        "boundingRegions": [{"pageNumber": 1,
                                            "polygon": [80, 400, 400, 400,
                                                      400, 440, 80, 440]}]},
            },
        }],
        "pages": [{"pageNumber": 1, "width": IMAGE_W, "height": IMAGE_H, "unit": "pixel"}],
    }


@pytest.fixture(autouse=True)
def isolated_reports(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(report, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(report, "CHEQUES_JSON", reports_dir / "cheques.json")
    monkeypatch.setattr(report, "REVIEWS_JSON", reports_dir / "reviews.json")
    monkeypatch.setattr(report, "AUDIT_LOG", reports_dir / "audit.jsonl")
    monkeypatch.setattr(report, "HTML_REPORT", reports_dir / "chequemate_report.html")
    from chequemate import verification_report
    monkeypatch.setattr(verification_report, "HTML_VERIFICATION_REPORT",
                        reports_dir / "chequemate_verification_report.html")
    return reports_dir


@pytest.fixture
def replay_json_and_image(tmp_path):
    """A .json replay file (so process_one never calls Azure) plus the
    matching synthetic image file process_one opens for cropping."""
    raw = _synthetic_raw_response()
    json_path = tmp_path / "synthetic_cheque.json"
    json_path.write_text(json.dumps(raw), encoding="utf-8")
    image_path = tmp_path / "synthetic_cheque.png"
    Image.new("RGB", (IMAGE_W, IMAGE_H), (255, 255, 255)).save(image_path)
    return json_path, image_path


def test_end_to_end_replay_with_trocr_produces_full_schema(
        replay_json_and_image, monkeypatch, isolated_reports):
    json_path, image_path = replay_json_and_image
    fake = _FakeClient(text_by_field={
        "payee": "Town of Whitby", "amount_words": "Two hundred dollars and 00/100",
        "memo": "Roll number 998877"})
    monkeypatch.setattr(ocr_verify, "get_or_create_client", lambda cfg: fake)

    cfg = run_batch.Config(expected_payee="Town of Whitby")
    trocr_cfg = ocr_verify.TrOCRVerificationConfig(enabled=True)

    # process_one reads the replay JSON for `raw` (no image needed for
    # that), but TrOCR verification needs the actual image file - since
    # the replay path here is a bare .json, verification only triggers for
    # genuine image files. Exercise the real path used in production: an
    # image file, with Azure replaced by monkeypatching analyze_raw.
    monkeypatch.setattr(run_batch, "analyze_raw", lambda *a, **kw: _synthetic_raw_response())

    outcome = run_batch.process_one(image_path, "https://fake", "fake-key", cfg, "DMY",
                                    save_raw_dir=None, trocr_cfg=trocr_cfg)

    assert outcome.created is True
    record = outcome.record

    # --- schema: ocr_verifications present with all three eligible fields ---
    assert "ocr_verifications" in record
    ocr = record["ocr_verifications"]
    assert set(ocr) == {"payee", "amount_words", "memo"}
    for field_name in ("payee", "amount_words", "memo"):
        fv = ocr[field_name]
        assert set(fv) == {"field_name", "primary", "crop", "secondary", "comparison"}
        assert fv["comparison"]["status"] in (
            "AGREE_EXACT", "AGREE_NORMALIZED", "AGREE_ALLOWLIST", "DISAGREE",
            "DI_ONLY", "TROCR_ONLY", "CROP_REJECTED", "TROCR_NOT_RUN",
            "TROCR_FAILED", "BOTH_UNDETERMINED")

    # --- deterministic rules still ran, untouched ---
    assert set(record["rules"]) == {"payee", "amount_match", "signature", "date", "memo"}
    assert record["verdict"] in ("VALID", "REVIEW", "INVALID")

    # --- audit payload ---
    audit_lines = report.AUDIT_LOG.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in audit_lines]
    processed_events = [e for e in events if e["event"] == "cheque_processed"]
    assert len(processed_events) == 1
    assert "ocr_verification_summary" in processed_events[0]

    # --- report creation (both versions) ---
    assert report.HTML_REPORT.is_file()
    from chequemate import verification_report
    assert verification_report.HTML_VERIFICATION_REPORT.is_file()
    verification_html = verification_report.HTML_VERIFICATION_REPORT.read_text(encoding="utf-8")
    assert record["record_id"] in verification_html

    # --- never sends the full cheque to TrOCR ---
    assert len(fake.seen_image_sizes) == 3
    for size in fake.seen_image_sizes:
        assert size != (IMAGE_W, IMAGE_H)


def test_trocr_disabled_produces_identical_schema_shape_minus_verification(
        replay_json_and_image, monkeypatch, isolated_reports):
    json_path, image_path = replay_json_and_image
    monkeypatch.setattr(run_batch, "analyze_raw", lambda *a, **kw: _synthetic_raw_response())
    cfg = run_batch.Config(expected_payee="Town of Whitby")

    outcome = run_batch.process_one(image_path, "https://fake", "fake-key", cfg, "DMY",
                                    save_raw_dir=None, trocr_cfg=None)

    assert outcome.record["ocr_verifications"] is None
    assert report.HTML_REPORT.is_file()
    # the verification report is still (harmlessly) regenerated, showing an
    # honest "not run" state - it must not error just because no record has
    # OCR data yet.
    from chequemate import verification_report
    assert verification_report.HTML_VERIFICATION_REPORT.is_file()


def test_trocr_model_failure_does_not_crash_batch_processing(
        replay_json_and_image, monkeypatch, isolated_reports):
    json_path, image_path = replay_json_and_image
    monkeypatch.setattr(run_batch, "analyze_raw", lambda *a, **kw: _synthetic_raw_response())

    class _RaisingClient:
        def generate(self, image, *, field_name):
            raise RuntimeError("simulated model crash")

    monkeypatch.setattr(ocr_verify, "get_or_create_client", lambda cfg: _RaisingClient())
    cfg = run_batch.Config(expected_payee="Town of Whitby")
    trocr_cfg = ocr_verify.TrOCRVerificationConfig(enabled=True)

    outcome = run_batch.process_one(image_path, "https://fake", "fake-key", cfg, "DMY",
                                    save_raw_dir=None, trocr_cfg=trocr_cfg)

    assert outcome.created is True
    for fv in outcome.record["ocr_verifications"].values():
        assert fv["secondary"]["status"] == "failed"
    # the deterministic verdict is unaffected by the TrOCR crash
    assert outcome.record["verdict"] in ("VALID", "REVIEW", "INVALID")
