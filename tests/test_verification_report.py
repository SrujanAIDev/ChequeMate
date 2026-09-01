"""Tests for chequemate.verification_report - the new, additive "ChequeMate
AI Review and OCR Verification Report". All fixtures are synthetic dicts
shaped like report.py's own record schema; no real cheque data or file I/O
against reports/ is used (render functions are called directly with
in-memory record lists).
"""

from __future__ import annotations

from chequemate.verification_report import REVIEWER_NOTICE, generate_verification_html


def _record(**overrides):
    base = {
        "record_id": "CHQ-TEST-0001",
        "source_file": "test.png",
        "processed_time": "2026-08-27T00:00:00",
        "verdict": "VALID",
        "ruleset_version": "1.5.0",
        "rules": {
            "payee": {"status": "PASS", "message": "matched"},
            "amount_match": {"status": "PASS", "message": "figures match"},
        },
        "ocr_verifications": None,
    }
    base.update(overrides)
    return base


def _agree_field_verification(field_name="payee", value="Town of Whitby"):
    return {
        "field_name": field_name,
        "primary": {"raw_value": value, "confidence": 0.9,
                   "model_id": "prebuilt-check.us"},
        "crop": {"status": "accepted", "pixel_bbox": [10, 20, 200, 60],
                "validation_reasons": ["polygon converted and padded within bounds"]},
        "secondary": {"raw_value": value, "status": "completed", "score": None,
                     "score_type": "unavailable",
                     "model_id": "microsoft/trocr-base-handwritten",
                     "model_version": "rev1", "preprocessing_variant": "rgb_normalized"},
        "comparison": {"status": "AGREE_EXACT", "similarity": 1.0,
                      "selected_display_value": value,
                      "selection_source": "document_intelligence",
                      "manual_review_required": False, "reason_codes": []},
    }


def _disagree_field_verification():
    return {
        "field_name": "payee",
        "primary": {"raw_value": "Town of Whitby", "confidence": 0.6,
                   "model_id": "prebuilt-check.us"},
        "crop": {"status": "accepted", "pixel_bbox": [10, 20, 200, 60],
                "validation_reasons": []},
        "secondary": {"raw_value": "Something Else", "status": "completed",
                     "score": 0.71, "score_type": "raw_generation_evidence",
                     "model_id": "microsoft/trocr-base-handwritten",
                     "model_version": "rev1", "preprocessing_variant": "rgb_normalized"},
        "comparison": {"status": "DISAGREE", "similarity": 0.2,
                      "selected_display_value": "Town of Whitby",
                      "selection_source": "document_intelligence",
                      "manual_review_required": True,
                      "reason_codes": ["OCR_DISAGREEMENT"]},
    }


# ---------------------------------------------------------------------------
# basic rendering
# ---------------------------------------------------------------------------

def test_renders_reviewer_notice_verbatim():
    html_out = generate_verification_html([_record()], [])
    assert REVIEWER_NOTICE in html_out


def test_renders_with_no_records_at_all():
    html_out = generate_verification_html([], [])
    assert "No cheque records yet" in html_out
    assert "<html" in html_out


def test_renders_with_trocr_disabled_no_ocr_verifications_key():
    record = _record(ocr_verifications=None)
    html_out = generate_verification_html([record], [])
    assert "TrOCR verification was not run for this record" in html_out


def test_existing_rules_are_shown_alongside_ocr_data():
    record = _record(ocr_verifications={"payee": _agree_field_verification()})
    html_out = generate_verification_html([record], [])
    assert "payee" in html_out
    assert "matched" in html_out  # the rule's own message, untouched


# ---------------------------------------------------------------------------
# disagreement visibility
# ---------------------------------------------------------------------------

def test_disagreement_shows_both_values_and_warning():
    record = _record(verdict="REVIEW",
                     ocr_verifications={"payee": _disagree_field_verification()})
    html_out = generate_verification_html([record], [])
    assert "Town of Whitby" in html_out
    assert "Something Else" in html_out
    assert "Manual verification required" in html_out
    assert "Neither result has been automatically substituted" in html_out


def test_disagreement_counted_in_summary_tile():
    record = _record(ocr_verifications={"payee": _disagree_field_verification()})
    html_out = generate_verification_html([record], [])
    assert "OCR field disagreements" in html_out


# ---------------------------------------------------------------------------
# confidence discipline
# ---------------------------------------------------------------------------

def test_missing_score_labeled_not_calibrated():
    record = _record(ocr_verifications={"payee": _agree_field_verification()})
    html_out = generate_verification_html([record], [])
    assert "Not calibrated" in html_out


def test_raw_generation_evidence_never_called_confidence_percentage():
    record = _record(ocr_verifications={"payee": _disagree_field_verification()})
    html_out = generate_verification_html([record], [])
    assert "0.710" in html_out
    assert "raw generation evidence" in html_out
    assert "71%" not in html_out
    assert "71.0% confidence" not in html_out


# ---------------------------------------------------------------------------
# XSS / injection safety
# ---------------------------------------------------------------------------

def test_ocr_text_html_is_escaped():
    malicious = _agree_field_verification(value='<img src=x onerror=alert(1)>')
    malicious["secondary"]["raw_value"] = '<script>alert(1)</script>'
    record = _record(ocr_verifications={"payee": malicious})
    html_out = generate_verification_html([record], [])
    assert "<script>alert(1)</script>" not in html_out
    assert "<img src=x onerror=alert(1)>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_memo_text_with_template_syntax_is_escaped_not_executed():
    # No templating engine (Jinja2/moustache/etc.) is used anywhere in this
    # module - plain Python string formatting only - so "{{7*7}}" has
    # nothing to be evaluated by; this locks in that OCR text always
    # survives verbatim (HTML-escaped) rather than being treated as markup.
    fv = _agree_field_verification(field_name="memo", value="{{7*7}}")
    record = _record(ocr_verifications={"memo": fv})
    html_out = generate_verification_html([record], [])
    assert "{{7*7}}" in html_out


# ---------------------------------------------------------------------------
# masking - no MICR/account data ever reaches this report (none is ever
# persisted to cheques.json in the first place - report.py's privacy
# boundary - so this asserts the report doesn't introduce a new leak path)
# ---------------------------------------------------------------------------

def test_no_micr_or_account_field_rendered():
    import re
    record = _record(ocr_verifications={"payee": _agree_field_verification()})
    html_out = generate_verification_html([record], [])
    # word-boundary match: "MICROSOFT" (the TrOCR model vendor name, which
    # legitimately appears in the model-id line) must not false-positive
    # this check for an actual MICR-line/account-number leak.
    assert re.search(r"\bMICR\b", html_out.upper()) is None
    assert "routing_number" not in html_out
    assert "account_number" not in html_out


# ---------------------------------------------------------------------------
# resilience: crop rejected / model failure must still render cleanly
# ---------------------------------------------------------------------------

def test_renders_with_crop_rejected_field():
    fv = {
        "field_name": "amount_words",
        "primary": {"raw_value": "Two Hundred", "confidence": 0.5,
                   "model_id": "prebuilt-check.us"},
        "crop": {"status": "rejected", "pixel_bbox": None,
                "validation_reasons": ["crop area 50px is below the minimum 400px"]},
        "secondary": {"raw_value": None, "status": "not_run", "score": None,
                     "score_type": "unavailable", "model_id": None,
                     "model_version": None, "preprocessing_variant": None},
        "comparison": {"status": "CROP_REJECTED", "similarity": None,
                      "selected_display_value": "Two Hundred",
                      "selection_source": "document_intelligence",
                      "manual_review_required": False, "reason_codes": ["TROCR_UNAVAILABLE"]},
    }
    record = _record(ocr_verifications={"amount_words": fv})
    html_out = generate_verification_html([record], [])
    assert "Secondary verification unavailable" in html_out
    assert "Two Hundred" in html_out


def test_renders_with_trocr_model_failure():
    fv = {
        "field_name": "payee",
        "primary": {"raw_value": "Town of Whitby", "confidence": 0.9,
                   "model_id": "prebuilt-check.us"},
        "crop": {"status": "accepted", "pixel_bbox": [1, 2, 3, 4],
                "validation_reasons": []},
        "secondary": {"raw_value": None, "status": "failed", "score": None,
                     "score_type": "unavailable", "model_id": "microsoft/trocr-base-handwritten",
                     "model_version": None, "preprocessing_variant": "rgb_normalized",
                     "error_code": "model_load_failed:OSError"},
        "comparison": {"status": "TROCR_FAILED", "similarity": None,
                      "selected_display_value": "Town of Whitby",
                      "selection_source": "document_intelligence",
                      "manual_review_required": False, "reason_codes": ["TROCR_UNAVAILABLE"]},
    }
    record = _record(ocr_verifications={"payee": fv})
    html_out = generate_verification_html([record], [])
    assert "Town of Whitby" in html_out
    assert "Secondary verification unavailable" in html_out


def test_missing_optional_keys_do_not_crash_rendering():
    # a deliberately sparse record - only what's strictly required present.
    sparse = {
        "record_id": "CHQ-TEST-0002", "source_file": "x.png",
        "processed_time": "2026-08-27T00:00:00", "verdict": "INVALID",
        "ruleset_version": "1.5.0", "rules": {}, "ocr_verifications": {},
    }
    html_out = generate_verification_html([sparse], [])
    assert "CHQ-TEST-0002" in html_out
