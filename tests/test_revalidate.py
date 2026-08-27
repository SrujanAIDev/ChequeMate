"""Tests for scripts/revalidate.py's layering behaviour (Phase 4, Correction
A): a standing apply_visual_verification.py override must survive a
re-derivation from raw_values, not be discarded by it."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from chequemate import Config, validate  # noqa: E402
from chequemate.models import RuleStatus, Verdict  # noqa: E402
from chequemate.extract import to_normalized  # noqa: E402
import revalidate  # noqa: E402


def azure_doc(**overrides):
    fields = {
        "PayTo": {"valueString": "Some Garbled Text", "confidence": 0.5},
        "NumberAmount": {"content": "$\n125.50", "confidence": 0.95},
        "WordAmount": {"content": "One Hundred and twenty five dollars and 50"
                                  "\n/100 DOLLARS", "confidence": 0.91},
        "CheckDate": {"content": "17 08 2026", "confidence": 0.93},
        "Memo": {"valueString": "April Rent Payment", "confidence": 0.90},
    }
    for k, v in overrides.items():
        if v is None:
            fields.pop(k, None)
        else:
            fields[k] = v
    return {"fields": fields}


def test_layering_reapplies_signature_override_after_rederivation(monkeypatch):
    """A record with no PayerSignatures field at all (raw automated
    UNABLE/FAIL depending on config) must come back with signature PASS
    after revalidate.py re-derives it, because a standing visual-verification
    confirmation exists for it - the whole point of layering over skipping."""
    record_id = "CHQ-TEST-0001"
    monkeypatch.setattr(revalidate, "SIGNATURE_CONFIRMATIONS", {record_id})

    cheque = to_normalized(azure_doc())
    result = validate(cheque, Config())
    # raw automated result: no signature field at all -> UNABLE, not PASS
    sig_before = next(r for r in result.rules if r.rule_id == "signature")
    assert sig_before.status is RuleStatus.UNABLE

    overlay_applied = revalidate._apply_standing_overrides(record_id, cheque, result)

    assert overlay_applied is True
    sig_after = next(r for r in result.rules if r.rule_id == "signature")
    assert sig_after.status is RuleStatus.PASS
    assert cheque.signature.value is True


def test_layering_reapplies_payee_override_after_rederivation(monkeypatch):
    """A record whose raw OCR payee text would legitimately FAIL must come
    back PASS (with the override's explanatory message) once the standing
    payee confirmation is layered back on."""
    record_id = "CHQ-TEST-0002"
    monkeypatch.setattr(revalidate, "PAYEE_CONFIRMATIONS",
                        {record_id: ("Town of Whitby", "test override reason")})

    cheque = to_normalized(azure_doc(PayTo={"valueString": "Some Garbled Text"}))
    result = validate(cheque, Config())
    payee_before = next(r for r in result.rules if r.rule_id == "payee")
    assert payee_before.status is RuleStatus.FAIL
    assert result.verdict is Verdict.INVALID

    overlay_applied = revalidate._apply_standing_overrides(record_id, cheque, result)

    assert overlay_applied is True
    payee_after = next(r for r in result.rules if r.rule_id == "payee")
    assert payee_after.status is RuleStatus.PASS
    assert "test override reason" in payee_after.evidence
    # verdict must be recomputed AFTER the overlay, not left at the pre-overlay
    # value - here that's REVIEW, not VALID, because azure_doc() has no
    # PayerSignatures field at all, so signature is (correctly) still UNABLE.
    assert result.verdict is Verdict.REVIEW


def test_no_overlay_for_unlisted_record(monkeypatch):
    monkeypatch.setattr(revalidate, "SIGNATURE_CONFIRMATIONS", set())
    monkeypatch.setattr(revalidate, "PAYEE_CONFIRMATIONS", {})

    cheque = to_normalized(azure_doc())
    result = validate(cheque, Config())
    applied = revalidate._apply_standing_overrides("CHQ-NOT-LISTED", cheque, result)
    assert applied is False
