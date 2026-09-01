"""Tests for chequemate.extract's repeat-extraction reconciliation
(analyze_multi / to_normalized_multi / reconcile_cheques) - built after a
real experiment confirmed Azure Document Intelligence is not perfectly
deterministic run-to-run on unchanged input. No real cheque data or Azure
calls - all documents are synthetic, shaped like azure_doc() elsewhere in
this test suite.
"""

from __future__ import annotations

from decimal import Decimal

from chequemate.extract import (
    _reconcile_amount_words, _reconcile_generic_field, reconcile_cheques,
    to_normalized, to_normalized_multi,
)
from chequemate.models import ParseStatus


def azure_doc(**overrides):
    fields = {
        "PayTo": {"valueString": "Town of Whitby", "confidence": 0.97},
        "NumberAmount": {"content": "$\n125.50", "confidence": 0.95},
        "WordAmount": {"content": "One Hundred and twenty five dollars and 50"
                                  "\n/100 DOLLARS", "confidence": 0.91},
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


# ---------------------------------------------------------------------------
# _reconcile_generic_field
# ---------------------------------------------------------------------------

def test_unanimous_agreement_keeps_the_field_untouched():
    cheques = [to_normalized(azure_doc()) for _ in range(3)]
    fields = [c.payee for c in cheques]
    result = _reconcile_generic_field("payee", fields)
    assert result.parse_status is ParseStatus.OK
    assert result.value == fields[0].value
    assert result.alternate_readings is None


def test_disagreement_produces_ambiguous_with_no_value():
    cheques = [
        to_normalized(azure_doc(PayTo={"valueString": "Town of Whitby"})),
        to_normalized(azure_doc(PayTo={"valueString": "Town of Whitty"})),
        to_normalized(azure_doc(PayTo={"valueString": "Town of Whitby"})),
    ]
    fields = [c.payee for c in cheques]
    result = _reconcile_generic_field("payee", fields)
    assert result.parse_status is ParseStatus.AMBIGUOUS
    assert result.value is None
    assert "disagreed" in result.note.lower()


def test_all_readings_preserved_on_disagreement():
    cheques = [
        to_normalized(azure_doc(PayTo={"valueString": "Town of Whitby"})),
        to_normalized(azure_doc(PayTo={"valueString": "Royal Bank of Canada"})),
    ]
    fields = [c.payee for c in cheques]
    result = _reconcile_generic_field("payee", fields)
    assert result.alternate_readings is not None
    assert len(result.alternate_readings) == 2
    raw_texts = {r["raw_text"] for r in result.alternate_readings}
    assert "Town of Whitby" in raw_texts
    assert "Royal Bank of Canada" in raw_texts


def test_presence_vs_absence_across_runs_is_a_disagreement():
    """One run returning a field and another not returning it at all is
    itself a disagreement, not silently resolved either way."""
    cheques = [
        to_normalized(azure_doc(Memo={"valueString": "Roll 12345"})),
        to_normalized(azure_doc(Memo=None)),
    ]
    fields = [c.memo for c in cheques]
    result = _reconcile_generic_field("memo", fields)
    assert result.parse_status is ParseStatus.AMBIGUOUS
    assert result.value is None


def test_signature_disagreement_across_runs():
    cheques = [
        to_normalized(azure_doc(PayerSignatures={"valueSignature": "signed"})),
        to_normalized(azure_doc(PayerSignatures=None)),
        to_normalized(azure_doc(PayerSignatures={"valueSignature": "signed"})),
    ]
    fields = [c.signature for c in cheques]
    result = _reconcile_generic_field("signature", fields)
    assert result.parse_status is ParseStatus.AMBIGUOUS
    assert result.value is None


# ---------------------------------------------------------------------------
# _reconcile_amount_words - "any parseable run wins" special case
# ---------------------------------------------------------------------------

def test_one_parseable_run_among_failures_is_preferred():
    cheques = [
        to_normalized(azure_doc(WordAmount={"content": "garbled nonsense text"})),
        to_normalized(azure_doc(WordAmount={
            "content": "One Hundred and twenty five dollars and 50/100 DOLLARS"})),
        to_normalized(azure_doc(WordAmount=None)),
    ]
    fields = [c.amount_words for c in cheques]
    result = _reconcile_amount_words(fields)
    assert result.parse_status is ParseStatus.OK
    assert result.value == Decimal("125.50")


def test_all_runs_unparseable_stays_unparseable():
    cheques = [
        to_normalized(azure_doc(WordAmount={"content": "garbled one"})),
        to_normalized(azure_doc(WordAmount={"content": "garbled two"})),
    ]
    fields = [c.amount_words for c in cheques]
    result = _reconcile_amount_words(fields)
    assert result.parse_status is not ParseStatus.OK
    assert result.value is None
    assert result.alternate_readings is not None
    assert len(result.alternate_readings) == 2


def test_two_different_parseable_amounts_is_genuine_disagreement():
    cheques = [
        to_normalized(azure_doc(WordAmount={"content": "One Hundred DOLLARS"})),
        to_normalized(azure_doc(WordAmount={"content": "Two Hundred DOLLARS"})),
    ]
    fields = [c.amount_words for c in cheques]
    result = _reconcile_amount_words(fields)
    assert result.parse_status is ParseStatus.AMBIGUOUS
    assert result.value is None
    assert "100" in result.note and "200" in result.note


def test_same_parseable_amount_from_all_runs_is_not_flagged_ambiguous():
    cheques = [to_normalized(azure_doc()) for _ in range(3)]
    fields = [c.amount_words for c in cheques]
    result = _reconcile_amount_words(fields)
    assert result.parse_status is ParseStatus.OK
    assert result.value == Decimal("125.50")


# ---------------------------------------------------------------------------
# reconcile_cheques / to_normalized_multi - full pipeline
# ---------------------------------------------------------------------------

def test_to_normalized_multi_end_to_end_agreement():
    docs = [azure_doc() for _ in range(3)]
    cheque = to_normalized_multi(docs)
    assert cheque.payee.parse_status is ParseStatus.OK
    assert cheque.amount_words.value == Decimal("125.50")
    assert cheque.memo.parse_status is ParseStatus.OK


def test_to_normalized_multi_end_to_end_disagreement():
    docs = [
        azure_doc(PayTo={"valueString": "Town of Whitby"}),
        azure_doc(PayTo={"valueString": "Royal Bank of Canada Branch"}),
        azure_doc(PayTo={"valueString": "Town of Whitby"}),
    ]
    cheque = to_normalized_multi(docs)
    assert cheque.payee.parse_status is ParseStatus.AMBIGUOUS
    assert cheque.payee.value is None
    assert cheque.payee.alternate_readings is not None


def test_reconcile_cheques_keeps_source_id_and_raw_response_from_first():
    cheques = [
        to_normalized(azure_doc(), source_id="file.jpg"),
        to_normalized(azure_doc(), source_id="file.jpg"),
    ]
    result = reconcile_cheques(cheques)
    assert result.source_id == "file.jpg"
    assert result.raw_response is not None


def test_reconcile_cheques_requires_at_least_one():
    import pytest
    with pytest.raises(ValueError):
        reconcile_cheques([])
