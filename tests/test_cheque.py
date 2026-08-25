from datetime import date
from decimal import Decimal

import pytest

from chequemate import Config, ParseStatus, RuleStatus, Verdict, validate
from chequemate.extract import to_normalized
from chequemate.normalize import (
    normalize_amount_numeric,
    normalize_amount_words,
    normalize_date,
)

TODAY = date(2026, 8, 17)


def azure_doc(**overrides):
    """Shape of one entry in result.documents, as as_dict() returns it."""
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


# --- word amounts (from the real extractions) -------------------------------

@pytest.mark.parametrize("text,expected", [
    ("One Hundred and twenty five dollars and 50\n/100 DOLLARS", "125.50"),
    ("Three Hundred\n/100 DOLLARS", "300.00"),          # empty cents fraction
    ("Three Hundred and no/100 DOLLARS", "300.00"),
    ("Twenty-five and xx/100", "25.00"),
    ("One thousand two hundred forty seven and 05/100", "1247.05"),
    ("Two million and 99/100 dollars only", "2000000.99"),
    ("Fourty and 10/100", "40.10"),                      # common misspelling
])
def test_word_amounts(text, expected):
    f = normalize_amount_words(text)
    assert f.parse_status is ParseStatus.OK
    assert f.value == Decimal(expected)


def test_word_amount_refuses_to_guess():
    assert normalize_amount_words("One Hundrd and fifty").parse_status \
        is ParseStatus.UNPARSEABLE


# --- numeric amounts --------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("$\n125.50", "125.50"),
    ("$ 300", "300.00"),
    ("$1,247.05", "1247.05"),
    ("2 000,50", "2000.50"),
])
def test_numeric_amounts(text, expected):
    assert normalize_amount_numeric(text).value == Decimal(expected)


# --- dates ------------------------------------------------------------------

def test_date_plain():
    assert normalize_date("17 08 2026").value == date(2026, 8, 17)


def test_date_strips_printed_guide():
    """The real cheque 2 output leaked its pre-printed 'DDM MY YYY' guide."""
    f = normalize_date("15062025\nDDM MY YYY")
    assert f.parse_status is ParseStatus.OK
    assert f.value == date(2025, 6, 15)


def test_date_iso():
    assert normalize_date("2026-08-17").value == date(2026, 8, 17)


def test_ambiguous_date_flagged_not_guessed():
    f = normalize_date("05 06 2026")
    assert f.parse_status is ParseStatus.AMBIGUOUS
    assert f.value == date(2026, 6, 5)          # DMY convention
    assert "2026-05-06" in f.note


def test_unparseable_date():
    assert normalize_date("blurred").parse_status is ParseStatus.UNPARSEABLE


# --- end to end -------------------------------------------------------------

def test_cheque_one_is_valid():
    result = validate(to_normalized(azure_doc()), today=TODAY)
    assert result.verdict is Verdict.VALID


def test_leading_the_is_not_a_spelling_error():
    doc = azure_doc(PayTo={"valueString": "Town of Whitby"})
    assert validate(to_normalized(doc), today=TODAY).verdict is Verdict.VALID


def test_minor_payee_typo_is_tolerated():
    """Real cheque OCR routinely misreads 'Whitby' by 1-2 characters
    (whitty, whitley, white...) — a single dropped letter must not flip a
    legitimate cheque to INVALID."""
    doc = azure_doc(PayTo={"valueString": "Town of Witby"})  # missing 'h'
    r = validate(to_normalized(doc), today=TODAY)
    assert r.verdict is Verdict.VALID
    payee_result = next(f for f in r.rules if f.rule_id == "payee")
    assert payee_result.status is RuleStatus.PASS


def test_wrong_payee_fails():
    doc = azure_doc(PayTo={"valueString": "City of Oshawa"})
    (f,) = validate(to_normalized(doc), today=TODAY).failures
    assert f.rule_id == "payee" and "does not reference" in f.evidence


def test_bank_name_in_payee_field_is_unable_not_fail():
    """Azure's PayTo extraction sometimes grabs the drawee bank's own
    letterhead instead of the handwritten payee line — that's a missed
    extraction, not proof of a wrong payee, so it should read UNABLE."""
    doc = azure_doc(PayTo={
        "valueString": "ROYAL BANK OF CANADA FINCH & MCCOWAN BRANCH"})
    r = validate(to_normalized(doc), today=TODAY)
    payee_result = next(f for f in r.rules if f.rule_id == "payee")
    assert payee_result.status is RuleStatus.UNABLE
    assert "bank" in payee_result.evidence.lower()


def test_amount_mismatch_fails():
    doc = azure_doc(WordAmount={"content": "One Hundred and 50/100 DOLLARS"})
    (f,) = validate(to_normalized(doc), today=TODAY).failures
    assert f.rule_id == "amount_match" and "BEA s.28(2)" in f.evidence


def test_missing_signature_field_is_unable_and_does_not_invalidate():
    """Cheque 2: no PayerSignatures key at all. UNABLE means 'couldn't tell',
    not 'known to be wrong' — it still needs a human look (surfaced via
    .failures) but must not by itself flip an otherwise-clean cheque to
    INVALID."""
    doc = azure_doc(PayerSignatures=None)
    r = validate(to_normalized(doc), today=TODAY)
    (f,) = r.failures
    assert f.rule_id == "signature" and f.status is RuleStatus.UNABLE
    assert r.verdict is Verdict.VALID


def test_signature_read_from_bounding_regions_only():
    doc = azure_doc(PayerSignatures={"boundingRegions": [{"pageNumber": 1}]})
    assert validate(to_normalized(doc), today=TODAY).verdict is Verdict.VALID


def test_unsigned_cheque_fails():
    doc = azure_doc(PayerSignatures={"valueSignature": "unsigned"})
    (f,) = validate(to_normalized(doc), today=TODAY).failures
    assert f.status is RuleStatus.FAIL


def test_stale_dated_cheque_fails():
    doc = azure_doc(CheckDate={"content": "15062025\nDDM MY YYY"})
    (f,) = validate(to_normalized(doc), today=TODAY).failures
    assert f.rule_id == "date" and "stale-dated" in f.evidence


def test_boundary_exactly_six_months_passes():
    doc = azure_doc(CheckDate={"content": "17 02 2026"})
    assert validate(to_normalized(doc), today=TODAY).verdict is Verdict.VALID


def test_one_day_past_six_months_fails():
    doc = azure_doc(CheckDate={"content": "16 02 2026"})
    (f,) = validate(to_normalized(doc), today=TODAY).failures
    assert f.rule_id == "date"


def test_postdated_cheque_passes_but_is_labelled():
    """Post-dated tax instalment cheques are a normal, expected category —
    they must PASS (and not invalidate the cheque), but the message still
    flags them for routing to post-dated batch handling."""
    doc = azure_doc(CheckDate={"content": "01 12 2026"})
    r = validate(to_normalized(doc), today=TODAY)
    date_result = next(f for f in r.rules if f.rule_id == "date")
    assert date_result.status is RuleStatus.PASS
    assert "post-dated" in date_result.evidence
    assert "post-dated batch" in date_result.evidence
    assert r.verdict is Verdict.VALID


def test_postdated_cheque_still_fails_when_configured_strict():
    doc = azure_doc(CheckDate={"content": "01 12 2026"})
    cfg = Config(reject_postdated=True)
    r = validate(to_normalized(doc), cfg, today=TODAY)
    (f,) = r.failures
    assert f.rule_id == "date" and "post-dated" in f.evidence
    assert r.verdict is Verdict.INVALID


def test_multiple_failures_all_reported():
    doc = azure_doc(
        PayTo={"valueString": "Town of Ajax"},
        PayerSignatures={"valueSignature": "unsigned"},
    )
    r = validate(to_normalized(doc), today=TODAY)
    assert {f.rule_id for f in r.failures} == {"payee", "signature"}


def test_strictness_is_configurable():
    doc = azure_doc(PayTo={"valueString": "Town of Whitbv"})  # OCR y->v
    cfg = Config(payee_edit_tolerance=1)
    assert validate(to_normalized(doc), cfg, today=TODAY).verdict is Verdict.VALID


def test_blank_memo_fails_and_invalidates():
    doc = azure_doc(Memo=None)
    r = validate(to_normalized(doc), today=TODAY)
    (f,) = r.failures
    assert f.rule_id == "memo" and f.status is RuleStatus.FAIL
    assert r.verdict is Verdict.INVALID


def test_memo_present_passes():
    doc = azure_doc(Memo={"valueString": "Water bill Q3"})
    r = validate(to_normalized(doc), today=TODAY)
    assert r.verdict is Verdict.VALID
    memo_result = next(f for f in r.rules if f.rule_id == "memo")
    assert memo_result.status is RuleStatus.PASS


# --- signature diagnostics --------------------------------------------------

def test_diagnose_reports_absent_key():
    from chequemate.extract import diagnose_signature
    out = diagnose_signature(azure_doc(PayerSignatures=None))
    assert "KEY ABSENT" in out and "PayTo" in out


def test_diagnose_reports_detection():
    from chequemate.extract import diagnose_signature
    out = diagnose_signature(azure_doc())
    assert "valueSignature" in out and "detected = True" in out


def test_first_document_raises_on_empty():
    from chequemate.extract import first_document
    with pytest.raises(ValueError):
        first_document({"documents": []})


# --- adopted from the pre-split monolith ------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("12 Mar 2026", date(2026, 3, 12)),
    ("12 March 2026", date(2026, 3, 12)),
    ("March 12 2026", date(2026, 3, 12)),
    ("Mar 12, 2026", date(2026, 3, 12)),
    ("12th March, 2026", date(2026, 3, 12)),
    ("1 Sept 2026", date(2026, 9, 1)),
])
def test_month_name_dates(text, expected):
    f = normalize_date(text)
    assert f.parse_status is ParseStatus.OK   # never AMBIGUOUS
    assert f.value == expected


@pytest.mark.parametrize("text,expected", [
    ("Three Hundred dollars and sixty cents", "300.60"),
    ("One Hundred and twenty five dollars and fifty cents", "125.50"),
    ("Three Hundred and none/100 DOLLARS", "300.00"),
    ("Fivety and 25/100", "50.25"),          # misspelling table
    ("Ninty eigthy and xx/100", "170.00"),
    ("Two Hundered and 05/100", "200.05"),
    ("One Thousend and no/100", "1000.00"),
])
def test_word_amount_extras(text, expected):
    f = normalize_amount_words(text)
    assert f.parse_status is ParseStatus.OK
    assert f.value == Decimal(expected)


def test_spelled_cents_not_folded_into_dollars():
    """'and sixty cents' must not become $360."""
    assert normalize_amount_words(
        "Three Hundred dollars and sixty cents").value == Decimal("300.60")


def test_stray_digits_in_word_amount_rejected():
    f = normalize_amount_words("One Hundred 25 dollars")
    assert f.parse_status is ParseStatus.UNPARSEABLE
    assert "stray digits" in f.note


def test_malformed_numeric_amount_rejected():
    """'125.501' must not silently become 125501.00."""
    f = normalize_amount_numeric("$125.501")
    assert f.parse_status is ParseStatus.UNPARSEABLE
    assert "malformed" in f.note


def test_no_fuzzy_snapping_on_unknown_words():
    """Unknown number words are refused, never snapped to a neighbour."""
    for text in ("Sixtty and 00/100", "One Hundrd and fifty",
                 "Seventty five and 10/100"):
        assert normalize_amount_words(text).parse_status \
            is ParseStatus.UNPARSEABLE


def test_payee_matches_on_town_name_alone():
    """Real payees are phrased many ways ('Town Whitby' missing 'of',
    'Whitby Taxes', a drawer's name printed next to 'Town of Whitby') — the
    town name itself is what matters, not exact whole-string phrasing."""
    doc = azure_doc(PayTo={"valueString": "Town Whitby"})
    assert validate(to_normalized(doc), today=TODAY).verdict is Verdict.VALID


def test_cents_only_amount():
    assert normalize_amount_words("sixty cents only").value == Decimal("0.60")


def test_unknown_dollar_word_with_valid_cents_still_fails():
    """Regression: this parsed as $0.10 before the meaningful-token check."""
    f = normalize_amount_words("Seventty five and 10/100")
    assert f.parse_status is ParseStatus.UNPARSEABLE