from datetime import date
from decimal import Decimal

import pytest

from chequemate import Config, ParseStatus, RuleStatus, Verdict, validate
from chequemate.extract import to_normalized
from chequemate.normalize import (
    _ONES,
    _SCALES,
    _TENS,
    expected_amount_token_forms,
    normalize_amount_numeric,
    normalize_amount_words,
    normalize_date,
    verify_amount_by_tokens,
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


# --- six-digit date resolution: converge or refuse (ruleset 1.10.0) --------

def test_real_record_0012_diverges_to_unable_not_a_wrong_year():
    """Regression: CHQ-20260824-0012's '26-08-30' used to confidently parse
    as 2030-08-26 (the old 6-digit heuristic misfiring on a truncated grid
    read as if it were a genuine short year). Both hypotheses now use the
    SAME pivot convention and genuinely disagree (2026-08-30 vs
    2030-08-26) - the honest answer is UNABLE, not a pick of either one."""
    f = normalize_date("26-08-30")
    assert f.parse_status is ParseStatus.UNPARSEABLE
    assert "2026-08-30" in f.note and "2030-08-26" in f.note


def test_six_digit_convergent_reading_resolves():
    """Constructed so the day-pair equals the year-suffix-pair (15/08/15):
    DDMMYY and the truncated-year-leading-grid reading trivially agree,
    MDY is invalid (month=15) - exactly one candidate, so this resolves."""
    f = normalize_date("150815")
    assert f.parse_status is ParseStatus.OK
    assert f.value == date(2015, 8, 15)
    assert f.note is not None and "resolved via" in f.note


def test_six_digit_date_never_confidently_wrong_across_the_full_space():
    """General invariant, asserted across the space rather than per-case
    (28 x 12 x 100 = 33,600 combinations of day/month/2-digit-year, one
    test): for EVERY 6-digit DDMMYY string, normalize_date either (a)
    resolves OK, and only when that value is the one every valid
    hypothesis (DDMMYY, MMDDYY where applicable, truncated-year-leading-
    grid) that parses successfully agrees on, or (b) refuses
    (UNPARSEABLE). It must never silently return a value some other
    equally-valid reading of the same six digits contradicts."""
    def try_(y, m, d):
        try:
            return date(y + (2000 if y < 70 else 1900), m, d) if y < 100 else None
        except ValueError:
            return None

    checked = 0
    for dd in range(1, 29):
        for mm in range(1, 13):
            for yy in range(0, 100):
                digits = f"{dd:02d}{mm:02d}{yy:02d}"
                f = normalize_date(digits)
                candidates = {v for v in (
                    try_(int(digits[4:6]), int(digits[2:4]), int(digits[0:2])),
                    try_(int(digits[4:6]), int(digits[0:2]), int(digits[2:4])),
                    try_(int(digits[0:2]), int(digits[2:4]), int(digits[4:6])),
                ) if v is not None}

                if len(candidates) == 1:
                    assert f.parse_status is ParseStatus.OK
                    assert f.value == next(iter(candidates))
                else:
                    assert f.parse_status is ParseStatus.UNPARSEABLE, (
                        f"digits={digits!r} candidates={candidates} but got "
                        f"a confident value {f.value!r} despite "
                        f"{'no' if not candidates else 'multiple non-converging'} "
                        f"valid reading(s)")
                checked += 1
    assert checked == 28 * 12 * 100


@pytest.mark.parametrize("raw,digit_count", [
    ("026-08-28", 7),   # real CHQ-20260824-0004
    ("2\n082026", 7),   # real CHQ-20260824-0019
])
def test_seven_digit_dates_get_no_new_repair(raw, digit_count):
    """A 7-digit read is missing exactly one digit from an assumed 8-digit
    grid with no independently-motivated second hypothesis to check
    convergence against - every one of the 10 possible values for the
    missing digit produces an equally structurally-valid date (verified:
    day/month validity essentially never depends on the missing year
    digit), so filling one in would be an unjustified plausibility guess,
    not a structural repair. Must stay UNABLE exactly as before."""
    f = normalize_date(raw)
    assert f.parse_status is ParseStatus.UNPARSEABLE
    assert f"got {digit_count}" in f.note


def test_contaminated_and_correction_mark_dates_stay_unable():
    """CHQ-20260824-0005 (cheque-number contamination, 9 digits) and
    CHQ-20260824-0020 (drawer wrote the date twice, in two formats, 10
    digits) are not form properties and are not engineered around - both
    still correctly refuse."""
    f5 = normalize_date("26- 08-28\n011")
    assert f5.parse_status is ParseStatus.UNPARSEABLE
    f20 = normalize_date("458\n0\n0\n31\n26\nAu\n2")
    assert f20.parse_status is ParseStatus.UNPARSEABLE


def test_six_digit_note_visible_in_rule_evidence():
    """The repaired-reading provenance (or the divergence reason) must be
    visible in the report, not just internal to Field.note - check_date
    now surfaces any note, not just AMBIGUOUS ones."""
    doc = azure_doc(CheckDate={"content": "150815"})
    r = validate(to_normalized(doc), today=date(2015, 8, 20))
    dr = next(x for x in r.rules if x.rule_id == "date")
    assert "resolved via" in dr.evidence


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


def test_bank_letterhead_payee_with_no_town_reference_fails():
    """Ruleset 1.7.0: presence is sufficient, so text with zero reference
    to the expected payee - bank letterhead or not - is a plain FAIL, not
    a special-cased UNABLE. (Earlier rulesets routed bank/branch-looking
    text to UNABLE on the theory it was probably a missed extraction; that
    carve-out is gone - see check_payee's docstring.)"""
    doc = azure_doc(PayTo={
        "valueString": "ROYAL BANK OF CANADA FINCH & MCCOWAN BRANCH"})
    r = validate(to_normalized(doc), today=TODAY)
    payee_result = next(f for f in r.rules if f.rule_id == "payee")
    assert payee_result.status is RuleStatus.FAIL


def test_bank_letterhead_with_town_reference_still_passes():
    """Extra text (bank letterhead merged in by a bad extraction) is
    ignored as long as the distinctive payee token is present somewhere."""
    doc = azure_doc(PayTo={
        "valueString": "RBC Town of Whitby FINCH & MCCOWAN BRANCH"})
    r = validate(to_normalized(doc), today=TODAY)
    payee_result = next(f for f in r.rules if f.rule_id == "payee")
    assert payee_result.status is RuleStatus.PASS


def test_clean_amount_mismatch_fails():
    """Ruleset 1.8.0: a mismatch between two CLEAN (non-degraded) readings
    is a genuine contradiction between two independent sources - still
    FAILs. No printed cents-suffix construct here at all, so no repair was
    needed on either side."""
    doc = azure_doc(WordAmount={"content": "One Hundred Dollars"})
    (f,) = validate(to_normalized(doc), today=TODAY).failures
    assert f.rule_id == "amount_match" and "BEA s.28(2)" in f.evidence


def test_degraded_amount_mismatch_is_unable_not_fail():
    """Ruleset 1.8.0's core invariant: a written-amount reading that only
    exists because the cents suffix was repaired (numeric-cents fallback)
    is not equivalent evidence to a clean reading. Disagreeing with the
    numeral is insufficient evidence of a genuine mismatch - not proof of
    one - so this must resolve to UNABLE, never FAIL."""
    doc = azure_doc(WordAmount={"content": "One Hundred and 50/100 DOLLARS"})
    r = validate(to_normalized(doc), today=TODAY)
    amount_result = next(f for f in r.rules if f.rule_id == "amount_match")
    assert amount_result.status is RuleStatus.UNABLE
    assert "insufficient evidence" in amount_result.evidence.lower()
    assert r.verdict is Verdict.REVIEW


def test_degraded_amount_match_passes_via_corroboration():
    """Two independent sources converging is itself the PASS condition,
    even when the written-amount side needed the cents-suffix repair to
    get there - real corroboration, not a clean primary read, and the
    evidence text says so plainly."""
    doc = azure_doc(WordAmount={"content": "Nine Hundred Sixty five and 460 100 DOLLARS"},
                    NumberAmount={"content": "$ 965.47"})
    r = validate(to_normalized(doc), today=TODAY)
    amount_result = next(f for f in r.rules if f.rule_id == "amount_match")
    assert amount_result.status is RuleStatus.PASS
    assert "corroboration" in amount_result.evidence.lower()
    assert r.verdict is Verdict.VALID


@pytest.mark.parametrize("words_text,numeric_text", [
    # the real CHQ-20260824-0010 case: cents-suffix repair correctly
    # discards '4/kg', but the remaining text still mis-parses a
    # hyphenated compound number ('-eight-one') to the wrong dollar figure
    ("one thousand nine hundred -eight-one 4/kg 100 Dollars", "$ 1,981.46"),
    ("One Hundred and 50/100 DOLLARS", "$ 125.50"),
    ("Five hundred dollars 1 100 Dollars", "$ 400.00"),
    ("Three Hundred /100 DOLLARS", "$ 250.00"),
])
def test_no_transformation_converts_unable_into_fail(words_text, numeric_text):
    """General invariant (ruleset 1.8.0): whenever the written amount only
    parsed via the cents-suffix repair and still disagrees with the
    numeral, the rule must never assert a confident FAIL - the repair
    itself means this was never a fully independent reading. UNABLE is the
    ceiling; FAIL requires two clean readings actually contradicting."""
    doc = azure_doc(WordAmount={"content": words_text},
                    NumberAmount={"content": numeric_text})
    r = validate(to_normalized(doc), today=TODAY)
    amount_result = next(f for f in r.rules if f.rule_id == "amount_match")
    assert amount_result.status is not RuleStatus.FAIL


def test_amount_words_unparseable_surfaces_numeric_amount_and_stays_unable():
    """Ruleset 1.6.0: when the numeric amount parses cleanly but the words
    amount doesn't, the rule's own message must show the numeric figure
    and say plainly that the written amount couldn't be read - not just
    describe the words failure in isolation. Still UNABLE (-> REVIEW),
    never a silent PASS on the numeric alone."""
    doc = azure_doc(WordAmount={"content": "sihatred deputy sixtillers severity"})
    r = validate(to_normalized(doc), today=TODAY)
    amount_result = next(f for f in r.rules if f.rule_id == "amount_match")
    assert amount_result.status is RuleStatus.UNABLE
    assert "125.5" in amount_result.evidence  # the numeric amount, plainly stated
    assert "written amount" in amount_result.evidence.lower()
    assert "could not be read" in amount_result.evidence.lower()
    assert r.verdict is Verdict.REVIEW


def test_missing_signature_field_is_unable_and_routes_to_review():
    """Cheque 2: no PayerSignatures key at all. UNABLE means 'couldn't tell',
    not 'known to be wrong' — it must not flip an otherwise-clean cheque to
    INVALID, but it must not be silently waved through as VALID either: it
    needs a human look, which is exactly what REVIEW means."""
    doc = azure_doc(PayerSignatures=None)
    r = validate(to_normalized(doc), today=TODAY)
    (f,) = r.failures
    assert f.rule_id == "signature" and f.status is RuleStatus.UNABLE
    assert r.verdict is Verdict.REVIEW


def test_signature_read_from_bounding_regions_only():
    doc = azure_doc(PayerSignatures={"boundingRegions": [{"pageNumber": 1}]})
    assert validate(to_normalized(doc), today=TODAY).verdict is Verdict.VALID


def test_unsigned_cheque_fails():
    doc = azure_doc(PayerSignatures={"valueSignature": "unsigned"})
    (f,) = validate(to_normalized(doc), today=TODAY).failures
    assert f.status is RuleStatus.FAIL


def test_orientation_indeterminate_maps_to_unable_not_fail_or_pass():
    """The pipeline-boundary contract for imageprep.OrientationIndeterminate
    (Phase 3, Amendment 3): a signature reading from an unresolvable
    orientation must never look like a confident PASS or FAIL. It must
    become ParseStatus.AMBIGUOUS -> RuleStatus.UNABLE, which must not
    invalidate an otherwise-clean cheque outright, but also must not be
    silently reported as VALID — it routes to REVIEW."""
    from chequemate.imageprep import OrientationIndeterminate
    from chequemate.normalize import normalize_signature

    try:
        raise OrientationIndeterminate(
            "neither orientation cleared the confidence floor",
            top_score=0.31, bottom_score=0.30, dpi=300.0, dpi_source="jfif")
    except OrientationIndeterminate as exc:
        sig = normalize_signature(None, ambiguous_reason=str(exc))

    assert sig.parse_status is ParseStatus.AMBIGUOUS

    doc = azure_doc()
    cheque = to_normalized(doc)
    cheque.signature = sig
    result = validate(cheque, today=TODAY)

    sig_result = next(r for r in result.rules if r.rule_id == "signature")
    assert sig_result.status is RuleStatus.UNABLE
    assert result.verdict is Verdict.REVIEW


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


def test_fail_beats_unable_for_verdict():
    """A cheque with one genuine FAIL and one UNABLE must be INVALID, not
    REVIEW - a known-wrong field always outranks an unreadable one."""
    doc = azure_doc(
        PayTo={"valueString": "City of Oshawa"},  # FAIL
        PayerSignatures=None,                      # UNABLE
    )
    r = validate(to_normalized(doc), today=TODAY)
    statuses = {f.rule_id: f.status for f in r.rules}
    assert statuses["payee"] is RuleStatus.FAIL
    assert statuses["signature"] is RuleStatus.UNABLE
    assert r.verdict is Verdict.INVALID


def test_review_verdict_only_when_no_fail_present():
    """The exact three-way split: FAIL anywhere -> INVALID; no FAIL but at
    least one UNABLE -> REVIEW; no FAIL and no UNABLE -> VALID."""
    clean = validate(to_normalized(azure_doc()), today=TODAY)
    assert clean.verdict is Verdict.VALID

    unable_only = validate(to_normalized(azure_doc(PayerSignatures=None)), today=TODAY)
    assert unable_only.verdict is Verdict.REVIEW

    with_fail = validate(to_normalized(azure_doc(
        PayerSignatures=None, PayTo={"valueString": "City of Oshawa"})), today=TODAY)
    assert with_fail.verdict is Verdict.INVALID


def test_strictness_is_configurable():
    doc = azure_doc(PayTo={"valueString": "Town of Whitbv"})  # OCR y->v
    cfg = Config(payee_edit_tolerance=1)
    assert validate(to_normalized(doc), cfg, today=TODAY).verdict is Verdict.VALID


def test_blank_memo_is_unable_and_routes_to_review():
    """Ruleset 1.6.0: a blank/absent memo can't be confidently told apart
    from Azure simply never returning the field (a real, confirmed
    extraction gap on real cheques) - see rules.check_memo's docstring.
    UNABLE -> REVIEW, never a hard FAIL -> INVALID."""
    doc = azure_doc(Memo=None)
    r = validate(to_normalized(doc), today=TODAY)
    (f,) = r.failures
    assert f.rule_id == "memo" and f.status is RuleStatus.UNABLE
    assert r.verdict is Verdict.REVIEW


def test_memo_present_passes():
    doc = azure_doc(Memo={"valueString": "Water bill Q3"})
    r = validate(to_normalized(doc), today=TODAY)
    assert r.verdict is Verdict.VALID
    memo_result = next(f for f in r.rules if f.rule_id == "memo")
    assert memo_result.status is RuleStatus.PASS


# ---------------------------------------------------------------------------
# repeat-extraction disagreement (ruleset 1.6.0): every rule must route an
# AMBIGUOUS field with no chosen value (extract.reconcile_cheques's
# disagreement case) to UNABLE, never attempt to read a fabricated value.
# Fields are hand-built here (not via azure_doc/to_normalized) since only
# reconcile_cheques ever actually produces this state - see
# tests/test_extract_multi.py for that production path end-to-end.
# ---------------------------------------------------------------------------

from chequemate.models import Field, NormalizedCheque  # noqa: E402
from chequemate.rules import (  # noqa: E402
    check_amounts_match, check_date, check_memo, check_payee,
)


def _ambiguous_disagreement_field(name: str) -> Field:
    return Field(name, value=None, raw_text="something", parse_status=ParseStatus.AMBIGUOUS,
                note="3 repeat-extraction runs disagreed on this field",
                alternate_readings=[{"raw_text": "a"}, {"raw_text": "b"}, {"raw_text": "c"}])


def _base_cheque(**overrides) -> NormalizedCheque:
    doc = azure_doc()
    cheque = to_normalized(doc)
    for name, value in overrides.items():
        setattr(cheque, name, value)
    return cheque


def test_check_payee_routes_disagreement_ambiguous_to_unable():
    cheque = _base_cheque(payee=_ambiguous_disagreement_field("payee"))
    result = check_payee(cheque, Config())
    assert result.status is RuleStatus.UNABLE
    assert "disagreed" in result.evidence.lower()


def test_check_date_routes_disagreement_ambiguous_to_unable():
    cheque = _base_cheque(cheque_date=_ambiguous_disagreement_field("cheque_date"))
    result = check_date(cheque, Config(), today=TODAY)
    assert result.status is RuleStatus.UNABLE
    assert "disagreed" in result.evidence.lower()


def test_check_memo_routes_disagreement_ambiguous_to_unable():
    cheque = _base_cheque(memo=_ambiguous_disagreement_field("memo"))
    result = check_memo(cheque, Config())
    assert result.status is RuleStatus.UNABLE
    assert "disagreed" in result.evidence.lower()


def test_check_amounts_match_routes_numeric_disagreement_to_unable():
    cheque = _base_cheque(amount_numeric=_ambiguous_disagreement_field("amount_numeric"))
    result = check_amounts_match(cheque, Config())
    assert result.status is RuleStatus.UNABLE
    assert "numeric" in result.evidence.lower()
    assert "disagreed" in result.evidence.lower()


def test_check_amounts_match_routes_words_disagreement_to_unable_and_shows_numeric():
    cheque = _base_cheque(amount_words=_ambiguous_disagreement_field("amount_words"))
    result = check_amounts_match(cheque, Config())
    assert result.status is RuleStatus.UNABLE
    assert "125.5" in result.evidence  # numeric amount still surfaced
    assert "disagreed" in result.evidence.lower()


def test_disagreement_ambiguous_never_silently_passes():
    """The core safety property: disagreement must never be treated as a
    confident value that lets the overall verdict come out clean."""
    for field_name in ("payee", "cheque_date", "memo", "amount_numeric", "amount_words"):
        cheque = _base_cheque(**{field_name: _ambiguous_disagreement_field(field_name)})
        r = validate(cheque, Config(), today=TODAY)
        assert r.verdict is Verdict.REVIEW, (
            f"a disagreement-ambiguous {field_name} must route to REVIEW, "
            f"got {r.verdict}")


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


# --- numeric-cents fallback for the printed '.../100' construct (1.7.0) ----

@pytest.mark.parametrize("text,numeric_cents,expected", [
    # garbled numerator with no '/' at all - real batch failure
    ("Nine Hundred Sixty five and 460 100 DOLLARS", 47, "965.47"),
    # numerator merged with unrelated OCR garbage - real batch failure
    ("one thousand nine hundred eighty one 4/kg 100 Dollars", 46, "1981.46"),
    # stray single-digit numerator - real batch failure
    ("Five hundred dollars 1 100 Dollars", 0, "500.00"),
    # a clean, well-formed '/100' still works exactly as before
    ("One Hundred and twenty five dollars and 50/100 DOLLARS", 50, "125.50"),
    # empty-numerator forms still work
    ("Three Hundred /100 DOLLARS", 0, "300.00"),
])
def test_garbled_cents_suffix_recovered_via_numeric_cents(text, numeric_cents, expected):
    f = normalize_amount_words(text, numeric_cents=numeric_cents)
    assert f.parse_status is ParseStatus.OK, (text, f.note)
    assert f.value == Decimal(expected)


def test_numeric_cents_fallback_is_noted_for_provenance():
    """A fallback-resolved reading must be distinguishable from a clean
    primary read - the note carries that provenance."""
    f = normalize_amount_words(
        "Nine Hundred Sixty five and 460 100 DOLLARS", numeric_cents=47)
    assert f.note is not None
    assert "numeric amount field" in f.note


def test_well_formed_cents_suffix_still_carries_fallback_note():
    """Ruleset 1.7.0 stops parsing the printed '.../100' numerator
    unconditionally - even a well-formed '50/100' is discarded rather than
    read, so its cents always come from numeric_cents, and the note always
    reflects that (the words field's cents digit is never independently
    confirmed any more when numeric_cents is supplied)."""
    f = normalize_amount_words(
        "One Hundred and twenty five dollars and 50/100 DOLLARS",
        numeric_cents=50)
    assert f.value == Decimal("125.50")
    assert f.note is not None and "numeric amount field" in f.note


def test_spelled_cents_still_wins_over_numeric_cents():
    """Genuinely spelled-out cents ('and sixty cents') are an independent
    written statement, not the printed boilerplate construct - they still
    take priority over the numeric-cents fallback."""
    f = normalize_amount_words(
        "Three Hundred dollars and sixty cents", numeric_cents=99)
    assert f.value == Decimal("300.60")
    assert f.note is None


def test_stray_digit_outside_cents_zone_still_rejected_with_numeric_cents():
    """A stray digit that is NOT part of the trailing '.../100' construct
    is still genuinely ambiguous (real contamination of the dollar words,
    not boilerplate) and must still be rejected, even when numeric_cents
    is available."""
    f = normalize_amount_words(
        "One Hundred 25 dollars and 50/100 DOLLARS", numeric_cents=50)
    assert f.parse_status is ParseStatus.UNPARSEABLE
    assert "stray digits" in f.note


def test_gibberish_dollar_words_stay_unparseable_not_falsely_zeroed():
    """Regression: a '100' anchor must only pull in digit/slash-shaped
    numerator noise, never sweep genuinely unparseable (but non-numeric)
    dollar-words text into the discard zone. Discarding
    'sihatred deputy sixtillers severity' along with the anchor would
    silently produce dollars=0 - a confidently wrong parsed value - instead
    of correctly staying UNPARSEABLE."""
    f = normalize_amount_words(
        "sihatred deputy sixtillers severity 100 DOLLARS", numeric_cents=76)
    assert f.parse_status is ParseStatus.UNPARSEABLE


def test_ocr_punctuation_glued_to_real_word_is_not_discarded():
    """Regression: 'hundred->>' (a real dollar word OCR-glued to unrelated
    punctuation) must be kept, not swept away as numerator noise just
    because stripping only the token's edges left '>>' unrecognised."""
    f = normalize_amount_words(
        "Three thousand five hundred->> 100 DOLLARS", numeric_cents=0)
    assert f.parse_status is ParseStatus.OK
    assert f.value == Decimal("3500.00")


def test_no_anchor_no_fallback_used():
    """With no recognisable '.../100'-shaped construct anywhere, there is
    nothing to discard - numeric_cents must not be silently substituted in
    for a genuinely unrecognisable words amount."""
    f = normalize_amount_words("sihatred deputy sixtillers severity",
                               numeric_cents=50)
    assert f.parse_status is ParseStatus.UNPARSEABLE


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


# --- token-match amount verification fallback (ruleset 1.9.0) --------------

def test_token_match_basic_example():
    """The exact example from the brief: $933 -> nine, hundred, thirty,
    three, all present, no unexpected scale word."""
    r = verify_amount_by_tokens("Nine Hundred and thirty three dallas", 933)
    assert r.outcome == "matched"


@pytest.mark.parametrize("altered_text", [
    "nine thousand three hundred thirty",  # the exact scale-trap example given
])
def test_scale_trap_933_vs_9330_is_caught(altered_text):
    r = verify_amount_by_tokens(altered_text, 933)
    assert r.outcome == "contradiction"
    assert "thousand" in r.unexpected_claims


@pytest.mark.parametrize("numeral,tenx_text", [
    (650, "six thousand five hundred"),      # $650 vs $6,500, literal
    (1208, "twelve thousand eighty"),        # $1,208 vs $12,080, literal
])
def test_10x_alteration_never_produces_a_false_pass(numeral, tenx_text):
    """Every 10x alteration must not pass — whichever mechanism catches it
    (missing tokens or a contradicting scale claim), the outcome must
    never be 'matched'."""
    r = verify_amount_by_tokens(tenx_text, numeral)
    assert r.outcome != "matched"


def test_10x_alteration_scale_guard_proven_with_adversarial_overlap():
    """The literal $650/$6,500 and $1,208/$12,080 pairs above happen to be
    caught by missing tokens alone (their natural phrasings don't share
    every token). Constructed here: text sharing ALL of $650's expected
    tokens {six, hundred, fifty} while genuinely meaning $6,550 - this is
    what actually requires the scale guard, not just token presence."""
    r = verify_amount_by_tokens("six thousand five hundred fifty", 650)
    assert r.outcome == "contradiction"
    assert r.unexpected_claims.get("thousand") == 6


def test_no_real_number_word_fuzzy_collides_with_a_different_one():
    """The categorical safety rule (see normalize.py's token-match section
    docstring): a text token that is itself a real, recognised number/
    scale word is never fuzzy-substituted for a different expected token,
    checked against the FULL vocabulary, not just the named pairs. This is
    what actually prevents nine/five, six/ten, two/ten, eight/eighty and
    million/billion from colliding - not the distance bound alone."""
    from chequemate.normalize import _match_form
    value_of = {**_ONES, **_TENS, "hundred": "hundred"}
    for name, mult in _SCALES.items():
        value_of[name] = mult
    real_words = {w for w in value_of if w != "zero"}
    for a in real_words:
        for b in real_words:
            if a == b or value_of[a] == value_of[b]:
                continue  # same value, e.g. forty/fourty - a real synonym, not a collision
            ok, _found, _missing = _match_form([a], [b])
            assert not ok, f"{b!r} must never satisfy expected {a!r} (both real words)"


@pytest.mark.parametrize("a,b", [("nine", "five"), ("six", "ten"), ("two", "ten")])
def test_named_short_word_pairs_do_not_collide(a, b):
    """The specific pairs named in the brief, checked directly against the
    matching primitive (not just distance, since distance alone doesn't
    tell the whole story - see the module docstring)."""
    from chequemate.normalize import _match_form
    ok, _, _ = _match_form([a], [b])
    assert not ok
    ok, _, _ = _match_form([b], [a])
    assert not ok


def test_alternate_phrasing_compact_hundreds():
    """$1,150 as 'eleven hundred fifty' - a real, different token set for
    the same amount (not the standard 'one thousand one hundred fifty')."""
    r = verify_amount_by_tokens("eleven hundred fifty", 1150)
    assert r.outcome == "matched"
    r = verify_amount_by_tokens("one thousand one hundred fifty", 1150)
    assert r.outcome == "matched"


def test_alternate_phrasing_matches_real_batch_record_0008():
    """CHQ-20260824-0008 is genuinely written 'Twenty-two hundred eighteen'
    for $2,218.36 - confirms the compact-hundreds form isn't speculative.
    numeral_cents=36 matches the real cheque's actual cents (the trailing
    '36' in the raw text is its genuine, un-stripped numerator, not a
    discrepancy - see the stray-digit guard tests)."""
    r = verify_amount_by_tokens("Twenty- two hundred eighteen - 36", 2218, numeral_cents=36)
    assert r.outcome == "matched"


def test_hyphenation_and_and_do_not_affect_matching():
    r1 = verify_amount_by_tokens("nine hundred and thirty-three dollars", 933)
    r2 = verify_amount_by_tokens("nine hundred thirty three dollars", 933)
    assert r1.outcome == r2.outcome == "matched"


def test_fuzzy_ocr_typo_still_matches():
    """The brief's own named typo forms for 'thirty'."""
    for typo in ("thirtty", "thurty"):
        r = verify_amount_by_tokens(f"nine hundred and {typo} three dollars", 933)
        assert r.outcome == "matched", typo


def test_total_gibberish_stays_tokens_missing_not_matched():
    r = verify_amount_by_tokens("sihatred deputy sixtillers severity", 1688)
    assert r.outcome == "tokens_missing"


def test_token_match_licensing_via_check_amounts_match():
    """The three outcomes, run through the actual rule (not just the
    normalize.py primitive), confirming the licensing table: matched ->
    PASS, contradiction -> FAIL, tokens_missing -> UNABLE."""
    matched_doc = azure_doc(WordAmount={"content": "Nine Hundred and thirty three dallas"},
                            NumberAmount={"content": "$933.00"})
    r = validate(to_normalized(matched_doc), today=TODAY)
    ar = next(x for x in r.rules if x.rule_id == "amount_match")
    assert ar.status is RuleStatus.PASS
    assert "token-match" in ar.evidence.lower()

    contradiction_doc = azure_doc(WordAmount={"content": "nine thousand three hundred thirty"},
                                  NumberAmount={"content": "$933.00"})
    r = validate(to_normalized(contradiction_doc), today=TODAY)
    ar = next(x for x in r.rules if x.rule_id == "amount_match")
    assert ar.status is RuleStatus.FAIL

    missing_doc = azure_doc(WordAmount={"content": "sihatred deputy sixtillers severity"},
                            NumberAmount={"content": "$1688.76"})
    r = validate(to_normalized(missing_doc), today=TODAY)
    ar = next(x for x in r.rules if x.rule_id == "amount_match")
    assert ar.status is RuleStatus.UNABLE


def test_token_match_is_a_fallback_never_the_primary_path():
    """A clean parse that already matches must PASS via the exact path and
    never even reach token-match - confirmed by evidence text NOT
    mentioning token-match for an ordinary clean cheque."""
    doc = azure_doc()  # the default fixture: clean, matching amount
    r = validate(to_normalized(doc), today=TODAY)
    ar = next(x for x in r.rules if x.rule_id == "amount_match")
    assert ar.status is RuleStatus.PASS
    assert "token-match" not in ar.evidence.lower()


def test_real_record_0010_stays_unable_not_falsely_rescued():
    """CHQ-20260824-0010's real text ('...-eight-one 4/kg 100 Dollars' for
    $1,981.46) is NOT rescued by token-match: 'eighty' is missing because
    the text says 'eight' - a real, different word - and the safety rule
    (real words are never fuzzy-collapsed into a different expected token)
    correctly refuses to treat that as a match. This is the deliberate
    trade-off named in the brief: safety over resolving one more record."""
    doc = azure_doc(WordAmount={"content": "one thousand nine hundred -eight-one 4/kg 100 Dollars"},
                    NumberAmount={"content": "$ 1,981.46"})
    r = validate(to_normalized(doc), today=TODAY)
    ar = next(x for x in r.rules if x.rule_id == "amount_match")
    assert ar.status is RuleStatus.UNABLE


def test_real_record_0020_stray_cents_digit_blocks_false_pass():
    """Regression: CHQ-20260824-0020 reads 'Five Hundred Ninety Dollars-6
    100 DOLLARS' against a $590.00 numeral. 'five'/'hundred'/'ninety' are
    all genuinely present, so a word-only token-match would call this a
    PASS - but the un-placed '6' beside '/100' is legible evidence of a
    real $590.06, which the numeral disagrees with (confirmed by direct
    visual inspection of the source image). Token-match can't weigh a bare
    digit on its own, so it must not confidently PASS here - a wrong
    confident approval is worse than staying UNABLE."""
    doc = azure_doc(WordAmount={"content": "Five Hundred Ninety Dollars-6 100 DOLLARS"},
                    NumberAmount={"content": "$590.00"})
    r = validate(to_normalized(doc), today=TODAY)
    ar = next(x for x in r.rules if x.rule_id == "amount_match")
    assert ar.status is not RuleStatus.PASS


def test_stray_digit_consistent_with_numeral_cents_does_not_block():
    """CHQ-20260824-0013's genuine '42' cents numerator sits outside any
    '/100' the parser recognised (no '100' substring at all in this raw
    text - the printed fraction line itself just wasn't OCR'd), but it
    agrees with the numeral's own cents - not a discrepancy, must not
    block the match the way an inconsistent stray digit does."""
    r = verify_amount_by_tokens("Five Hundred Thirty 42 DOLLARS", 530, numeral_cents=42)
    assert r.outcome == "matched"
    # sanity: a DIFFERENT stray value against the same text does block
    r2 = verify_amount_by_tokens("Five Hundred Thirty 42 DOLLARS", 530, numeral_cents=17)
    assert r2.outcome != "matched"


def test_expected_amount_token_forms_canonical_spelling():
    assert expected_amount_token_forms(933) == [["nine", "hundred", "thirty", "three"]]
    # below $1000 there's no alternate "compact hundreds" reading available
    assert expected_amount_token_forms(933) == [expected_amount_token_forms(933)[0]]
    forms = expected_amount_token_forms(1150)
    assert ["one", "thousand", "one", "hundred", "fifty"] in forms
    assert ["eleven", "hundred", "fifty"] in forms
    # $2000: both 'two thousand' and the less common but valid 'twenty
    # hundred' (same pattern as the everyday 'nineteen hundred' for 1900)
    forms2000 = expected_amount_token_forms(2000)
    assert ["two", "thousand"] in forms2000
    assert ["twenty", "hundred"] in forms2000