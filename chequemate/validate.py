"""Run the four rules over a NormalizedCheque and produce a verdict."""

from __future__ import annotations

from datetime import date

from .models import NormalizedCheque, RuleStatus, ValidationResult, Verdict
from .rules import (
    Config,
    check_amounts_match,
    check_date,
    check_memo,
    check_payee,
    check_signature,
)

RULE_SET_VERSION = "1.10.0"
# 1.10.0: six-digit dates get the same "generate every interpretation,
# converge or refuse" discipline as 1.9.0's amount token-match (see
# normalize._resolve_six_digit_date's module note). A 6-digit CheckDate
# read admits two independently-motivated hypotheses - a genuine short
# year (DDMMYY/MMDDYY, the module's original heuristic) and a truncated
# CPA Standard 006 year-leading grid missing its century+decade prefix -
# both using the SAME existing 2-digit-year pivot convention, no new
# plausibility judgment. When they agree, the date resolves (with its
# repaired provenance visible in the report, same as amount_match's
# corroboration/token-match notes); when they genuinely disagree (a real
# record in this batch: 2026-08-30 under one hypothesis, 2030-08-26 under
# the other, from the identical 6 digits), the honest answer is UNABLE,
# never a confident pick of either one - this is the fix for the class of
# bug record CHQ-20260824-0012 exposed (a silent wrong year), not a
# patch for that one record. 7-digit reads get no new repair: checked
# exhaustively, every one of the 10 possible values for the single
# missing digit produces an equally structurally-valid date, so there is
# no way to narrow it down without preferring a century as more likely -
# exactly the judgment this ruleset refuses to make. Records where the
# raw value is contaminated by another field (a cheque number bleeding
# into the date box) or a drawer's own correction mark are not form
# properties and are not engineered around; they stay UNABLE. Stale-dated
# and post-dated logic in check_date is untouched - recency is Config's
# job (max_age_months, reject_postdated), never the parser's.
# 1.9.0: token-match fallback for amount_match (see
# normalize.verify_amount_by_tokens and check_amounts_match's docstring for
# the mechanism and full decision order). When the exact word-amount parse
# fails to produce a matching value, derive the number words the numeral
# implies and search the written-amount text for them by fuzzy presence,
# ignoring everything else - then guard against the "$933 altered to look
# like $9,330" trap by confirming no scale word/claim is stated that the
# numeral doesn't permit. All expected tokens present + no contradicting
# claim -> PASS (resolved-by-token-match). A contradicting claim -> FAIL,
# on its own evidence. Tokens missing -> no new evidence, UNABLE as before
# (the 1.8.0 degraded-never-promotes-to-FAIL invariant is preserved for
# that path). Strictly a fallback: a clean parse that already matches never
# reaches this at all.
# 1.7.0: (1) check_payee no longer routes bank/branch-letterhead-looking
# extracted text to UNABLE (_BANK_KEYWORDS removed) - presence of the
# distinctive payee token is now the only thing that matters; text with no
# such token is a plain FAIL like any other wrong payee. (2)
# normalize_amount_words discards the printed '.../100' cents-suffix
# construct outright instead of trying to parse a possibly-garbled
# numerator from it, taking cents from the numeric amount field's own
# (far more reliably OCR'd) cents digits instead - see its docstring.
#
# 1.8.0: corroboration over failure-counting for amount_match, the one rule
# with a genuine second independent source (the numeral and the written
# amount both state the same fact - see check_amounts_match's docstring).
# Two sources converging is now itself the PASS condition, even when one
# side needed the 1.7.0 cents-suffix repair to get there (Field.degraded);
# two sources genuinely disagreeing is still FAIL, UNLESS one side is
# degraded, in which case the disagreement is insufficient evidence rather
# than proof of a wrong cheque, and resolves to UNABLE - never FAIL. That
# asymmetry (degraded readings can promote UNABLE->PASS via corroboration,
# never UNABLE->FAIL) is a general invariant enforced in Field.degraded's
# docstring and check_amounts_match, not a one-off fix for the record that
# surfaced it. date/payee/signature/memo were evaluated for a second
# source and don't have a validated one: date's only candidate (Azure's
# own `valueDate`) was checked against real unparseable dates and found
# unreliable (see the 1.7.0 investigation - not implemented, on purpose);
# payee/signature/memo have exactly one field each, nothing to corroborate
# against. validate()'s own three-way collapse below is UNCHANGED by this -
# FAIL still always wins, UNABLE still always means REVIEW - what changed
# is what feeds into it: UNABLE now means "genuinely insufficient evidence
# across every source this cheque states the fact through", not "one field
# happened to need repair."


def validate(cheque: NormalizedCheque, cfg: Config | None = None,
             today: date | None = None) -> ValidationResult:
    cfg = cfg or Config()
    results = [
        check_payee(cheque, cfg),
        check_amounts_match(cheque, cfg),
        check_signature(cheque, cfg),
        check_date(cheque, cfg, today=today),
        check_memo(cheque, cfg),
    ]
    # Three-valued verdict, deliberately not collapsed back to two: UNABLE
    # means "couldn't tell" (unreadable field, extraction gap, indeterminate
    # image prep) - that's not evidence the cheque is wrong, but it is NOT
    # evidence it's fine either, and reporting it as VALID (the previous
    # behaviour) means every cheque a rule couldn't assess gets waved
    # through unreviewed. A genuine FAIL (something we could read and know
    # is wrong) still invalidates outright - FAIL always wins over UNABLE.
    # Everything else, with no UNABLE anywhere, is a clean VALID.
    if any(r.status is RuleStatus.FAIL for r in results):
        verdict = Verdict.INVALID
    elif any(r.status is RuleStatus.UNABLE for r in results):
        verdict = Verdict.REVIEW
    else:
        verdict = Verdict.VALID
    return ValidationResult(verdict=verdict, rules=results, cheque=cheque)