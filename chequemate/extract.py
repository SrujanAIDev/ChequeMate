"""Azure Document Intelligence adapter.

Isolates every Azure-specific detail. Swap this module to change extractor
without touching normalisation or rules.
"""

from __future__ import annotations

from datetime import date as _date
from decimal import Decimal
from typing import Any

from .models import Field, NormalizedCheque, ParseStatus
from .normalize import (
    normalize_amount_numeric,
    normalize_amount_words,
    normalize_date,
    normalize_payee,
    normalize_signature,
)

MODEL_ID = "prebuilt-check.us"

# prebuilt-check.us field names -> our field names
FIELD_MAP = {
    "PayTo": "payee",
    "NumberAmount": "amount_numeric",
    "WordAmount": "amount_words",
    "CheckDate": "cheque_date",
    "PayerSignatures": "signature",
    "Memo": "memo",
}


# check_memo's note-matching discriminator (ruleset 1.6.0+): distinguishes
# "Azure never returned a Memo field at all" from "Azure returned the
# field but it had no usable text" - both are ParseStatus.ABSENT, and per
# the 1.6.0 change neither is a confident FAIL any more (see check_memo's
# docstring for why "field present but blank" isn't trusted as a genuine
# human-confirmed-blank read either), but the distinction is kept in
# `note` for a reviewer's own audit context even though both now receive
# identical RuleStatus treatment.
MEMO_KEY_ABSENT_NOTE = "Memo field not returned by extractor at all"


def normalize_memo(text: str | None, confidence: float | None = None,
                   bbox: list | None = None, field_present: bool = True) -> Field:
    """Blank/absent memo is ABSENT. `field_present=False` means Azure's
    response never contained a Memo key at all (see MEMO_KEY_ABSENT_NOTE) -
    `field_present=True` with no usable text means the key was present but
    empty. check_memo (ruleset 1.6.0+) treats both as UNABLE, not FAIL."""
    if not text or not text.strip():
        note = None if field_present else MEMO_KEY_ABSENT_NOTE
        return Field("memo", parse_status=ParseStatus.ABSENT, raw_text=text,
                     confidence=confidence, bbox=bbox, note=note)
    return Field("memo", value=text.strip(), raw_text=text,
                 parse_status=ParseStatus.OK, confidence=confidence, bbox=bbox)


def _field_dict(field: Any) -> dict:
    if isinstance(field, dict):
        return field
    if hasattr(field, "as_dict"):
        return field.as_dict()
    return {}


def _text(d: dict) -> str | None:
    for key in ("valueString", "value_string", "content", "valueDate",
                "value_date"):
        v = d.get(key)
        if isinstance(v, str) and v.strip():
            return v
    v = d.get("value")
    return v if isinstance(v, str) and v.strip() else None


def _meta(d: dict) -> dict:
    return {
        "confidence": d.get("confidence"),
        "bbox": (d.get("boundingRegions") or d.get("bounding_regions")),
    }


def _signature_detected(d: dict | None) -> bool | None:
    """Return True/False if Azure gave a verdict, None if it gave nothing.

    prebuilt-check.us reports signature fields via a signature-typed value,
    NOT via `.value`. Reading `.value` alone yields None even when a
    signature was found — check every shape.
    """
    if not d:
        return None
    for key in ("valueSignature", "value_signature", "valueSelectionMark"):
        v = d.get(key)
        if isinstance(v, str):
            return v.lower() in ("signed", "selected", "present")
    for key in ("value", "valueBoolean", "value_boolean"):
        v = d.get(key)
        if isinstance(v, bool):
            return v
    # An array of detected signature regions is itself the verdict.
    arr = d.get("valueArray") or d.get("value_array")
    if isinstance(arr, list):
        return len(arr) > 0
    if d.get("boundingRegions") or d.get("bounding_regions"):
        return True
    return None


def amount_numeric_cents(field: Field) -> int | None:
    """The cents component of an already-normalised numeric-amount Field,
    or None if it never parsed. `value` is always constructed with exactly
    2 decimal places (see normalize_amount_numeric), so this is exact -
    never float rounding. Shared by extract.to_normalized and the
    raw_values-rebuilding scripts (apply_visual_verification.py,
    revalidate.py) so amount_words's numeric-cents fallback (ruleset 1.7.0)
    stays consistent between a fresh Azure call and an offline rebuild."""
    if not field.ok or field.value is None:
        return None
    return int((field.value * 100) % 100)


def to_normalized(document: Any, source_id: str | None = None,
                  date_convention: str = "DMY") -> NormalizedCheque:
    """`document` is one entry from `result.documents`, or a plain dict."""
    fields = document.get("fields", {}) if isinstance(document, dict) \
        else getattr(document, "fields", {}) or {}
    raw = {k: _field_dict(v) for k, v in fields.items()}

    def get(azure_name: str) -> dict:
        return raw.get(azure_name, {})

    sig_raw = get("PayerSignatures")

    amount_numeric = normalize_amount_numeric(
        _text(get("NumberAmount")), **_meta(get("NumberAmount")))

    return NormalizedCheque(
        payee=normalize_payee(_text(get("PayTo")), **_meta(get("PayTo"))),
        amount_numeric=amount_numeric,
        amount_words=normalize_amount_words(
            _text(get("WordAmount")),
            numeric_cents=amount_numeric_cents(amount_numeric),
            **_meta(get("WordAmount"))),
        cheque_date=normalize_date(
            _text(get("CheckDate")), prefer=date_convention,
            **_meta(get("CheckDate"))),
        signature=normalize_signature(
            _text(sig_raw), detected=_signature_detected(sig_raw or None),
            **_meta(sig_raw)),
        memo=normalize_memo(_text(get("Memo")), **_meta(get("Memo")),
                           field_present=bool(get("Memo"))),
        source_id=source_id,
        raw_response=raw,
    )


def analyze_raw(path: str, endpoint: str, key: str) -> dict:
    """Call Azure DI and return the FULL serialised response.

    Capture this once per cheque and replay from disk afterwards — every
    later change to normalisation or rules can then be tested offline,
    with no further Azure calls and no re-scanning.
    """
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    client = DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(key))
    with open(path, "rb") as fh:
        print("STEP 1")
        poller = client.begin_analyze_document(MODEL_ID, body=fh)
        print("STEP 2")

    print("STEP 3")
    result = poller.result()
    print("STEP 4")
    return result.as_dict() if hasattr(result, "as_dict") else dict(result)


def first_document(raw: dict) -> dict:
    docs = raw.get("documents") or []
    if not docs:
        raise ValueError("no cheque detected in response")
    return docs[0]


def analyze(path: str, endpoint: str, key: str,
            date_convention: str = "DMY") -> NormalizedCheque:
    """Call Azure DI on one cheque image."""
    raw = analyze_raw(path, endpoint, key)
    return to_normalized(first_document(raw), source_id=path,
                         date_convention=date_convention)


# ---------------------------------------------------------------------------
# repeat-extraction reconciliation
#
# A real experiment (re-sending imageprep-corrected images to DI) confirmed
# Azure's own extraction is not perfectly deterministic run-to-run on
# unchanged input: WordAmount, Memo, and PayerSignatures all showed cases
# where a second read of an equally-good (or better) image returned LESS
# than a prior read had. Calling DI multiple times per cheque and
# reconciling surfaces that instability as an honest AMBIGUOUS/UNABLE
# signal instead of silently trusting whichever single call happened to
# run. Off by default (runs=1 behaves exactly as before this existed).
# ---------------------------------------------------------------------------

def analyze_raw_multi(path: str, endpoint: str, key: str, runs: int = 3) -> list[dict]:
    """Call Azure DI on the SAME image `runs` times. Each call is a full,
    independent Azure request - this is not free, use deliberately."""
    return [analyze_raw(path, endpoint, key) for _ in range(runs)]


def _value_repr(value: Any) -> Any:
    """A hashable, JSON-safe form for comparing/storing typed field values."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, _date):
        return value.isoformat()
    return value


def _reading_dict(field: Field) -> dict:
    return {
        "raw_text": field.raw_text,
        "value": _value_repr(field.value) if field.ok else None,
        "parse_status": field.parse_status.value,
    }


def _reconcile_generic_field(name: str, fields: list[Field]) -> Field:
    """Unanimous agreement (identical parse_status + value across every
    run) keeps that reading untouched. Any disagreement - including one
    run returning a field and another not - becomes AMBIGUOUS with
    `value=None` (never a fabricated best guess) and every run's actual
    reading preserved in `alternate_readings`. rules.py routes
    AMBIGUOUS-with-no-value straight to UNABLE for every field (see each
    check_*'s guard) - the one existing exception, check_date's own
    DMY/MDY AMBIGUOUS, always carries a real `value`, so it is untouched
    by this and keeps its existing PASS/FAIL-with-disclosure behaviour."""
    signatures = {(f.parse_status, _value_repr(f.value) if f.ok else None) for f in fields}
    if len(signatures) == 1:
        return fields[0]
    readings = [_reading_dict(f) for f in fields]
    return Field(
        name, value=None, raw_text=fields[0].raw_text,
        parse_status=ParseStatus.AMBIGUOUS,
        note=f"{len(fields)} repeat-extraction runs disagreed on this field",
        confidence=fields[0].confidence, bbox=fields[0].bbox,
        alternate_readings=readings)


def _reconcile_amount_words(fields: list[Field]) -> Field:
    """Special-cased per the task that introduced this: if ANY run
    produces a parseable amount, prefer it over runs that failed to parse
    - a parse failure on one run is not evidence against a genuine parse
    on another. Only actual disagreement between two or more DIFFERENT
    parsed values is treated as ambiguous."""
    ok_fields = [f for f in fields if f.parse_status is ParseStatus.OK]
    readings = [_reading_dict(f) for f in fields]

    if not ok_fields:
        return Field(
            "amount_words", value=None, raw_text=fields[0].raw_text,
            parse_status=fields[0].parse_status,
            note=(f"no run out of {len(fields)} produced a parseable amount"
                 + (f" ({fields[0].note})" if fields[0].note else "")),
            confidence=fields[0].confidence, bbox=fields[0].bbox,
            alternate_readings=readings)

    distinct_values = {_value_repr(f.value) for f in ok_fields}
    if len(distinct_values) == 1:
        return ok_fields[0]

    return Field(
        "amount_words", value=None, raw_text=fields[0].raw_text,
        parse_status=ParseStatus.AMBIGUOUS,
        note=(f"repeat-extraction runs produced different parseable amounts: "
             f"{sorted(distinct_values)}"),
        confidence=fields[0].confidence, bbox=fields[0].bbox,
        alternate_readings=readings)


def reconcile_cheques(cheques: list[NormalizedCheque]) -> NormalizedCheque:
    """Combine N independent to_normalized() results (from N Azure calls on
    the SAME image) into one reconciled NormalizedCheque. `raw_response`
    is kept from the first run only (diagnostic reference; all N raw
    responses are the caller's to keep separately if needed)."""
    if not cheques:
        raise ValueError("reconcile_cheques() needs at least one NormalizedCheque")
    first = cheques[0]
    return NormalizedCheque(
        payee=_reconcile_generic_field("payee", [c.payee for c in cheques]),
        amount_numeric=_reconcile_generic_field(
            "amount_numeric", [c.amount_numeric for c in cheques]),
        amount_words=_reconcile_amount_words([c.amount_words for c in cheques]),
        cheque_date=_reconcile_generic_field(
            "cheque_date", [c.cheque_date for c in cheques]),
        signature=_reconcile_generic_field("signature", [c.signature for c in cheques]),
        memo=_reconcile_generic_field("memo", [c.memo for c in cheques]),
        source_id=first.source_id,
        raw_response=first.raw_response,
    )


def to_normalized_multi(documents: list[Any], source_id: str | None = None,
                        date_convention: str = "DMY") -> NormalizedCheque:
    """to_normalized() for each of N documents (from N Azure calls on the
    same image), reconciled via reconcile_cheques()."""
    cheques = [to_normalized(doc, source_id=source_id, date_convention=date_convention)
              for doc in documents]
    return reconcile_cheques(cheques)


def analyze_multi(path: str, endpoint: str, key: str, runs: int = 3,
                  date_convention: str = "DMY") -> NormalizedCheque:
    """Call Azure DI `runs` times on one cheque image and reconcile."""
    raw_responses = analyze_raw_multi(path, endpoint, key, runs=runs)
    documents = [first_document(r) for r in raw_responses]
    return to_normalized_multi(documents, source_id=path, date_convention=date_convention)


# --- signature diagnostics --------------------------------------------------

SIGNATURE_KEYS = (
    "type", "valueSignature", "value_signature", "valueSelectionMark",
    "value", "valueBoolean", "valueArray", "content", "confidence",
    "boundingRegions", "bounding_regions",
)


def diagnose_signature(document: dict) -> str:
    """Show exactly what Azure sent for PayerSignatures, key by key.

    Answers the open question: is the signature miss a model failure or a
    client-side read of the wrong attribute?
    """
    fields = document.get("fields", {})
    d = fields.get("PayerSignatures")
    if d is None:
        present = ", ".join(sorted(fields)) or "(none)"
        return ("PayerSignatures: KEY ABSENT from response\n"
                f"  fields returned: {present}")

    lines = ["PayerSignatures: key present"]
    for k in SIGNATURE_KEYS:
        if k in d:
            v = d[k]
            if k in ("boundingRegions", "bounding_regions") and isinstance(v, list):
                v = f"<{len(v)} region(s)>"
            elif k == "valueArray" and isinstance(v, list):
                v = f"<{len(v)} entry(ies)>"
            lines.append(f"  {k} = {v!r}")
    unexpected = sorted(set(d) - set(SIGNATURE_KEYS))
    if unexpected:
        lines.append(f"  other keys: {unexpected}")
    lines.append(f"  -> detected = {_signature_detected(d)}")
    return "\n".join(lines)