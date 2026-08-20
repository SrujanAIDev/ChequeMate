"""Azure Document Intelligence adapter.

Isolates every Azure-specific detail. Swap this module to change extractor
without touching normalisation or rules.
"""

from __future__ import annotations

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


def normalize_memo(text: str | None, confidence: float | None = None,
                   bbox: list | None = None) -> Field:
    """Blank/absent memo is a real ABSENT — check_memo turns that into a FAIL."""
    if not text or not text.strip():
        return Field("memo", parse_status=ParseStatus.ABSENT, raw_text=text,
                     confidence=confidence, bbox=bbox)
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


def to_normalized(document: Any, source_id: str | None = None,
                  date_convention: str = "DMY") -> NormalizedCheque:
    """`document` is one entry from `result.documents`, or a plain dict."""
    fields = document.get("fields", {}) if isinstance(document, dict) \
        else getattr(document, "fields", {}) or {}
    raw = {k: _field_dict(v) for k, v in fields.items()}

    def get(azure_name: str) -> dict:
        return raw.get(azure_name, {})

    sig_raw = get("PayerSignatures")

    return NormalizedCheque(
        payee=normalize_payee(_text(get("PayTo")), **_meta(get("PayTo"))),
        amount_numeric=normalize_amount_numeric(
            _text(get("NumberAmount")), **_meta(get("NumberAmount"))),
        amount_words=normalize_amount_words(
            _text(get("WordAmount")), **_meta(get("WordAmount"))),
        cheque_date=normalize_date(
            _text(get("CheckDate")), prefer=date_convention,
            **_meta(get("CheckDate"))),
        signature=normalize_signature(
            _text(sig_raw), detected=_signature_detected(sig_raw or None),
            **_meta(sig_raw)),
        memo=normalize_memo(_text(get("Memo")), **_meta(get("Memo"))),
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