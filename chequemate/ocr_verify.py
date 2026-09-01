"""Field-level secondary OCR verification: Document Intelligence vs. TrOCR.

WHAT THIS IS NOT (see the module docstrings of geometry.py, crops.py,
trocr_adapter.py for the mechanics; this is the policy layer):

  - not a whole-cheque OCR engine - TrOCR only ever sees a crop already
    tied to one field DI itself located,
  - not an autonomous correction engine - it never overwrites a DI value,
  - not a majority-voting member - there are exactly two observations
    (DI, TrOCR) and the decision table below is transparent, not a vote,
  - not a participant in the deterministic pass/fail rules in rules.py -
    see the SAFE SOURCE POLICY section: every rule's input is untouched by
    anything in this module.

CONFIDENCE DISCIPLINE: TrOCR's `score` (when present at all) is raw
per-token generation likelihood from greedy decoding - `score_type` is
always "raw_generation_evidence", NEVER "confidence", because no
calibration procedure against held-out labeled cheque fields exists in
this repo. Report code must render this as "Not calibrated" evidence, not
a percentage. See models.SecondaryObservation's docstring.

TRIGGERING POLICY: one of "disabled", "always_on_eligible_fields",
"low_confidence_only", "handwritten_or_low_confidence". This repo has no
handwriting classifier and none is invented here (per the task's own
instruction) - "handwritten_or_low_confidence" is therefore currently
identical to "low_confidence_only"; the name is kept distinct only so a
future real handwriting signal can be wired in without a config rename.

SAFE SOURCE POLICY for rules.py inputs (section 15 of the assignment):
this module NEVER mutates a NormalizedCheque's existing Field objects.
`verify_cheque_fields()` is called AFTER `validate()` has already produced
its RuleResults from the untouched cheque - the comparison results are
attached to the report record as advisory data alongside the existing
verdict, never fed back into it. An OCR disagreement is real, reportable
information for a human reviewer; it is deliberately not wired into
`Config`/`rules.py`/`validate.py` at all, so "does this comparison change
the deterministic verdict" is not just handled correctly, it is
structurally impossible to get wrong by construction.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field as dc_field
from decimal import Decimal
from typing import Any

from PIL import Image

from .crops import FieldCropConfig, save_debug_crop, validate_and_generate_crop
from .geometry import convert_bounding_region_to_pixels
from .imageprep import NoChequeFound, OrientationIndeterminate, detect_rotation_only
from .models import (ComparisonResult, ComparisonStatus, CropResult, CropStatus,
                     FieldVerification, NormalizedCheque, ParseStatus,
                     PrimaryObservation, SecondaryObservation, TrOCRStatus)
from .normalize import _clean, canonical_payee, levenshtein, normalize_amount_words
from .rules import _distinctive_token
from .trocr_adapter import (DEFAULT_MODEL_NAME, TrOCRModelClient,
                            TransformersTrOCRClient, run_trocr)

DI_MODEL_ID = "prebuilt-check.us"

# field_name (this repo's NormalizedCheque attribute) -> Azure's own field name
FIELD_TO_AZURE_KEY = {
    "payee": "PayTo",
    "amount_words": "WordAmount",
    "memo": "Memo",
}
# Fields deliberately NEVER eligible, regardless of configuration - see the
# assignment's "initial verification scope": full MICR/cheque image, bank
# name/address, signature authenticity, logos, routing symbols and any
# field with no reliable region are excluded categorically, not by config,
# so a future config mistake cannot silently enable them.
NEVER_ELIGIBLE_FIELDS = frozenset({
    "signature", "bank_name", "bank_address", "micr", "full_image",
})

TRIGGER_POLICIES = ("disabled", "always_on_eligible_fields",
                   "low_confidence_only", "handwritten_or_low_confidence")


def _default_crop_configs() -> dict[str, FieldCropConfig]:
    return {
        "payee": FieldCropConfig(field_name="payee"),
        "amount_words": FieldCropConfig(field_name="amount_words",
                                        min_aspect_ratio=0.3, max_aspect_ratio=30.0),
        "memo": FieldCropConfig(field_name="memo"),
    }


def _default_confidence_thresholds() -> dict[str, float]:
    return {"payee": 0.7, "amount_words": 0.7, "memo": 0.7}


@dataclass(frozen=True)
class TrOCRVerificationConfig:
    """The single configuration surface for this feature - follows the
    project's existing pattern (a frozen dataclass, same as rules.Config)
    rather than introducing a second configuration mechanism."""

    enabled: bool = False
    model_name_or_path: str = DEFAULT_MODEL_NAME
    model_revision: str | None = None
    device_preference: str = "auto"                    # "auto" | "cpu" | "cuda"
    eligible_fields: tuple[str, ...] = ("payee", "amount_words", "memo")
    trigger_policy: str = "always_on_eligible_fields"
    field_confidence_thresholds: dict[str, float] = dc_field(
        default_factory=_default_confidence_thresholds)
    crop_configs: dict[str, FieldCropConfig] = dc_field(
        default_factory=_default_crop_configs)
    max_new_tokens: int = 32
    inference_timeout_s: float | None = 10.0
    debug_retain_crops: bool = False
    report_crop_visibility: bool = True
    local_files_only: bool = False   # True in production: never attempt a download
    expected_payee: str = "Town of Whitby"
    payee_edit_tolerance: int = 2

    def __post_init__(self):
        if self.trigger_policy not in TRIGGER_POLICIES:
            raise ValueError(f"trigger_policy must be one of {TRIGGER_POLICIES}, "
                             f"got {self.trigger_policy!r}")
        bad = set(self.eligible_fields) & NEVER_ELIGIBLE_FIELDS
        if bad:
            raise ValueError(f"these fields are never eligible for TrOCR "
                             f"verification: {sorted(bad)}")


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def config_from_env(base: TrOCRVerificationConfig | None = None) -> TrOCRVerificationConfig:
    """Reads CHEQUEMATE_TROCR_* environment variables, following this repo's
    existing convention (see run_batch.py's load_dotenv()) - call that
    first if .env file support is wanted; real env vars always win there,
    matching how AZURE_DI_ENDPOINT/AZURE_DI_KEY are already resolved."""
    base = base or TrOCRVerificationConfig()
    local_path = os.getenv("CHEQUEMATE_TROCR_LOCAL_MODEL_PATH")
    return TrOCRVerificationConfig(
        enabled=_env_bool("CHEQUEMATE_TROCR_ENABLED", base.enabled),
        model_name_or_path=local_path or os.getenv(
            "CHEQUEMATE_TROCR_MODEL", base.model_name_or_path),
        model_revision=os.getenv("CHEQUEMATE_TROCR_MODEL_REVISION", base.model_revision),
        device_preference=os.getenv("CHEQUEMATE_TROCR_DEVICE", base.device_preference),
        eligible_fields=base.eligible_fields,
        trigger_policy=os.getenv("CHEQUEMATE_TROCR_TRIGGER_POLICY", base.trigger_policy),
        field_confidence_thresholds=base.field_confidence_thresholds,
        crop_configs=base.crop_configs,
        max_new_tokens=int(os.getenv("CHEQUEMATE_TROCR_MAX_NEW_TOKENS",
                                     str(base.max_new_tokens))),
        inference_timeout_s=base.inference_timeout_s,
        debug_retain_crops=_env_bool("CHEQUEMATE_TROCR_DEBUG_RETAIN_CROPS",
                                     base.debug_retain_crops),
        report_crop_visibility=base.report_crop_visibility,
        local_files_only=_env_bool("CHEQUEMATE_TROCR_LOCAL_FILES_ONLY",
                                   base.local_files_only),
        expected_payee=base.expected_payee,
        payee_edit_tolerance=base.payee_edit_tolerance,
    )


# ---------------------------------------------------------------------------
# model client cache - "load once per process, not once per field" (section 21)
# ---------------------------------------------------------------------------

_client_cache: dict[tuple, TransformersTrOCRClient] = {}
_client_cache_lock = threading.Lock()


def get_or_create_client(cfg: TrOCRVerificationConfig) -> TransformersTrOCRClient:
    key = (cfg.model_name_or_path, cfg.model_revision, cfg.device_preference,
          cfg.local_files_only, cfg.max_new_tokens)
    with _client_cache_lock:
        client = _client_cache.get(key)
        if client is None:
            client = TransformersTrOCRClient(
                model_name_or_path=cfg.model_name_or_path,
                model_revision=cfg.model_revision,
                device_preference=cfg.device_preference,
                max_new_tokens=cfg.max_new_tokens,
                local_files_only=cfg.local_files_only,
                inference_timeout_s=cfg.inference_timeout_s)
            _client_cache[key] = client
        return client


# ---------------------------------------------------------------------------
# field-specific normalization (section 11)
# ---------------------------------------------------------------------------

# Fixed, literal label prefixes only - never fuzzy-stripped, for the same
# reason normalize.py's _MISSPELLINGS table is a fixed lookup: a field this
# consequential must never have words removed on a "looks close enough"
# basis.
_PAYEE_LABEL_PREFIXES = (
    "pay to the order of", "payable to the order of", "pay to the order",
    "pay to", "payable to",
)


def strip_known_payee_label(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in sorted(_PAYEE_LABEL_PREFIXES, key=len, reverse=True):
        if lowered.startswith(prefix):
            return stripped[len(prefix):].strip(" :-—")
    return stripped


def normalize_payee_for_comparison(raw_text: str | None) -> str | None:
    """Same canonical form rules.py's check_payee already compares against
    (case/punctuation/leading-article-insensitive) - reused, not
    reimplemented, plus a known-label-prefix strip TrOCR's raw line often
    needs that DI's own already-isolated PayTo field never does."""
    if not raw_text or not raw_text.strip():
        return None
    return canonical_payee(strip_known_payee_label(raw_text)) or None


def payee_matches_expected(normalized_text: str | None, expected_payee: str,
                           tolerance: int) -> bool:
    """Identical predicate to rules.check_payee's own match test (same
    distinctive-token + edit-distance logic) - both observations are
    checked against the SAME rule, not two different ones."""
    if not normalized_text:
        return False
    target = _distinctive_token(expected_payee)
    tokens = normalized_text.split()
    if not tokens:
        return False
    return min(levenshtein(tok, target) for tok in tokens) <= tolerance


def normalize_amount_words_for_comparison(raw_text: str | None
                                          ) -> tuple[Decimal | None, bool]:
    """Reuses normalize.normalize_amount_words verbatim - same Decimal
    parser, same refusal-over-guessing behaviour, for both DI and TrOCR
    text. Returns (value, is_determined)."""
    result = normalize_amount_words(raw_text)
    if result.parse_status is ParseStatus.OK:
        return result.value, True
    return None, False


def normalize_memo_for_comparison(raw_text: str | None) -> str | None:
    """No roll-number character set/pattern is defined anywhere in this
    repository (rules.py's require_memo only checks presence) - per the
    task's own instruction not to invent one, this is whitespace/Unicode
    normalization only, no autocorrection, no format inference."""
    if not raw_text or not raw_text.strip():
        return None
    return _clean(raw_text).upper() or None


_NORMALIZERS = {
    "payee": normalize_payee_for_comparison,
    "memo": normalize_memo_for_comparison,
}


# ---------------------------------------------------------------------------
# comparison / decision engine (section 12)
# ---------------------------------------------------------------------------

def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b), 1)
    return 1.0 - (levenshtein(a, b) / longest)


def compare_field(
    *, field_name: str,
    di_ok: bool, di_raw: str | None, di_normalized: Any,
    crop_status: CropStatus,
    trocr_status: TrOCRStatus, trocr_raw: str | None, trocr_normalized: Any,
    allowlist_expected: str | None = None, allowlist_tolerance: int = 2,
) -> ComparisonResult:
    """Transparent, non-voting comparison. Priority-ordered so every one of
    the ten ComparisonStatus values is reachable and mutually exclusive -
    see this module's docstring for the reasoning behind where each
    boundary falls (some of the ten labels inherently overlap in the
    assignment's own wording; this is the documented, deliberate partition
    chosen to make all ten distinct and defensible):

      1. an explicit crop rejection is always reported as CROP_REJECTED,
         even if DI also has nothing - it is the most specific, most
         actionable diagnostic available.
      2. a crop that was never even producible (no polygon /
         conversion failure) is TROCR_NOT_RUN, regardless of DI's status -
         distinct from "not eligible at all" (see case 4).
      3. a genuine model error is TROCR_FAILED, regardless of DI's status.
      4. TrOCR simply not attempted (disabled / not eligible / triggering
         policy excluded it) or attempted-but-empty maps to DI_ONLY when DI
         has a usable value, BOTH_UNDETERMINED when it does not - "only one
         side produced anything" vs. "neither side produced anything".
      5. both sides produced usable text: DI_ONLY is impossible here by
         construction; the result is one of TROCR_ONLY (DI empty),
         AGREE_EXACT / AGREE_NORMALIZED / AGREE_ALLOWLIST / DISAGREE
         (DI usable).
    """
    reason_codes: list[str] = []

    if crop_status is CropStatus.REJECTED:
        return ComparisonResult(
            status=ComparisonStatus.CROP_REJECTED,
            selected_display_value=di_raw, selection_source="document_intelligence",
            manual_review_required=not di_ok,
            reason_codes=["TROCR_UNAVAILABLE"] if di_ok else
                        ["TROCR_UNAVAILABLE", "BOTH_UNDETERMINED"])

    if crop_status is CropStatus.NOT_RUN and trocr_status is TrOCRStatus.NOT_RUN:
        return ComparisonResult(
            status=ComparisonStatus.TROCR_NOT_RUN,
            selected_display_value=di_raw, selection_source="document_intelligence",
            manual_review_required=not di_ok,
            reason_codes=["TROCR_UNAVAILABLE"] if di_ok else
                        ["TROCR_UNAVAILABLE", "BOTH_UNDETERMINED"])

    if trocr_status is TrOCRStatus.FAILED:
        return ComparisonResult(
            status=ComparisonStatus.TROCR_FAILED,
            selected_display_value=di_raw, selection_source="document_intelligence",
            manual_review_required=not di_ok,
            reason_codes=["TROCR_UNAVAILABLE"] if di_ok else
                        ["TROCR_UNAVAILABLE", "BOTH_UNDETERMINED"])

    trocr_usable = (trocr_status is TrOCRStatus.COMPLETED
                    and trocr_raw is not None and trocr_raw.strip() != "")

    if not trocr_usable:
        # covers trocr_status in {SKIPPED, UNDETERMINED}
        status = ComparisonStatus.DI_ONLY if di_ok else ComparisonStatus.BOTH_UNDETERMINED
        return ComparisonResult(
            status=status, selected_display_value=di_raw,
            selection_source="document_intelligence",
            manual_review_required=(status is ComparisonStatus.BOTH_UNDETERMINED),
            reason_codes=[] if status is ComparisonStatus.DI_ONLY else ["BOTH_UNDETERMINED"])

    if not di_ok:
        # Policy D: unverified suggestion only, never promoted.
        return ComparisonResult(
            status=ComparisonStatus.TROCR_ONLY,
            selected_display_value=di_raw,  # stays DI's (empty) value - never auto-promoted
            selection_source="document_intelligence",
            manual_review_required=True,
            reason_codes=["TROCR_UNVERIFIED_SUGGESTION"])

    # both sides produced a usable value - the only branch that can agree/disagree.
    raw_exact = (di_raw or "").strip() == (trocr_raw or "").strip()
    normalized_equal = di_normalized is not None and di_normalized == trocr_normalized
    sim = _similarity(str(di_raw or ""), str(trocr_raw or ""))

    if raw_exact:
        return ComparisonResult(
            status=ComparisonStatus.AGREE_EXACT, similarity=1.0,
            selected_display_value=di_raw, selection_source="document_intelligence",
            manual_review_required=False, reason_codes=[])
    if normalized_equal:
        return ComparisonResult(
            status=ComparisonStatus.AGREE_NORMALIZED, similarity=sim,
            selected_display_value=di_raw, selection_source="document_intelligence",
            manual_review_required=False, reason_codes=[])

    if allowlist_expected is not None:
        di_hit = payee_matches_expected(di_normalized, allowlist_expected, allowlist_tolerance)
        trocr_hit = payee_matches_expected(trocr_normalized, allowlist_expected,
                                           allowlist_tolerance)
        if di_hit and trocr_hit:
            return ComparisonResult(
                status=ComparisonStatus.AGREE_ALLOWLIST, similarity=sim,
                selected_display_value=di_raw, selection_source="document_intelligence",
                manual_review_required=False,
                # Spelling differences are recorded, never hidden, even
                # though the field itself is not flagged for review.
                reason_codes=["ALLOWLIST_SPELLING_DIFFERENCE"])

    return ComparisonResult(
        status=ComparisonStatus.DISAGREE, similarity=sim,
        selected_display_value=di_raw, selection_source="document_intelligence",
        manual_review_required=True, reason_codes=["OCR_DISAGREEMENT"])


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def _should_trigger(field_name: str, di_confidence: float | None,
                   cfg: TrOCRVerificationConfig) -> bool:
    if cfg.trigger_policy == "disabled":
        return False
    if cfg.trigger_policy == "always_on_eligible_fields":
        return True
    # "low_confidence_only" and "handwritten_or_low_confidence" (no
    # handwriting classifier exists in this repo - see module docstring)
    threshold = cfg.field_confidence_thresholds.get(field_name, 0.7)
    return di_confidence is None or di_confidence < threshold


def _get_field(cheque: NormalizedCheque, field_name: str):
    return getattr(cheque, field_name)


def resolve_rotation(source_path, image: Image.Image,
                     rotation_override: str | None = None
                     ) -> tuple[str | None, str | None]:
    """Learn whether the whole cheque needs a CW/CCW rotation correction
    before any field crop is taken from it - see crops.py's module
    docstring for why this matters (this repo's entire real corpus is
    stored portrait, un-rotated, in the same pixel space DI's polygons are
    defined in).

    `rotation_override` mirrors imageprep.prepare_cheque_image()'s own
    `rotation_override` parameter: an operator-confirmed direction for a
    file the detector already refused on (see
    scripts/apply_rotation_override.py). When given, it is used directly -
    the detector is never re-run and never re-refuses a file a human has
    already resolved. This is the SAME override, not a second one: pass
    `record["rotation"]["direction"]` when `record["rotation"]["source"] ==
    "operator_override"`.

    Returns (direction, refusal_reason):
      - (None, None): no correction needed (landscape image, or
        `source_path` wasn't supplied - the pre-fix, geometry-as-is
        behaviour, kept for callers/tests that don't pass a path).
      - ("CW"|"CCW", None): correction needed and confidently determined
        (by the detector, or supplied directly via `rotation_override`).
      - (None, "..."): the image is portrait (so DI's polygons ARE in a
        rotated space) but imageprep's own detector could not confidently
        pick a direction (or found no cheque at all) - callers must NOT
        fall back to raw, unrotated geometry in this case (that would
        validate/crop sideways text with no warning), so every field for
        this cheque should be marked NOT_RUN with this reason instead.
    """
    if rotation_override is not None:
        if rotation_override not in ("CW", "CCW"):
            raise ValueError(f"rotation_override must be 'CW' or 'CCW', "
                             f"got {rotation_override!r}")
        return rotation_override, None
    if source_path is None:
        return None, None
    if image.width >= image.height:
        return None, None  # already landscape - imageprep's detector assumes portrait input
    try:
        decision = detect_rotation_only(source_path)
    except (NoChequeFound, OrientationIndeterminate) as exc:
        return None, (f"cheque appears rotated but orientation could not be "
                      f"confirmed ({type(exc).__name__}: {exc}) - refusing to "
                      f"crop from unrotated geometry rather than guess")
    return decision.direction, None


def verify_cheque_fields(
    raw_response: dict,
    image: Image.Image,
    cheque: NormalizedCheque,
    cfg: TrOCRVerificationConfig,
    *, record_id: str = "unaudited",
    trocr_client: TrOCRModelClient | None = None,
    debug_dir=None,
    source_path=None,
    rotation_override: str | None = None,
) -> dict[str, FieldVerification]:
    """Run field-level TrOCR verification for every configured eligible
    field. Returns {} untouched when `cfg.enabled` is False - this is the
    entire rollback mechanism: existing extraction/report behaviour is
    unaffected because nothing downstream ever sees a non-empty dict.

    `raw_response` is the FULL Azure response dict (needs `pages[]` for
    width/height/unit - a single field's boundingRegions never carries
    that). `image` is the exact image DI analyzed (never a preprocessed/
    resized copy unless page dimensions are re-derived from it too).

    `source_path`, when given, lets this function correct for whole-cheque
    rotation (see resolve_rotation()) before validating/cropping any field.
    Omit it (the default) to keep the pre-fix behaviour of treating DI's
    polygon coordinates as already upright - existing callers/tests that
    don't care about rotation keep working unchanged.

    `rotation_override`, when given, is used instead of running the
    detector at all - pass a record's already-stored
    `rotation["direction"]` when `rotation["source"] == "operator_override"`
    so a file a human has already confirmed is never re-refused by a fresh
    detector call (see resolve_rotation()'s docstring).
    """
    if not cfg.enabled or cfg.trigger_policy == "disabled":
        return {}

    document = (raw_response.get("documents") or [{}])[0]
    fields = document.get("fields", {})
    pages = raw_response.get("pages", [])

    rotation_direction, rotation_refusal = resolve_rotation(
        source_path, image, rotation_override=rotation_override)

    results: dict[str, FieldVerification] = {}
    for field_name in cfg.eligible_fields:
        di_field = _get_field(cheque, field_name)
        di_ok = di_field.ok
        di_raw = di_field.raw_text
        di_confidence = di_field.confidence

        if field_name == "amount_words":
            di_normalized, _ = normalize_amount_words_for_comparison(di_raw)
        else:
            di_normalized = _NORMALIZERS[field_name](di_raw)

        azure_key = FIELD_TO_AZURE_KEY[field_name]
        azure_field = fields.get(azure_key, {}) or {}

        primary = PrimaryObservation(
            engine="azure_document_intelligence", raw_value=di_raw,
            normalized_value=(str(di_normalized) if di_normalized is not None else None),
            confidence=di_confidence,
            polygon=(azure_field.get("boundingRegions")
                    or azure_field.get("bounding_regions")),
            page_number=1, model_id=DI_MODEL_ID, model_version=None)

        if not _should_trigger(field_name, di_confidence, cfg):
            crop = CropResult(status=CropStatus.NOT_RUN,
                              validation_reasons=["triggering policy "
                                                  f"{cfg.trigger_policy!r} did not "
                                                  "select this field"])
            secondary = SecondaryObservation(status=TrOCRStatus.SKIPPED)
        elif rotation_refusal is not None:
            crop = CropResult(status=CropStatus.NOT_RUN,
                              validation_reasons=[rotation_refusal])
            secondary = SecondaryObservation(status=TrOCRStatus.NOT_RUN)
        else:
            bounding_regions = (azure_field.get("boundingRegions")
                               or azure_field.get("bounding_regions"))
            conversion = convert_bounding_region_to_pixels(
                bounding_regions, image_width_px=image.width,
                image_height_px=image.height, pages=pages, expected_page_number=1)
            crop_cfg = cfg.crop_configs.get(field_name, FieldCropConfig(field_name))
            crop, variants = validate_and_generate_crop(
                image, conversion, crop_cfg, rotation=rotation_direction)

            if variants is None:
                secondary = SecondaryObservation(status=TrOCRStatus.NOT_RUN)
            else:
                client = trocr_client or get_or_create_client(cfg)
                variant_name = crop_cfg.preprocessing_variants[0]
                variant_image = variants[variant_name]
                if cfg.debug_retain_crops and debug_dir is not None:
                    crop.image_reference = save_debug_crop(
                        variant_image, field_name, record_id, debug_dir)
                secondary = run_trocr(variant_image, field_name, variant_name, client)

        if field_name == "amount_words":
            trocr_normalized, _ = normalize_amount_words_for_comparison(secondary.raw_value)
        else:
            trocr_normalized = _NORMALIZERS[field_name](secondary.raw_value)
        secondary.normalized_value = (str(trocr_normalized)
                                      if trocr_normalized is not None else None)

        allowlist_expected = cfg.expected_payee if field_name == "payee" else None
        comparison = compare_field(
            field_name=field_name, di_ok=di_ok, di_raw=di_raw,
            di_normalized=di_normalized, crop_status=crop.status,
            trocr_status=secondary.status, trocr_raw=secondary.raw_value,
            trocr_normalized=trocr_normalized,
            allowlist_expected=allowlist_expected,
            allowlist_tolerance=cfg.payee_edit_tolerance)

        results[field_name] = FieldVerification(
            field_name=field_name, primary=primary, crop=crop,
            secondary=secondary, comparison=comparison)

    return results
