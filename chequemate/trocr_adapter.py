"""Local TrOCR inference adapter.

TrOCR runs LOCALLY ONLY. This module never sends a cheque crop to Hugging
Face's hosted inference API or any other network endpoint - `from_pretrained`
is used purely to load model weights (once, at process start, from either
the local Hugging Face cache or an explicit local filesystem path), and
every subsequent call is in-process `torch` inference on the caller's own
crop. In `local_files_only` (production/offline) mode, no network call is
attempted at all - a missing local cache is a load failure, not a download.

`torch`/`transformers` are imported lazily, inside functions, not at module
level: this lets the rest of the pipeline (and its test suite) import this
module and construct a `TrOCRVerificationConfig` with `enabled=False`
without those packages being installed at all - the whole point of "TrOCR
must be behind configuration and must not break existing behaviour when
disabled" (see ocr_verify.py).

Real inference is provided by `TransformersTrOCRClient`. Tests (and any
caller that wants to avoid a multi-hundred-MB model load) inject a fake
implementing the same `TrOCRModelClient` protocol via `run_trocr`'s
`client` parameter - this is the seam required so the adapter's own
correctness (thread-safe single load, structured status, sanitized errors)
can be tested without downloading real weights.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from .models import SecondaryObservation, TrOCRStatus

DEFAULT_MODEL_NAME = "microsoft/trocr-base-handwritten"


class TrOCRDependencyError(RuntimeError):
    """torch/transformers are not installed but TrOCR was invoked."""


@dataclass(frozen=True)
class TrOCRRawResult:
    status: TrOCRStatus
    raw_text: str | None
    sequence_score: float | None
    score_type: str
    error_code: str | None
    latency_ms: float
    model_id: str
    model_version: str | None
    device: str


class TrOCRModelClient(Protocol):
    """Dependency-injection seam: anything with this shape can stand in for
    the real transformers-backed client in tests."""

    def generate(self, image: Image.Image, *, field_name: str) -> TrOCRRawResult: ...


def _select_device(preference: str) -> str:
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        return "cuda"
    # "auto": prefer CUDA only if genuinely available, never assume it.
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _sanitize_error(exc: Exception) -> str:
    """Never let a raw exception message (which can embed file paths, or in
    principle a stack frame containing local variables) reach a report or
    audit log. Return only the exception's type name plus a short, generic,
    pre-approved description."""
    return type(exc).__name__


class TransformersTrOCRClient:
    """Loads the configured TrOCR processor+model exactly once per process
    (thread-safe), then serves `generate()` calls against that single
    instance. CPU and CUDA are both supported; device selection never
    silently falls back from a user-requested "cuda" to "cpu" (a requested-
    but-unavailable CUDA device is a load failure, reported as such, not a
    silent downgrade that would mislead capacity planning).
    """

    def __init__(self, model_name_or_path: str = DEFAULT_MODEL_NAME,
                model_revision: str | None = None,
                device_preference: str = "auto",
                max_new_tokens: int = 32,
                local_files_only: bool = False,
                inference_timeout_s: float | None = 10.0):
        self.model_name_or_path = model_name_or_path
        self.model_revision = model_revision
        self.device_preference = device_preference
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self.inference_timeout_s = inference_timeout_s
        self._lock = threading.Lock()
        self._processor = None
        self._model = None
        self._device: str | None = None
        self._resolved_revision: str | None = None
        self._load_error: str | None = None
        self.model_load_latency_ms: float | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        with self._lock:
            if self._model is not None or self._load_error is not None:
                return
            started = time.monotonic()
            try:
                try:
                    import torch
                    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                except ImportError as exc:
                    raise TrOCRDependencyError(
                        "torch/transformers are not installed - TrOCR "
                        "verification cannot run until the optional ML "
                        "dependencies are installed (see requirements.txt "
                        "and docs/trocr_verification.md)") from exc

                device = _select_device(self.device_preference)
                if self.device_preference == "cuda" and device != "cuda":
                    raise RuntimeError("cuda was explicitly requested but "
                                      "torch reports no CUDA device available")

                processor = TrOCRProcessor.from_pretrained(
                    self.model_name_or_path, revision=self.model_revision,
                    local_files_only=self.local_files_only)
                model = VisionEncoderDecoderModel.from_pretrained(
                    self.model_name_or_path, revision=self.model_revision,
                    local_files_only=self.local_files_only)
                model.to(device)
                model.eval()

                self._processor = processor
                self._model = model
                self._device = device
                self._resolved_revision = (
                    self.model_revision
                    or getattr(model.config, "_commit_hash", None)
                    or "unpinned")
            except Exception as exc:  # noqa: BLE001 - deliberately broad: any load
                # failure must degrade to a structured status, never crash
                # the cheque pipeline; the sanitized reason is still recorded.
                self._load_error = _sanitize_error(exc)
            finally:
                self.model_load_latency_ms = (time.monotonic() - started) * 1000

    def generate(self, image: Image.Image, *, field_name: str) -> TrOCRRawResult:
        started = time.monotonic()
        self._ensure_loaded()
        if self._load_error is not None:
            return TrOCRRawResult(
                status=TrOCRStatus.FAILED, raw_text=None, sequence_score=None,
                score_type="unavailable", error_code=f"model_load_failed:{self._load_error}",
                latency_ms=(time.monotonic() - started) * 1000,
                model_id=self.model_name_or_path, model_version=self.model_revision,
                device=self.device_preference)

        try:
            import torch

            pixel_values = self._processor(
                images=image.convert("RGB"), return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self._device)

            with torch.inference_mode():
                outputs = self._model.generate(
                    pixel_values, max_new_tokens=self.max_new_tokens,
                    num_beams=1, output_scores=True, return_dict_in_generate=True)

            raw_text = self._processor.batch_decode(
                outputs.sequences, skip_special_tokens=True)[0].strip()

            score, score_type = None, "unavailable"
            try:
                transition_scores = self._model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True)
                probs = transition_scores.exp()
                finite = probs[torch.isfinite(probs)]
                if finite.numel() > 0:
                    score = float(finite.mean().item())
                    score_type = "raw_generation_evidence"
            except Exception:  # noqa: BLE001 - scoring is best-effort only;
                # a scoring failure must not fail the whole recognition.
                score, score_type = None, "unavailable"

            status = TrOCRStatus.COMPLETED if raw_text else TrOCRStatus.UNDETERMINED
            return TrOCRRawResult(
                status=status, raw_text=raw_text or None, sequence_score=score,
                score_type=score_type, error_code=None,
                latency_ms=(time.monotonic() - started) * 1000,
                model_id=self.model_name_or_path,
                model_version=self._resolved_revision, device=self._device)
        except Exception as exc:  # noqa: BLE001 - see module docstring: a
            # field-level inference failure must only affect this one
            # observation, never crash cheque processing.
            return TrOCRRawResult(
                status=TrOCRStatus.FAILED, raw_text=None, sequence_score=None,
                score_type="unavailable",
                error_code=f"generation_failed:{_sanitize_error(exc)}",
                latency_ms=(time.monotonic() - started) * 1000,
                model_id=self.model_name_or_path, model_version=self.model_revision,
                device=self._device or self.device_preference)


def run_trocr(image: Image.Image, field_name: str,
             preprocessing_variant: str,
             client: TrOCRModelClient) -> SecondaryObservation:
    """High-level entry point: run one crop variant through `client` and
    package the result as a SecondaryObservation. `client` is required
    (not defaulted to a real TransformersTrOCRClient here) so every call
    site must make an explicit choice about which client to use - callers
    needing the real model construct a module-level TransformersTrOCRClient
    once and pass it in (see ocr_verify.py).

    `TransformersTrOCRClient.generate()` already catches its own exceptions
    and returns a structured FAILED result, but `client` is an injectable
    Protocol - a test double or a future client implementation is not
    guaranteed to. A failure here must only affect this one field's
    verification observation, never propagate into cheque processing, so
    this boundary is defended too (belt-and-braces, not a substitute for
    the real client's own handling).
    """
    try:
        raw = client.generate(image, field_name=field_name)
    except Exception as exc:  # noqa: BLE001 - see docstring: any client
        # implementation's failure must degrade to one failed observation.
        return SecondaryObservation(
            engine="trocr", status=TrOCRStatus.FAILED, raw_value=None,
            normalized_value=None, score=None, score_type="unavailable",
            model_id=None, model_version=None,
            preprocessing_variant=preprocessing_variant, latency_ms=None,
            error_code=f"client_exception:{_sanitize_error(exc)}")
    return SecondaryObservation(
        engine="trocr", status=raw.status, raw_value=raw.raw_text,
        normalized_value=None, score=raw.sequence_score, score_type=raw.score_type,
        model_id=raw.model_id, model_version=raw.model_version,
        preprocessing_variant=preprocessing_variant, latency_ms=raw.latency_ms,
        error_code=raw.error_code,
    )
