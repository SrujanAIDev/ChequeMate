"""Tests for chequemate.trocr_adapter that don't require a real model load -
these exercise the adapter's own contracts (structured status, sanitized
errors, the run_trocr() boundary) via injected fakes, per the dependency-
injection seam the module is built around. Tests that need the real
transformers/torch stack loaded live in test_trocr_adapter_integration.py
(skipped automatically when those packages or local weights aren't
available).
"""

from __future__ import annotations

from PIL import Image

from chequemate.models import SecondaryObservation, TrOCRStatus
from chequemate.trocr_adapter import TrOCRRawResult, _sanitize_error, run_trocr


class _RaisingClient:
    def generate(self, image, *, field_name):
        raise RuntimeError("simulated model crash with a secret path C:\\Users\\real\\name")


class _WorkingClient:
    def __init__(self, text="Town of Whitby", score=None, score_type="unavailable"):
        self.text = text
        self.score = score
        self.score_type = score_type

    def generate(self, image, *, field_name):
        return TrOCRRawResult(
            status=TrOCRStatus.COMPLETED, raw_text=self.text,
            sequence_score=self.score, score_type=self.score_type,
            error_code=None, latency_ms=5.0, model_id="fake-model",
            model_version="rev1", device="cpu")


def _tiny_image() -> Image.Image:
    return Image.new("RGB", (50, 20), (255, 255, 255))


def test_run_trocr_wraps_successful_result():
    obs = run_trocr(_tiny_image(), "payee", "rgb_normalized", _WorkingClient())
    assert isinstance(obs, SecondaryObservation)
    assert obs.status is TrOCRStatus.COMPLETED
    assert obs.raw_value == "Town of Whitby"
    assert obs.preprocessing_variant == "rgb_normalized"
    assert obs.model_id == "fake-model"


def test_run_trocr_client_exception_becomes_failed_observation_not_a_crash():
    obs = run_trocr(_tiny_image(), "payee", "rgb_normalized", _RaisingClient())
    assert obs.status is TrOCRStatus.FAILED
    assert obs.raw_value is None
    assert obs.error_code is not None
    assert obs.error_code.startswith("client_exception:")


def test_run_trocr_exception_message_is_sanitized_not_leaked():
    obs = run_trocr(_tiny_image(), "payee", "rgb_normalized", _RaisingClient())
    # the raw exception text (with its embedded fake "path") must never
    # reach the structured observation - only the exception type name does.
    assert "secret path" not in (obs.error_code or "")
    assert "C:\\Users\\real\\name" not in (obs.error_code or "")
    assert "RuntimeError" in obs.error_code


def test_sanitize_error_returns_type_name_only():
    try:
        raise ValueError("some message containing /a/sensitive/path and a key=SECRET123")
    except ValueError as exc:
        sanitized = _sanitize_error(exc)
    assert sanitized == "ValueError"
    assert "SECRET123" not in sanitized
    assert "sensitive" not in sanitized


def test_run_trocr_missing_score_reports_unavailable_not_a_fabricated_number():
    obs = run_trocr(_tiny_image(), "amount_words", "rgb_normalized",
                    _WorkingClient(score=None, score_type="unavailable"))
    assert obs.score is None
    assert obs.score_type == "unavailable"


def test_run_trocr_raw_generation_evidence_labeled_distinctly_from_confidence():
    obs = run_trocr(_tiny_image(), "payee", "rgb_normalized",
                    _WorkingClient(score=0.87, score_type="raw_generation_evidence"))
    assert obs.score == 0.87
    assert obs.score_type == "raw_generation_evidence"
    assert "confidence" not in obs.score_type
