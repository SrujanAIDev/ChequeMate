r"""ONE-OFF EXPERIMENT, not a pipeline change: does correcting a cheque's
orientation (isolate -> rotate -> deskew, via imageprep.py) BEFORE sending
it to Azure Document Intelligence produce better field extraction than the
sideways input every stored record in this repo was actually built from?

imageprep.py has never been wired into extract.py/run_batch.py - every
record in reports/cheques.json came from DI reading a 90-degree-rotated
cheque on a mostly-blank US Letter page. This script is the first time
that correction has ever been applied on the INPUT side rather than only
downstream (imageprep.py's existing use is region-based signature
detection, never re-extraction).

Deliberately does NOT call normalize_canvas() (the fixed 1860x780,
independent-x/y-scale resize) - that step exists for the signature
pipeline's fixed-coordinate-system needs, and stretching handwriting
non-uniformly before sending it to an OCR model has no equivalent
justification. The corrected image sent to DI is isolate -> rotate ->
deskew's own natural ("tight crop") output: cropped to the ink content,
upright, deskewed, nothing more.

This is a COMPARISON, not a migration:
  - reports/cheques.json is never read for writing and never modified.
  - No rule is changed. No verdict is recomputed. Nothing is re-scored.
  - Corrected raw Azure responses go to raw_corrected/ (gitignored,
    separate from raw/), never overwriting an existing raw/*.json.
  - Corrected input images (what was actually sent to DI) are saved to
    debug/corrected_input/ (already-gitignored) purely for audit -
    delete freely, they are not used by anything downstream.

Costs one real Azure Document Intelligence call per file processed.

Run:
    python scripts/experiment_imageprep_di_comparison.py
"""

from __future__ import annotations

import json
import os
import ssl
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import certifi  # noqa: E402

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where())

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from chequemate import imageprep  # noqa: E402
from chequemate.extract import analyze_raw, first_document, to_normalized  # noqa: E402
from chequemate.models import ParseStatus, RuleStatus  # noqa: E402
from chequemate.normalize import normalize_amount_words  # noqa: E402
from chequemate.rules import Config, check_memo, check_payee, check_signature  # noqa: E402
from run_batch import load_dotenv  # noqa: E402

CHEQUES_DIR = ROOT / "cheques"
RAW_DIR = ROOT / "raw"                              # stored/original - read-only
RAW_CORRECTED_DIR = ROOT / "raw_corrected"          # new - never overwrites raw/
DEBUG_INPUT_DIR = ROOT / "debug" / "corrected_input"

BATCH_FILES = [f"20260820113647241_{i:04d}.jpg" for i in range(1, 23)]

# Standing operator rotation confirmations (scripts/apply_rotation_override.py)
# - reused here exactly as recorded, never re-guessed.
ROTATION_OVERRIDES = {
    "20260820113647241_0001.jpg": "CCW",
    "20260820113647241_0006.jpg": "CCW",
    "20260820113647241_0011.jpg": "CCW",
}


def prepare_corrected_image(path: Path, rotation_override: str | None
                            ) -> tuple[Image.Image, imageprep.RotationDecision, float]:
    """isolate -> rotate -> deskew, deliberately stopping before
    normalize_canvas() - see this module's docstring for why."""
    image = Image.open(path).convert("RGB")
    dpi, dpi_source = imageprep._get_dpi(image)
    rgb = np.asarray(image)
    isolated = imageprep.isolate(rgb)
    rotated, rotation = imageprep.rotate(isolated, dpi, dpi_source,
                                         override=rotation_override)
    deskewed, skew_angle = imageprep.deskew(rotated)
    return Image.fromarray(deskewed), rotation, skew_angle


def run_experiment() -> None:
    load_dotenv()
    endpoint = os.getenv("AZURE_DI_ENDPOINT")
    key = os.getenv("AZURE_DI_KEY")
    if not endpoint or not key:
        print("Azure credentials not found.", file=sys.stderr)
        raise SystemExit(2)

    RAW_CORRECTED_DIR.mkdir(exist_ok=True)
    DEBUG_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in BATCH_FILES:
        dest = RAW_CORRECTED_DIR / (Path(name).stem + ".json")
        if dest.is_file():
            print(f"[skip] {name} (already fetched this experiment run)")
            continue

        path = CHEQUES_DIR / name
        if not path.is_file():
            print(f"[skip] {name} - not found in cheques/")
            continue

        try:
            corrected, rotation, skew = prepare_corrected_image(
                path, ROTATION_OVERRIDES.get(name))
        except imageprep.ChequeImagePrepError as exc:
            print(f"[FAILED PREP] {name}: {type(exc).__name__}: {exc}")
            continue

        debug_path = DEBUG_INPUT_DIR / name.replace(".jpg", ".png")
        corrected.save(debug_path)

        raw = analyze_raw(str(debug_path), endpoint, key)
        dest.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
        print(f"[fetched] {name}  rotation={rotation.direction} "
             f"({rotation.source})  skew={skew:+.2f}deg  -> {dest.name}")


if __name__ == "__main__":
    run_experiment()
