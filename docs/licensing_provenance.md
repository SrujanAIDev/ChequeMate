# Licensing and provenance register

Every new package and model added for TrOCR field-level secondary
verification (see `docs/trocr_verification.md` for the feature itself),
plus the direct dependencies that already shipped in ChequeMate before this
feature existed. No AGPL (or other network-copyleft) component is used
anywhere in this list.

## Model weights

| Component | Version / revision | Source | License | Intended use | Deployment determination |
|---|---|---|---|---|---|
| `microsoft/trocr-base-handwritten` | pinned via `CHEQUEMATE_TROCR_MODEL_REVISION` (unset = HF's default revision at install time; **production must pin an explicit revision**, see below) | Hugging Face Hub (`https://huggingface.co/microsoft/trocr-base-handwritten`), originally published by Microsoft (`microsoft/unilm/trocr`) | MIT (per Microsoft's `unilm` repository license, which covers the TrOCR model family) | Field-level secondary text recognition on a single validated crop (payee / legal amount / memo) - never the full cheque, never bank/MICR/signature regions | **Local only.** Weights are downloaded once during approved setup (`local_files_only=False`, an operator-run one-time step) into the Hugging Face cache or an explicit local path, then loaded from disk for every subsequent run. Runtime cheque crops are never uploaded anywhere; no hosted-inference API is used. Production deployments set `CHEQUEMATE_TROCR_LOCAL_FILES_ONLY=true` (or `CHEQUEMATE_TROCR_LOCAL_MODEL_PATH` to a pre-approved local directory) so no network access is even attempted at runtime. |

**Revision pinning**: this repo does not vendor a specific commit hash by
default. Before production use, resolve and record the exact revision hash
once (`git ls-remote https://huggingface.co/microsoft/trocr-base-handwritten
refs/heads/main`, or read it from a `from_pretrained(..., local_files_only=False)`
call's logged `_commit_hash`), then set `CHEQUEMATE_TROCR_MODEL_REVISION` to
that hash so re-running setup can never silently pull a different model
version.

## New Python packages (TrOCR feature)

| Package | Version | License | Intended use |
|---|---|---|---|
| `torch` | 2.13.0 (CPU build) | BSD-3-Clause | TrOCR model inference runtime |
| `transformers` | 4.57.6 | Apache-2.0 | `TrOCRProcessor` / `VisionEncoderDecoderModel` - loads and runs TrOCR locally. Pinned to the 4.x line, not 5.x - see `requirements.txt`'s comment: `transformers==5.16.1` cannot load this model's tokenizer at all (confirmed by hand against the real weights), a real incompatibility, not a hypothetical caveat. |
| `sentencepiece` | 0.2.2 | Apache-2.0 | Tokenizer backend transformers resolves for this model family |
| `safetensors` | 0.8.0 | Apache-2.0 | Safe (non-pickle) model weight deserialization, used by `from_pretrained` |
| `huggingface_hub` (transitive, via `transformers`) | 0.36.2 | Apache-2.0 | Local cache resolution for `from_pretrained`; never used to submit cheque data |
| `tokenizers` (transitive) | 0.22.2 | Apache-2.0 | Fast tokenizer backend |
| `numpy` (already present) | 2.5.2 | BSD-3-Clause | Already a core dependency (imageprep.py, signature.py); reused, not re-added |

Transitive dependencies of `torch`/`transformers` not listed individually
here (`sympy`, `networkx`, `jinja2`, `filelock`, `fsspec`, `pyyaml`,
`regex`, `tqdm`, `click`, `typer`, `rich`, etc.) were checked at install
time and are all BSD/MIT/Apache-2.0/MPL-2.0/public-domain licensed - see
`pip show <package>` for any individual package's declared license, or
`requirements.txt` for the exact pinned version set.

## Pre-existing packages (unchanged by this feature)

| Package | Version | License | Intended use |
|---|---|---|---|
| `azure-ai-documentintelligence` | 1.0.2 | MIT | Primary structured cheque field extraction (`prebuilt-check.us`) |
| `azure-core` | 1.41.0 | MIT | Azure SDK transport/auth |
| `pillow` | 12.3.0 | MIT-CMU | Image loading, cropping, preprocessing |
| `python-dateutil` | 2.9.0.post0 | Apache-2.0 / BSD (dual) | Date arithmetic in `rules.py` |
| `requests` | 2.34.2 | Apache-2.0 | HTTP transport dependency |
| `certifi` | 2026.7.22 | MPL-2.0 | CA bundle |
| `pytest` | 9.1.1 | MIT | Test runner |

## How this register is kept current

Update this file whenever `requirements.txt` gains a new direct dependency,
whenever `CHEQUEMATE_TROCR_MODEL`/`CHEQUEMATE_TROCR_MODEL_REVISION` changes
to a different model, or whenever a production deployment resolves and
pins a specific model revision hash (record it in the table above, not
just in an environment variable, so it survives outside any one
deployment's `.env`).
