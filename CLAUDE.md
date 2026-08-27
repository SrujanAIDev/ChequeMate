# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ChequeMate validates scanned/photographed cheques against Service Whitby's acceptance
rules. Azure Document Intelligence (`prebuilt-check.us`, a fixed prebuilt cloud model —
**not** retrainable from this codebase) extracts raw field text from a cheque image; this
repo turns that into a typed, provenance-tracked record, runs five independent rules over
it, and produces one auditable verdict plus a self-contained HTML dashboard. No ML lives in
this repo — `rules.py`/`normalize.py` are deterministic, hand-written Python.

## Commands

No `requirements.txt` / `pyproject.toml` exists. The `.venv/` (Python 3.14, created via
`uv`) already has the runtime deps installed: `azure-ai-documentintelligence`,
`azure-core`, `certifi`, `python-dateutil`, `pytest`, `requests`. On Windows, `python` is
not on PATH directly (it hits the Microsoft Store alias) — always invoke via
`.\.venv\Scripts\python.exe`. There is no linter/formatter configured and no build step.

```powershell
# Full test suite (tests/test_cheque.py + tests/test_report.py)
.\.venv\Scripts\python.exe -m pytest -q

# One file / one test
.\.venv\Scripts\python.exe -m pytest tests/test_cheque.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_cheque.py::test_minor_payee_typo_is_tolerated -q

# Zero-setup sanity check (hand-transcribed Azure-shaped dicts, no Azure/venv extras needed)
.\.venv\Scripts\python.exe demo.py

# Regenerate the HTML dashboard from reports/cheques.json (safe to rerun anytime, idempotent)
.\.venv\Scripts\python.exe -c "from chequemate import report; report.regenerate_report()"
```

Azure credentials (needed by `cli.py` without `--replay`, and by `scripts/run_batch.py` for
any genuinely new image): a gitignored `.env` at repo root with `AZURE_DI_ENDPOINT=` and
`AZURE_DI_KEY=`, or real env vars. In PowerShell, `set VAR=value` does **not** work — use
`$env:AZURE_DI_ENDPOINT = "..."`.

Ways to run the pipeline, in increasing order of "real":
- `cli.py` — per-image/glob CLI. `--save-raw DIR` captures the full Azure response once so
  everything downstream can be replayed offline forever after: `--replay` reads saved JSON
  instead of calling Azure. `--diagnose` dumps the `PayerSignatures` field key-by-key.
- `scripts/run_batch.py [--folder DIR] [--verbose] [--save-raw DIR]` — the actual production
  entrypoint. Incremental: hashes every file and skips anything already in `cheques.json`
  **before** touching Azure, so cost only scales with genuinely new cheques. One bad file
  logs a warning and doesn't abort the batch. `--save-raw` defaults **on** (writes to
  `raw/<name>.json`, same as `cli.py`'s flag) specifically so a "was field X actually absent
  from Azure's response, or lost in normalization?" question never again requires a second
  Azure call to answer — pass `--save-raw ""` to disable.

## Architecture

### Pipeline

`extract.py` (Azure adapter — the only file that knows Azure's field-naming quirks, e.g.
`PayerSignatures` needs multi-key handling because Azure reports it as a signature-typed
value, not via `.value`) → `normalize.py` (raw text → typed `Field`s) → `rules.py` (five
independent `check_*` functions, one `RuleResult` each) → `validate.py` (`validate()` runs
all five, collapses them to one `Verdict`) → `report.py` (persists to
`reports/cheques.json`, regenerates the HTML dashboard). `chequemate/__init__.py` is the
public surface every script imports from: `Config`, `Verdict`, `validate`, etc.

### Provenance over guessing

The load-bearing design decision (see `models.py`'s module docstring) is that every
extracted value carries a `ParseStatus`: `OK` / `ABSENT` (extractor returned nothing) /
`UNPARSEABLE` (text present, couldn't normalise) / `AMBIGUOUS` (parsed, more than one
valid reading). Every `normalize_*` function in `normalize.py` follows this shape and
never fabricates a value it isn't sure of — e.g. `normalize_amount_words`'s
`_MISSPELLINGS` table is a fixed lookup, deliberately **not** fuzzy string matching,
specifically so "sixty" can never silently snap to "sixteen".

### RuleStatus is 3-valued; Verdict is 3-valued — the collapse is deliberate

`RuleStatus` is `PASS` / `FAIL` / `UNABLE`. `Verdict` is `VALID` / `REVIEW` / `INVALID`.
`validate.py`'s verdict line is the single most important business-logic decision in the
codebase:

```python
if any(r.status is RuleStatus.FAIL for r in results):
    verdict = Verdict.INVALID
elif any(r.status is RuleStatus.UNABLE for r in results):
    verdict = Verdict.REVIEW
else:
    verdict = Verdict.VALID
```

`UNABLE` ("couldn't extract/read this field") does **not** invalidate by itself — only a
genuine `FAIL` does; that was a deliberate reversal from an earlier all-non-PASS-invalidates
design that was wrongly flagging huge numbers of real cheques INVALID purely because Azure
returned no signature verdict, not because anything was actually wrong. But `UNABLE` also
must not be reported as clean — `REVIEW` (added in `1.5.0`) exists because collapsing
`UNABLE` straight to `VALID` (the `1.3.0`/`1.4.0` behavior) meant every cheque a rule
couldn't assess got waved through unreviewed, e.g. every real unsigned cheque the signature
detector couldn't reach a verdict on. `FAIL` still always wins over `UNABLE`. Any rule work
needs to preserve this three-way distinction, not treat it as a bug or collapse it back to
two states.

`scripts/apply_visual_verification.py`'s `_recompute_verdict` mirrors this exact collapse
(it's the function `scripts/revalidate.py` reuses when layering a standing override back
onto a freshly re-derived record — see below) — if it ever drifts out of sync with
`validate.py`'s logic, re-applying a standing override will silently produce the wrong
verdict for that record.

### `Config` (rules.py) is the tunable surface

`expected_payee`, `payee_edit_tolerance` (2 — see below), `amount_tolerance`,
`max_age_months`, `reject_postdated` (`False` — post-dated tax-instalment cheques are
routine at Service Whitby and must `PASS` with a routing note, not `FAIL`; stale-dated,
unaffected, still hard-fails), `require_memo` (`True` — a blank memo is a hard business
`FAIL`, not `UNABLE`).

### Payee matching is token-based, not whole-string

`check_payee` does **not** check whether the extracted text equals `"Town of Whitby"`. It
extracts the one "distinctive token" from the expected payee (`_distinctive_token` strips
generic municipal words — "town", "city", "of", "corporation" — via `_GENERIC_PAYEE_WORDS`)
and checks whether **any token** in the extracted text is within edit distance of that word.
This is why `"WHITBY TAXES"`, `"FELICIA LYNN BAYLIS Town of Whitby"` (a drawer's own printed
name merged in by a bad extraction), and `"Down of whitby"` (OCR flipped T→D) all correctly
`PASS` — real cheques phrase the payee many ways, and the town name is what actually
matters. Separately, `_BANK_KEYWORDS` detects when Azure's `PayTo` field has latched onto
the drawee bank's own letterhead instead of the handwritten payee line (a confirmed, real
extraction failure mode) and routes that to `UNABLE` rather than `FAIL` — "extraction
grabbed the wrong text" and "this cheque is made out to someone else" need different staff
follow-up, so they must not look the same in the report.

### Amount-in-words cents recovery is deliberately conservative

Cheques universally print a `".../100 DOLLARS"` cents suffix; OCR frequently drops the `/`
but leaves the `100` token. `normalize_amount_words` recovers this (`_TRAILING_CENTS_100`:
a bare `100` near the end, optionally preceded by a 1-2 digit numerator, is the cents
denominator) before falling back to rejecting the whole amount as contaminated. This is
intentionally narrow: a bare small number with **no** `100` nearby is still rejected as
ambiguous (`test_stray_digits_in_word_amount_rejected` guards this) — `100` in that specific
position is unambiguous template boilerplate, a stray digit elsewhere might not be.

### report.py generates one fully self-contained, offline HTML file

No build step, no bundler, no external JS library ever (no CDN) — by design, since this may
run against real MFIPPA-covered cheque data and the report must never be able to phone home.
`_CSS` / `_JS_TEMPLATE` / `_HTML_TEMPLATE` are plain Python string constants stitched via
`.replace("__TOKEN__", ...)` — deliberately not `.format()`/f-strings, because the CSS/JS
bodies are full of literal `{`/`}` that would collide with Python's brace templating. Every
record + review is embedded as one JSON blob inside a `<script>` tag
(`_json_for_script`, which escapes `</` to prevent premature tag closure); the table,
filtering, sorting, CSV/XLSX export (including a hand-rolled ZIP/OOXML writer — no
SheetJS), and the cheque-image lightbox are all client-side JS over that embedded array.
There is no server. **The HTML is always fully rebuilt from `cheques.json`, never patched
in place** — hand-editing the deployed `.html` file gets silently discarded on the next
`regenerate_report()`.

### The `IMAGE_DIR` relative-path trap

The report never embeds cheque image bytes (same privacy reasoning as above) — the "View
Cheque" lightbox references images by a path relative to wherever the generated HTML
currently lives, resolved live by the browser. This value is topology-dependent and has
already broken once in production: local `regenerate_report()` writes to
`reports/chequemate_report.html` with images at repo-root `cheques/` (needs `../cheques/`),
but at least one real deployment puts the HTML and `cheques/` as **siblings** in the same
folder (needs `cheques/`, no `../`). A leading slash (`/cheques/`) is a third, always-wrong
value — browsers resolve it as root-absolute and silently discard whatever subdirectory the
report was deployed into. Know which layout you're generating for before touching this.

### Privacy boundary: `reports/`, `cheques/`, `raw/`, `.env` are gitignored on purpose

Real cheque images, extracted PII, and Azure credentials must never enter git history. Every
script touching these directories assumes real sensitive data may be present: console output
stays to record IDs/filenames/verdicts, never full field values, and nothing ever copies
image bytes anywhere.

### `scripts/` is one-shot, idempotent maintenance ops, not a package

Each script bootstraps its own `sys.path` and follows the same pattern: back up
`cheques.json` once (never overwrite an existing `.bak`), process and persist one record at
a time with `save_records()` + `append_audit_event()` called together per record (so a crash
mid-run can never leave the audit trail ahead of the data — a re-run safely skips whatever
already completed), then call `report.regenerate_report()`. `run_batch.py` is the odd one
out — it's the ingestion entrypoint (calls Azure), not a re-scoring script. The re-scoring
scripts differ in what they can recover from: `revalidate.py` re-runs current
`normalize.py`/`rules.py` logic against every record's already-stored `raw_values`, entirely
offline — correct after a **rules** change, useless after a **schema/extraction** change
(the field was never captured, so there's no raw text to re-parse). `migrate_memo.py` shows
the pattern for that harder case: re-processing from the original source image via Azure
(offline replay preferred when a saved raw response exists) with a fallback path when the
source is gone. `apply_visual_verification.py` is for the hardest case — a human visually
confirmed the true value against the source image and automation still can't resolve it; it
never overwrites the original OCR text, only adds a `reviews.json` entry + audit event
alongside it.

**`revalidate.py` layers standing visual-verification overrides back on, it never skips
records that have them.** It rebuilds every record purely from `raw_values` via current
`rules.py`/`normalize.py` logic exactly as always — that's the whole point of the script — but
for any record with a standing `apply_visual_verification.py` correction, the confirmed
value(s) (imported from that script's `SIGNATURE_CONFIRMATIONS`/`PAYEE_CONFIRMATIONS`, the
same source it uses itself — `reviews.json`'s `note` is prose for a person, not a
machine-readable value) are re-applied on top, and the verdict is recomputed *after* the
overlay. This happened for real during the 1.3.0 → 1.4.0 bump: the first fix was to skip such
records entirely, which stopped the immediate data loss but created a slower version of the
same problem — a human confirming one field would freeze all five against every future rules
improvement, and the record's `ruleset_version` would silently drift from the rest of the
corpus. Layer, don't skip.

### Ruleset versioning

`validate.RULE_SET_VERSION` is stamped on every record and bumped whenever
`rules.py`/`normalize.py` behavior changes: `1.0.0` → `1.1.0` (added the memo field/rule) →
`1.2.0` (post-dated cheques `PASS` instead of `FAIL`) → `1.3.0` (token-based payee matching,
amount cents recovery, `UNABLE` no longer invalidates) → `1.4.0` (an orientation-indeterminate
source image maps to `ParseStatus.AMBIGUOUS` on the signature field, which `check_signature`
routes to `UNABLE` — the pipeline-boundary contract for `imageprep.OrientationIndeterminate`)
→ `1.5.0` (added the `REVIEW` verdict — any `UNABLE` on a load-bearing rule now routes there
instead of silently collapsing to `VALID`; `FAIL` still always produces `INVALID`).
After bumping it, run `scripts/revalidate.py` to re-score existing history.

### Image preparation (`chequemate/imageprep.py`)

Pure functions (`isolate` → `rotate` → `deskew` → `normalize_canvas`) that turn a raw,
physically-scanned cheque photo into an upright, fixed-size image for region-based detectors
(the Phase 4 signature check). Not yet wired into `extract.py`/`run_batch.py` — building this
module and validating it against the real 22-file `20260820...` batch was itself the
deliverable; live integration is a later phase.

The rotation step is the interesting part: the batch's cheques are all rotated 90° with no
consistent direction, and generic ink-density/texture heuristics are actively unreliable here
because several cheque templates carry a decorative border denser and more "texty" than the
real MICR line. The one signal specific to MICR is that E-13B is fixed-pitch (8 chars/inch);
a genuine MICR line produces a strong autocorrelation peak at `DPI / 8` (read from the image's
own JFIF tag, never hardcoded) reinforced by its 2x harmonic, which a decorative border
essentially never reproduces at both. A required confidence margin between the winning and
losing orientation (not just clearing an absolute floor) trades a few extra
`OrientationIndeterminate` refusals for zero confident-wrong rotations — on the real batch,
19/22 resolve confidently and correctly, 3/22 (`_0001`, `_0006`, `_0011`) correctly refuse
rather than guess, because their template's border periodicity genuinely rivals the true MICR
pitch. See `docs/scanning_recommendations.md` for the operational fix (feed cheques landscape,
clear of the leading edge) that would eliminate this class of problem at the source instead of
detecting around it.

A permanent refusal needs a resolution path — `scripts/apply_rotation_override.py` is that
path: a human confirms the direction for one specific, individually-checked file (never a
batch-wide "assume CCW" default — see the module docstring), and
`prepare_cheque_image(path, rotation_override=...)` consumes it. `RotationDecision.source`
distinguishes `"detector"` from `"operator_override"`, and an override's record carries a
`detector_note` recording exactly what the detector refused with — the refusal is never
silenced, only supplemented. All 3 of the batch's indeterminate files have been resolved this
way (all genuinely CCW, individually confirmed, matching the rest of the batch).

`OrientationIndeterminate` must be caught at whichever pipeline boundary eventually calls
`prepare_cheque_image()` and mapped via `normalize_signature(..., ambiguous_reason=str(exc))`
— never left to propagate and abort a batch, and never silently treated as a PASS. See
`test_orientation_indeterminate_maps_to_unable_not_fail_or_pass` in `tests/test_cheque.py`.

### Signature-zone ink detector (`chequemate/signature.py`)

Boolean `signature_present` (not a graded scale): any ink in the signature zone counts,
including a stray mark or a partial stroke. That rule makes **line removal the entire
detector**, not a preprocessing step for it — the printed PER/signature line is itself ink
sitting inside the zone, and an unremoved line reads as "present" on a genuinely blank
cheque.

**The zone is derived from the line, never assumed.** An earlier version fixed a
`SIGNATURE_ZONE_FRAC` rectangle on the canvas, calibrated by eyeballing a contact sheet of the
22-file batch. That rectangle was the root cause of a real, confirmed failure: on
`cheques/cheque2.png.png` (a different, older template — 96 DPI, 365×816 native, not part of
the batch), the fixed zone happened to contain both the real signature line and unrelated
printed `"/100 DOLLARS"` text, and density-based selection picked the wrong one — a **confident
false positive**, not a refusal. The fix is architectural: `find_signature_line()` searches a
generous region (`SIGNATURE_SEARCH_REGION_FRAC`) for the printed line first — the one feature
guaranteed present on every cheque, signed or not — and `derive_zone_from_line()` builds the
zone from whatever was actually found (a band above that specific line, bounded by its own
horizontal extent). If no line is found anywhere in the region, this refuses
(`ParseStatus.AMBIGUOUS` → `RuleStatus.UNABLE`, same pipeline-boundary pattern as
`OrientationIndeterminate`) rather than measuring ink in a zone nobody confirmed is meaningful.

The printed line is **not always a solid rule** — on most of this batch's templates it's a
microprinted security text line (tiny repeated words with gaps), which never forms one long
contiguous run. Detection is span-based: a row's leftmost-to-rightmost ink extent covering
`LINE_MIN_SPAN_CANVAS_FRAC` of the *canvas* width (not the search region's width, which can
vary). Among all qualifying windows, **the lowest (bottom-most) one wins, not the densest** —
this is the specific fix for the cheque2.png.png failure mode: dense printed TEXT (like
`"/100 DOLLARS"`) routinely beats a genuine but sparser printed line on density alone, but the
signature/PER line is structurally the *last* ruled line before a cheque's bank-name/MICR
block, so "lowest valid" reliably lands on the right one where "highest density" did not. The
search region's bounds (`0.45, 0.45, 1.0, 0.92` of canvas) were the widest found that fixed
cheque2.png.png without introducing a new false-confident match on any of the 22 real files —
wider regions were tried and rejected (they merge the separate MEMO line into the signature
line's "span" since both sit on the same cheque row on some templates, and start matching MICR
text/bank-address blocks that are also wide and dense).

**Hard input guard**: every constant here is fitted to 300 DPI scans of one template family.
`analyze_signature_zone` now takes a `PreparedCheque` (not a bare `Image`) specifically so it
can check DPI and canvas provenance on entry, raising `SignatureEnvelopeError` for anything
outside `VALIDATED_DPI ± VALIDATED_DPI_TOLERANCE_FRAC` or a non-JFIF (`fallback`) DPI source —
never computing a confident number on input nobody has validated this pipeline against. This
converts the cheque2.png.png failure into an early, honest refusal categorically: its 96 DPI is
caught before line-search ever runs, regardless of how good or bad the line-search heuristic
turns out to be on any given file.

`NOISE_FLOOR` is calibrated in `scripts/calibrate_signature_threshold.py` against a synthetic
negative built from real, known-signed cheques with the handwritten signature erased but the
printed line left intact (`synthesize_unsigned_variant(keep_line=True)`) — the case that
actually exercises line removal. Current separation is narrow (0.00079 between the real-signed
floor and the synthetic-negative ceiling) — reported honestly, not hidden; a fully-blanked-zone
synthetic variant (erasing the line too) no longer produces a meaningful measurement under this
architecture, since erasing the line removes the exact anchor `find_signature_line` needs,
usually triggering `AMBIGUOUS` but occasionally finding a different, spurious wide-dense row
elsewhere in the region — a known, reported limitation of the *synthetic test methodology*, not
a real-world scenario (a genuine unsigned cheque always still has its printed line).

**Absence detection has never been validated on a real unsigned cheque of the current
template** — every "correctly absent" result comes from a synthetic negative, and the one real
non-batch negative available (`cheque2.png.png`) is refused by the envelope guard rather than
measured. Run `scripts/signature_scorecard.py` for the current honest summary. Not wired into
any verdict.

**Every constant in this module is batch-fitted, not physically derived** — unlike
`imageprep.py`'s MICR pitch (`DPI / 8`, a real property of the E-13B font, resolution-scaled by
construction), `SIGNATURE_ZONE_FRAC`, `LINE_SPAN_WIDTH_FRAC`, `LINE_MAX_GROUP_HEIGHT_PX`,
`LINE_MIN_GROUP_DENSITY`, `NEUTRAL_CHROMA_MAX`, and `NOISE_FLOOR` were all confirmed or
calibrated only against the 22-file `20260820...` batch's specific templates and 300 DPI scan
characteristics. This is not a hypothetical caveat: `cheques/cheque2.png.png` (a much older,
unrelated test fixture — 96 DPI, 365×816 native, a different template) is the one real,
independently-known-unsigned cheque in this repo, was never part of that batch or its
calibration, and the detector gets it **wrong** — a confident false positive
(`ink_coverage=0.0103`, `stroke_extent_frac=0.69`, comfortably above `NOISE_FLOOR`), not a
coincidental refusal. Root cause, confirmed by inspection: this template's proportions place
printed `"/100 DOLLARS"` text inside `SIGNATURE_ZONE_FRAC`'s bounds; a thin slice of that text
gets misidentified as "the line" and removed, leaving the rest of the static text as residual
"ink". `test_real_unsigned_cheque_is_a_known_false_positive` in `tests/test_signature.py`
locks this in deliberately as a documented limitation, not a bug to silently start passing.
**Do not retune any constant here against that one file** — one sample is not a distribution;
the fix (if pursued) is re-confirming the zone against that template's own contact sheet, not
nudging a threshold. This module should be understood as validated for the 22-file batch's
template family and resolution specifically, not as a general-purpose detector yet.

### Extending the pipeline (adding a field, as memo did)

Touch, in this order: `models.py` (add the `Field` to `NormalizedCheque`, with a
`default_factory` so existing callers don't break) → `extract.py` (map the Azure field name
in `FIELD_MAP`, add a `normalize_*` call in `to_normalized`) → `rules.py` (add a
`check_*` function, add it to `ALL_RULES`) → `validate.py` (call the new check in
`validate()`, bump `RULE_SET_VERSION`) → `report.py` (surface the field in the record dict
built by `create_report_record`, and in the JS table/detail-panel templates if it should be
visible).

### Not part of the live pipeline

`archive/Cheque_Validator.py` (gitignored) is a pre-split monolith prototype using an
incompatible vocabulary (`PASS`/`REVIEW`/`FLAG` verdicts, per-field confidence floors) —
historical reference only, don't try to reconcile it with the current design.
