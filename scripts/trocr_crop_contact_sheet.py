r"""Contact sheet of the exact crops sent to TrOCR for payee and
amount_words, across every record with a saved raw response - so a human
can visually confirm the crop is actually upright and readable before
trusting any agreement/disagreement statistic derived from it.

Re-derives crops via the exact same code path apply_trocr_verification.py
used (geometry.convert_bounding_region_to_pixels -> crops.validate_and_
generate_crop, with the same rotation/rotation_override resolution) -
this is a read-only diagnostic, it does not call TrOCR and does not touch
reports/cheques.json.

Each cell is labeled with the record id, DI's raw text, and (from the
already-persisted ocr_verifications) TrOCR's raw text, so a mismatch can
be judged against the actual pixels, not just the stored strings.

Run:
    python scripts/trocr_crop_contact_sheet.py

Writes debug/trocr_crop_contact_sheet_<field>_*.png - gitignored, never
committed, since the output embeds real cheque image content (same
privacy constraint as signature_contact_sheet.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from chequemate import ocr_verify, report  # noqa: E402
from chequemate.crops import FieldCropConfig, validate_and_generate_crop  # noqa: E402
from chequemate.geometry import convert_bounding_region_to_pixels  # noqa: E402

RAW_DIR = ROOT / "raw"
CHEQUES_DIR = ROOT / "cheques"
DEBUG_DIR = ROOT / "debug"

FIELDS = ("payee", "amount_words")
CELLS_PER_SHEET = 12
COLS = 3
CELL_W, CELL_H = 320, 180


def build_sheet(entries: list[tuple[str, Image.Image]], out_path: Path) -> None:
    rows = (len(entries) + COLS - 1) // COLS
    sheet = Image.new("RGB", (CELL_W * COLS, CELL_H * rows), "white")
    draw_sheet = ImageDraw.Draw(sheet)

    for idx, (label, img) in enumerate(entries):
        r, c = divmod(idx, COLS)
        x0, y0 = c * CELL_W, r * CELL_H
        # fit the crop into the top of the cell, preserving aspect ratio,
        # leaving room at the bottom for the label text
        img_area_h = CELL_H - 40
        scale = min((CELL_W - 8) / img.width, img_area_h / img.height, 4.0)
        thumb = img.resize((max(1, round(img.width * scale)),
                           max(1, round(img.height * scale))), resample=Image.LANCZOS)
        sheet.paste(thumb, (x0 + 4, y0 + 4))
        draw_sheet.rectangle([x0, y0, x0 + CELL_W - 1, y0 + CELL_H - 1], outline="gray")
        for i, line in enumerate(label.split("\n")):
            draw_sheet.text((x0 + 4, y0 + img_area_h + 6 + i * 12), line, fill="black")

    sheet.save(out_path)
    print(f"[saved] {out_path} ({sheet.size[0]}x{sheet.size[1]})")


def main() -> int:
    DEBUG_DIR.mkdir(exist_ok=True)
    records = report.load_records()

    per_field: dict[str, list[tuple[str, Image.Image]]] = {f: [] for f in FIELDS}
    skipped_no_raw = skipped_no_image = skipped_no_crop = 0

    for record in records:
        stem = Path(record["source_file"]).stem
        raw_path = RAW_DIR / (stem + ".json")
        image_path = CHEQUES_DIR / record["source_file"]
        if not raw_path.is_file():
            skipped_no_raw += 1
            continue
        if not image_path.is_file():
            skipped_no_image += 1
            continue

        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        doc = (raw.get("documents") or [{}])[0]
        fields = doc.get("fields", {})
        pages = raw.get("pages", [])

        rotation_meta = record.get("rotation") or {}
        rotation_override = (rotation_meta.get("direction")
                             if rotation_meta.get("source") == "operator_override"
                             else None)

        ocr = record.get("ocr_verifications") or {}

        with Image.open(image_path) as image:
            rotation_direction, refusal = ocr_verify.resolve_rotation(
                image_path, image, rotation_override=rotation_override)
            if refusal is not None:
                continue

            for field_name in FIELDS:
                azure_key = ocr_verify.FIELD_TO_AZURE_KEY[field_name]
                azure_field = fields.get(azure_key, {}) or {}
                bounding_regions = (azure_field.get("boundingRegions")
                                   or azure_field.get("bounding_regions"))
                conversion = convert_bounding_region_to_pixels(
                    bounding_regions, image_width_px=image.width,
                    image_height_px=image.height, pages=pages, expected_page_number=1)
                crop_cfg = ocr_verify._default_crop_configs()[field_name]
                crop, variants = validate_and_generate_crop(
                    image, conversion, crop_cfg, rotation=rotation_direction)
                if variants is None:
                    skipped_no_crop += 1
                    continue

                fv = ocr.get(field_name, {})
                di_text = (fv.get("primary") or {}).get("raw_value") or "(none)"
                trocr_text = (fv.get("secondary") or {}).get("raw_value") or "(none)"
                comparison = (fv.get("comparison") or {}).get("status", "?")
                label = (f"{record['record_id']} [{comparison}]\n"
                        f"DI: {di_text[:28]}\n"
                        f"TrOCR: {trocr_text[:28]}")
                per_field[field_name].append((label, variants["rgb_normalized"].copy()))

    for field_name, entries in per_field.items():
        for i in range(0, len(entries), CELLS_PER_SHEET):
            chunk = entries[i:i + CELLS_PER_SHEET]
            build_sheet(chunk, DEBUG_DIR /
                       f"trocr_crop_contact_sheet_{field_name}_{i // CELLS_PER_SHEET + 1}.png")
        print(f"{field_name}: {len(entries)} crops")

    print(f"\nSkipped - no raw response: {skipped_no_raw}")
    print(f"Skipped - no source image: {skipped_no_image}")
    print(f"Skipped - crop not accepted/rotation refused: {skipped_no_crop}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
