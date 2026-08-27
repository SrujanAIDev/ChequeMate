r"""Signature-zone contact sheet: draws the FOUND printed line and the zone
DERIVED from it on every prepared cheque in a folder, so a human can
confirm the line-search is landing on the real signature/PER line before
it becomes load-bearing.

The zone is no longer a fixed rectangle assumed in advance - it's derived
per file from wherever `find_signature_line` actually finds a qualifying
line (see chequemate/signature.py's module docstring for why: a fixed
zone was the root cause of a confirmed false positive on a different
template). This sheet's job is now to confirm the SEARCH found the right
thing, not to confirm a pre-drawn box.

Run:
    python scripts/signature_contact_sheet.py

Writes debug/signature_contact_sheet_*.png - gitignored, never committed,
since the output embeds real cheque image content.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from chequemate.imageprep import (  # noqa: E402
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    ChequeImagePrepError,
    prepare_cheque_image,
)
from chequemate.signature import SIGNATURE_SEARCH_REGION_FRAC, derive_zone_from_line, find_signature_line  # noqa: E402

CHEQUES_DIR = ROOT / "cheques"
DEBUG_DIR = ROOT / "debug"

# Files the automated rotation detector refuses on (Correction B) - resolved
# via scripts/apply_rotation_override.py's operator-confirmed direction, so
# they can still go through the same pipeline for this check.
ROTATION_OVERRIDES = {
    "20260820113647241_0001.jpg": "CCW",
    "20260820113647241_0006.jpg": "CCW",
    "20260820113647241_0011.jpg": "CCW",
}

CELLS_PER_SHEET = 11
COLS = 3
CELL_SCALE = 0.5


def build_sheet(entries: list[tuple[str, Image.Image]], out_path: Path) -> None:
    cell_w, cell_h = int(CANVAS_WIDTH * CELL_SCALE), int(CANVAS_HEIGHT * CELL_SCALE)
    rows = (len(entries) + COLS - 1) // COLS
    sheet = Image.new("RGB", (cell_w * COLS, cell_h * rows), "white")
    draw_sheet = ImageDraw.Draw(sheet)

    for idx, (label, img) in enumerate(entries):
        thumb = img.resize((cell_w, cell_h), resample=Image.LANCZOS)
        r, c = divmod(idx, COLS)
        sheet.paste(thumb, (c * cell_w, r * cell_h))
        draw_sheet.text((c * cell_w + 6, r * cell_h + 6), label, fill="blue")

    sheet.save(out_path)
    print(f"[saved] {out_path} ({sheet.size[0]}x{sheet.size[1]})")


def annotate(image: Image.Image) -> tuple[Image.Image, str]:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = image.size

    x1f, y1f, x2f, y2f = SIGNATURE_SEARCH_REGION_FRAC
    draw.rectangle([round(x1f * w), round(y1f * h), round(x2f * w), round(y2f * h)],
                  outline="gray", width=2)

    line = find_signature_line(image)
    if line is None:
        return annotated, "NO LINE FOUND"

    draw.rectangle([line.col_start, line.row_start, line.col_end, line.row_end],
                  outline="red", width=4)
    zx1, zy1, zx2, zy2 = derive_zone_from_line(line, w, h)
    draw.rectangle([zx1, zy1, zx2, zy2], outline="blue", width=3)
    return annotated, f"line y=[{line.row_start},{line.row_end}) d={line.density:.2f}"


def main() -> int:
    DEBUG_DIR.mkdir(exist_ok=True)
    files = sorted(CHEQUES_DIR.glob("20260820113647241_*.jpg"))

    prepared: list[tuple[str, Image.Image]] = []
    failed: list[str] = []
    no_line: list[str] = []
    for path in files:
        try:
            result = prepare_cheque_image(
                path, rotation_override=ROTATION_OVERRIDES.get(path.name))
        except ChequeImagePrepError as exc:
            failed.append(f"{path.name}: {exc}")
            continue
        annotated, status = annotate(result.image)
        if status == "NO LINE FOUND":
            no_line.append(path.name)
        prepared.append((f"#{path.stem[-4:]} {status}", annotated))

    for i in range(0, len(prepared), CELLS_PER_SHEET):
        chunk = prepared[i:i + CELLS_PER_SHEET]
        build_sheet(chunk, DEBUG_DIR / f"signature_contact_sheet_{i // CELLS_PER_SHEET + 1}.png")

    print(f"\nPrepared: {len(prepared)}/{len(files)}")
    print(f"No line found (will report AMBIGUOUS): {len(no_line)} {no_line}")
    if failed:
        print("Failed (unresolved rotation or other prep error):")
        for line_ in failed:
            print(f"  {line_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
