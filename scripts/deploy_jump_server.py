r"""Deploy the HTML dashboard to the \\Th240netsrv\chequemate jump-server
share, where the layout is different from this repo's own: the HTML and
cheques/ sit as SIBLINGS in the same folder, not as reports/<html> next to
a repo-root cheques/ (see CLAUDE.md's "IMAGE_DIR relative-path trap").
regenerate_report()'s own output always bakes in '../cheques/', which is
wrong for this layout and silently breaks every "View Cheque" lightbox
there - this script calls report.generate_html() directly with the correct
image_dir for THIS target instead of copying regenerate_report()'s file
and hand-editing it (that edit would just be discarded the next time
anyone regenerates normally - see CLAUDE.md).

Read-only w.r.t. cheques.json - never writes it, only reads records to
render the same dashboard regenerate_report() would, with one templating
difference. Also mirrors cheques/ to the share (copies anything missing or
changed; never deletes anything there) so "View Cheque" has images to find.

Run:
    python scripts/deploy_jump_server.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chequemate import report  # noqa: E402

JUMP_SERVER_ROOT = Path(r"\\Th240netsrv\chequemate")
JUMP_SERVER_HTML = JUMP_SERVER_ROOT / "chequemate_report.html"
JUMP_SERVER_CHEQUES = JUMP_SERVER_ROOT / "cheques"


def render_for_sibling_layout() -> str:
    records = report.load_records()
    reviews = report.load_reviews()
    return report.generate_html(records, reviews, image_dir="cheques/")


def sync_cheques(local_dir: Path, remote_dir: Path) -> tuple[int, int]:
    """Copy anything missing or changed (by size) from local_dir to
    remote_dir. Never deletes anything on the remote side - this mirrors
    forward only, the same one-directional, non-destructive spirit as
    run_batch.py's own incremental design."""
    remote_dir.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for src in local_dir.iterdir():
        if not src.is_file():
            continue
        dest = remote_dir / src.name
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(src, dest)
        copied += 1
    return copied, skipped


def main() -> int:
    if not JUMP_SERVER_ROOT.is_dir():
        print(f"{JUMP_SERVER_ROOT} is not reachable from here.", file=sys.stderr)
        return 2

    copied, skipped = sync_cheques(ROOT / "cheques", JUMP_SERVER_CHEQUES)
    print(f"cheques/: {copied} copied, {skipped} already present")

    html = render_for_sibling_layout()
    JUMP_SERVER_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {JUMP_SERVER_HTML} ({len(html)} bytes, image_dir='cheques/')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
