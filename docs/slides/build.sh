#!/usr/bin/env bash
# Build a beamer deck to PDF and PPTX:  bash docs/slides/build.sh path/to/deck.tex
# The PPTX has one full-slide image per PDF page (faithful to the PDF, not editable text).
# Needs: latexmk, pdftoppm (poppler), uv.
set -euo pipefail
TEX="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
DIR="$(dirname "$TEX")"; NAME="$(basename "$TEX" .tex)"
cd "$DIR"
latexmk -pdf -interaction=nonstopmode -quiet "$NAME.tex"
latexmk -c "$NAME.tex" >/dev/null 2>&1 || true; rm -f "$NAME.nav" "$NAME.snm"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pdftoppm -png -r 200 "$NAME.pdf" "$TMP/p"
uv run -q --with python-pptx python - "$TMP" "$NAME.pptx" <<'PY'
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Emu
from PIL import Image

pages, out = sorted(Path(sys.argv[1]).glob("p-*.png")), sys.argv[2]
w, h = Image.open(pages[0]).size
prs = Presentation()
prs.slide_width = Emu(12192000)                       # 13.333 in
prs.slide_height = Emu(round(12192000 * h / w))
blank = prs.slide_layouts[6]
for p in pages:
    prs.slides.add_slide(blank).shapes.add_picture(str(p), 0, 0, prs.slide_width, prs.slide_height)
prs.save(out)
print(f"wrote {out} ({len(pages)} slides)")
PY
echo "wrote $DIR/$NAME.pdf"
