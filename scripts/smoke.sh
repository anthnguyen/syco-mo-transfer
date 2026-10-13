#!/usr/bin/env bash
# Local / pod pipeline verification: unit tests, preflight, the whole v2 pipeline on tiny
# models, the run checks, and a resume check (a second invocation must be a pure no-op).
#   bash scripts/smoke.sh                       # fresh run of configs/v2_smoke.yaml
#   KEEP=1 bash scripts/smoke.sh                # reuse the existing smoke run directory
#   bash scripts/smoke.sh configs/other.yaml    # another (small) config
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-configs/v2_smoke.yaml}"
uv sync --frozen
PY=.venv/bin/python
OUT=$("$PY" -c "import sys; from sycomo.config import load_config; print(load_config(sys.argv[1]).out_dir)" "$CFG")

$PY -m pytest -q
if [ "${KEEP:-0}" != "1" ]; then
  case "$OUT" in *smoke*) rm -rf "$OUT" ;; *) echo "refusing to delete $OUT (not a smoke dir); use KEEP=1"; exit 1 ;; esac
fi
$PY scripts/preflight.py "$CFG"
$PY -u -m sycomo --config "$CFG"
$PY scripts/check_run.py "$CFG"

# resume: a second invocation must skip every stage and change no artifact
hash256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
snap() { find "$OUT" -type f -not -name manifest.json -not -name 'check_report*.json' -not -name '*.log' \
           | LC_ALL=C sort | while IFS= read -r f; do hash256 "$f"; done | hash256; }
before=$(snap)
second=$($PY -u -m sycomo --config "$CFG" 2>&1)
grep -q "skip report (done)" <<<"$second" || { echo "FAIL resume: report stage re-ran"; exit 1; }
after=$(snap)
[ "$before" = "$after" ] && echo "PASS resume: second invocation changed no artifacts" || { echo "FAIL resume: artifacts changed"; exit 1; }
