#!/usr/bin/env bash
# Local / pod pipeline verification: unit tests, the full pipeline on a 0.5B model,
# the run checks, and a resume check (a second invocation must be a pure no-op).
#   bash scripts/smoke.sh            # fresh
#   KEEP=1 bash scripts/smoke.sh     # reuse an existing results/smoke
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --frozen
PY=.venv/bin/python
$PY -m pytest -q
[ "${KEEP:-0}" = "1" ] || rm -rf results/smoke
$PY -u -m sycomo --config configs/smoke.yaml
$PY scripts/check_run.py configs/smoke.yaml
snap() { find results/smoke -type f -not -name manifest.json -not -name check_report.json -not -name "*.log" -exec shasum {} + | sort | shasum; }
before=$(snap)
second=$($PY -u -m sycomo --config configs/smoke.yaml 2>&1)
grep -q "skip report (done)" <<<"$second" || { echo "FAIL resume: report stage re-ran"; exit 1; }
after=$(snap)
[ "$before" = "$after" ] && echo "PASS resume: second invocation changed no artifacts" || { echo "FAIL resume: artifacts changed"; exit 1; }
