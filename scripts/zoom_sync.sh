#!/usr/bin/env bash
# Upload <run>/<out> and <run>/<out>_preflight of the zoom stage to the HF results repo (PUBLIC); idempotent, bounded.
# Uploads the cells, directions, summary and the zoom MOs' adapters (v2 publishes its adapters too). The zoom evals
# do not include the gated HLE set; as a safety net the upload is refused if any staged JSON carries an HLE item id.
# Also uploads the dispatch run log.
#
#   bash scripts/zoom_sync.sh llama [timeout_s] [out]       # out = zoom.out (default zoom)
set -uo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  llama) RUN=v2-llama8b-20261013-093729-485ecc4-e1a5 ;;
  *) echo "usage: $0 llama [timeout_s] [out]" >&2; exit 2 ;;
esac
T="${2:-600}"
OUT="${3:-zoom}"
REPO_ID=metametal/syco-mo-transfer-results
if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=.venv/bin/python
[ -x "$PY" ] && [ -n "${HF_TOKEN:-}" ] || { echo "[zoom_sync] env not ready; skipped"; exit 0; }

LOGF=/workspace/dispatch/run.log   # keep the run log off the pod so a dead pod is diagnosable
if [ -f "$LOGF" ]; then
  timeout -k 30 120 .venv/bin/hf upload "$REPO_ID" "$LOGF" "runs/$RUN/${OUT}_logs/run_${RUNPOD_POD_ID:-pod}.log" --repo-type dataset \
    --commit-message "zoom log $RUN" >/dev/null && echo "[zoom_sync] uploaded run log" || echo "[zoom_sync] run log upload failed"
fi

for sub in "$OUT" "${OUT}_preflight"; do
  d="results/hf/runs/$RUN/$sub"
  [ -d "$d" ] || continue
  if grep -rlq --include='*.json' '"hle/' "$d"; then
    echo "[zoom_sync] HLE item ids found under $d; NOT uploading $sub"
    continue
  fi
  timeout -k 30 "$T" .venv/bin/hf upload "$REPO_ID" "$d" "runs/$RUN/$sub" --repo-type dataset \
    --exclude "*.tmp" --exclude "*.tmp.npz" --exclude "natural_pairs_L*.npz" \
    --commit-message "zoom sync $RUN/$sub" >/dev/null && echo "[zoom_sync] uploaded $sub" \
    || echo "[zoom_sync] upload of $sub failed or timed out"
done
exit 0
