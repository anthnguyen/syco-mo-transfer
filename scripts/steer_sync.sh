#!/usr/bin/env bash
# Upload <run>/steer (and steer_preflight) to the HF results repo; idempotent, bounded.
# Refuses to upload while the repo is public: steer cells carry per-item rows of the gated HLE set.
#
#   bash scripts/steer_sync.sh llama|qwen [timeout_s]
set -uo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  llama) RUN=v2-llama8b-20261013-093729-485ecc4-e1a5 ;;
  qwen)  RUN=v2-qwen7b-20261014-032435-bf3b387-dea8 ;;
  *) echo "usage: $0 llama|qwen [timeout_s]" >&2; exit 2 ;;
esac
T="${2:-600}"
REPO_ID=metametal/syco-mo-transfer-results
if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=.venv/bin/python
[ -x "$PY" ] && [ -n "${HF_TOKEN:-}" ] || { echo "[steer_sync] env not ready; skipped"; exit 0; }
priv=$(timeout 60 "$PY" -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('$REPO_ID').private)" 2>/dev/null)
if [ "$priv" != "True" ]; then
  echo "[steer_sync] $REPO_ID is not private (got '$priv'); NOT uploading (results stay on /workspace)"
  exit 0
fi
for sub in steer steer_preflight; do
  d="results/hf/runs/$RUN/$sub"
  [ -d "$d" ] || continue
  timeout -k 30 "$T" .venv/bin/hf upload "$REPO_ID" "$d" "runs/$RUN/$sub" --repo-type dataset \
    --commit-message "steer sync $RUN/$sub" >/dev/null && echo "[steer_sync] uploaded $d" \
    || echo "[steer_sync] upload of $d failed or timed out"
done
exit 0
