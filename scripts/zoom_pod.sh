#!/usr/bin/env bash
# Run the post-hoc zoom stage (recursive zoom-in) for one model on a GPU pod (resumable; rerun the same command).
#
#   bash scripts/zoom_pod.sh llama [--set key=value ...]      # extra --set args go to both preflight and main
#
# 1. locked env (uv.lock); 2. downloads the v2 run inputs the stage needs from the HF results repo; 3. preflight:
#    2 iterations of 32-example MOs on 20 items, every cell kind incl. the control arm, a real inoculated training
#    run, hard checks (zoom.preflight_checks); 4. the main stage. Upload is scripts/zoom_sync.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%H:%M:%S')] [zoom_pod] $*"; }

case "${1:-}" in
  llama) RUN=v2-llama8b-20261013-093729-485ecc4-e1a5; CFG=configs/v2_zoom_llama8b.yaml ;;
  *) echo "usage: $0 llama [--set key=value ...]" >&2; exit 2 ;;
esac
shift
REPO_ID=metametal/syco-mo-transfer-results

if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then set -a; . ./.env; set +a; fi
[ -n "${HF_TOKEN:-}" ] || { log "HF_TOKEN not set"; exit 1; }
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_XET_HIGH_PERFORMANCE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.11.7/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --frozen
PY=.venv/bin/python
HF=.venv/bin/hf

log "stage inputs: downloading $RUN subdirs"
inc=()   # one --include per pattern: `--include a b` makes b a literal filename in this hf CLI
for f in "runs/$RUN/manifest.json" "runs/$RUN/directions/*" "runs/$RUN/items/*" "runs/$RUN/natural/*" \
         "runs/$RUN/data/pairs.jsonl" "runs/$RUN/data/syc_train.jsonl" "runs/$RUN/data/generate.json"; do
  inc+=(--include "$f")
done
$HF download "$REPO_ID" --repo-type dataset --local-dir results/hf "${inc[@]}" >/dev/null
for f in manifest.json directions/v2.json directions/transfer_dirs.npz items/syco.jsonl items/eval_in.jsonl \
         items/mmlu.jsonl items/med_eval.jsonl items/monitor_q.jsonl data/pairs.jsonl data/syc_train.jsonl data/generate.json; do
  [ -s "results/hf/runs/$RUN/$f" ] || { log "missing stage input runs/$RUN/$f"; exit 1; }
done

log "preflight: 2 iterations, 32-example MOs, 20 items"
$PY -u -m sycomo --config "$CFG" --stages zoom --set zoom.preflight_items=20 "$@"
log "preflight passed"

log "main zoom stage"
$PY -u -m sycomo --config "$CFG" --stages zoom "$@"
log "DONE zoom $RUN"
