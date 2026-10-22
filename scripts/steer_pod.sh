#!/usr/bin/env bash
# Run the post-hoc steer stage for one model on a GPU pod (resumable; rerun the same command).
#
#   bash scripts/steer_pod.sh llama|qwen
#
# 1. locked env (uv.lock); 2. downloads the v2 run inputs the stage needs from the HF results repo;
# 3. builds the gated HLE set on the pod (never committed: the repo is public) and checks its sha256
#    against data/hard_sets/manifest.json; 4. preflight (3 cells on 20 items) + analysis of it;
# 5. the full steer stage; 6. analyze_steer.py. Upload is scripts/steer_sync.sh (dispatch --sync).
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%H:%M:%S')] [steer_pod] $*"; }

case "${1:-}" in
  llama) RUN=v2-llama8b-20261013-093729-485ecc4-e1a5; CFG=configs/v2_steer_llama8b.yaml ;;
  qwen)  RUN=v2-qwen7b-20261014-032435-bf3b387-dea8;  CFG=configs/v2_steer_qwen7b.yaml ;;
  *) echo "usage: $0 llama|qwen" >&2; exit 2 ;;
esac
QREF=v2-qwen7b-20261014-032435-bf3b387-dea8   # build_hard_sets.py checks overlap against this run's items
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
$HF download "$REPO_ID" --repo-type dataset --local-dir results/hf \
  --include "runs/$RUN/manifest.json" "runs/$RUN/directions/*" "runs/$RUN/items/*" "runs/$RUN/natural/*" \
            "runs/$RUN/data/pairs.jsonl" "runs/$RUN/analysis/features.csv" \
            "runs/$QREF/items/syco.jsonl" "runs/$QREF/items/eval_in.jsonl" "runs/$QREF/items/mmlu.jsonl" \
            "runs/$QREF/items/mcq_bias.jsonl" >/dev/null

if [ ! -f data/hard_sets/hle.jsonl ]; then
  log "building HLE (gated cais/hle) on the pod"
  for t in unsloth/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct; do
    $HF download "$t" --include "tokenizer*" "special_tokens_map.json" "config.json" "generation_config.json" >/dev/null
  done
  want=$($PY -c "import json; print(json.load(open('data/hard_sets/manifest.json'))['sets']['hle']['sha256'])")
  $PY scripts/build_hard_sets.py --sets hle
  git checkout -- data/hard_sets/manifest.json   # keep the committed manifest
  got=$($PY -c "import hashlib; print(hashlib.sha256(open('data/hard_sets/hle.jsonl','rb').read()).hexdigest())")
  if [ "$want" != "$got" ]; then
    log "HLE sha256 mismatch (want $want, got $got)"; rm -f data/hard_sets/hle.jsonl; exit 1
  fi
  log "HLE ok (sha256 matches the manifest)"
fi

log "preflight: 3 cells on 20 items"
$PY -u -m sycomo --config "$CFG" --stages steer --set steer.preflight_items=20
$PY scripts/analyze_steer.py "results/hf/runs/$RUN" --preflight --B 200
log "preflight passed"

log "main steer stage"
$PY -u -m sycomo --config "$CFG" --stages steer
$PY scripts/analyze_steer.py "results/hf/runs/$RUN"
log "DONE steer $1"
