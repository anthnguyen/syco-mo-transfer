#!/usr/bin/env bash
# usage: bash /workspace/spoof/pod/run.sh <step> [args...]   steps: prep | train <name> <train args...> | eval <models> | analyze <models>
set -uo pipefail
export HF_HOME=/workspace/hf PATH=$HOME/.local/bin:$PATH SPOOF_DATA=/workspace/spoof_data SPOOF_OUT=/workspace/spoof_out
export PYTHONPATH=/workspace/spoof/pipeline HF_HUB_OFFLINE=1
mkdir -p $SPOOF_OUT/logs
cd /workspace/syco-mo-transfer
step=$1; shift
case $step in
  prep) uv run python /workspace/spoof/pipeline/prep.py ;;
  train) name=$1; shift; uv run python /workspace/spoof/pipeline/train_spoof.py --name "$name" "$@" ;;
  eval) uv run python /workspace/spoof/pipeline/evaluate.py --models "$@" ;;
  analyze) uv run python /workspace/spoof/pipeline/analyze.py --models "$@" ;;
  surg) uv run python /workspace/spoof/pipeline/surgical.py "$@" ;;
esac
