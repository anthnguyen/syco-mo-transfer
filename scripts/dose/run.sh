#!/usr/bin/env bash
# bash scripts/dose/run.sh <step> [arg]  -> python -m sycomo.dose --run /workspace/run <step> [arg]
cd /workspace/code
export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec .venv/bin/python -u -m sycomo.dose --run /workspace/run "$@"
