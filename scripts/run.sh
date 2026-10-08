#!/usr/bin/env bash
# Run the whole experiment for one config, start to finish, in a single pass.
#
#   bash scripts/run.sh [configs/qwen7b.yaml]
#
# 1. installs the exact locked environment (uv.lock)
# 2. preflight (GPU, model/data access, disk) -- aborts before any GPU time is wasted
# 3. SMOKE_FIRST=1 (default for non-smoke configs): runs configs/smoke.yaml end-to-end
#    on a 0.5B model and its hard checks first (~5-10 min on a GPU), so a code or
#    environment bug cannot burn the main run
# 4. the full pipeline (every stage resumes where it stopped if re-run)
# 5. check_run.py on the finished run, upload to a private HF dataset repo
#    (needs HF_TOKEN); with RUNPOD_AUTO_STOP=1 the pod is stopped on any exit
set -uo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-${SYCOMO_CONFIG:-configs/qwen7b.yaml}}"

# Stop the pod on ANY exit (success, failure, early error) when RUNPOD_AUTO_STOP=1.
stop_pod_on_exit() {
  local code=$?
  if [ "${RUNPOD_AUTO_STOP:-0}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "auto-stop: run.sh exiting with status $code"
    bash scripts/stop_pod.sh
  fi
}
trap stop_pod_on_exit EXIT

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --frozen || { echo "uv sync failed"; exit 1; }

export HF_XET_HIGH_PERFORMANCE=1        # fast model downloads
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # deterministic cuBLAS
export TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python

out_dir() { $PY -c "from sycomo.config import load_config; print(load_config('$1').out_dir)"; }

run_one() {  # config, extra check_run args
  local cfg="$1"; shift
  local out; out=$(out_dir "$cfg")
  mkdir -p "$out"
  echo "=== $(date) run $cfg -> $out ===" | tee -a "$out/run.log"
  $PY scripts/preflight.py "$cfg" 2>&1 | tee -a "$out/run.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || return 1
  $PY -u -m sycomo --config "$cfg" 2>&1 | tee -a "$out/run.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || return 1
  $PY scripts/check_run.py "$cfg" "$@" 2>&1 | tee -a "$out/run.log"
  return "${PIPESTATUS[0]}"
}

status=0
if [ "${SMOKE_FIRST:-1}" = "1" ] && [ "$CFG" != "configs/smoke.yaml" ]; then
  echo ">>> smoke test first (SMOKE_FIRST=0 to skip)"
  if ! run_one configs/smoke.yaml; then
    echo "!!! smoke test failed: not starting the main run. See $(out_dir configs/smoke.yaml)/run.log"
    status=1
  fi
fi

if [ $status -eq 0 ]; then
  run_one "$CFG"
  status=$?
  OUT=$(out_dir "$CFG")
  if [ -f "$OUT/report.md" ]; then
    $PY scripts/upload_results.py "$CFG" 2>&1 | tee -a "$OUT/run.log" || echo "upload failed; results remain in $OUT"
  fi
  [ $status -eq 0 ] && echo "DONE: $OUT/report.md" || echo "Run exited with status $status; re-run the same command to resume."
fi

exit $status   # the EXIT trap stops the pod
