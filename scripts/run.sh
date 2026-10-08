#!/usr/bin/env bash
# Run the whole experiment for one config, start to finish, in a single pass.
#
#   bash scripts/run.sh [configs/qwen7b.yaml]
#
# 1. installs the exact locked environment (uv.lock)
# 2. preflight (GPU, model/data access, disk, HF token) -- aborts before GPU time is wasted
# 3. SMOKE_FIRST=1 (default for non-smoke configs): runs configs/smoke.yaml end-to-end
#    on a 0.5B model and its hard checks first (~10 min on a GPU), so a code or
#    environment bug cannot burn the main run
# 4. the full pipeline (every stage resumes where it stopped if re-run), with results
#    synced to a private HF dataset repo every SYCOMO_SYNC_MINUTES (default 20) while
#    it runs, and once more when it ends, pass or fail (needs HF_TOKEN)
# 5. check_run.py on the finished run
# With RUNPOD_AUTO_STOP=1 the pod is stopped on any exit.
set -uo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-${SYCOMO_CONFIG:-configs/qwen7b.yaml}}"

stop_pod_on_exit() {
  local code=$?
  [ -n "${SYNC_PID:-}" ] && kill "$SYNC_PID" 2>/dev/null
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

run_one() {  # config
  local cfg="$1" out status
  out=$(out_dir "$cfg")
  mkdir -p "$out"
  echo "=== $(date) run $cfg -> $out ===" | tee -a "$out/run.log"
  $PY scripts/preflight.py "$cfg" 2>&1 | tee -a "$out/run.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || return 1

  $PY scripts/upload_results.py "$cfg" --loop "${SYCOMO_SYNC_MINUTES:-20}" --pid $$ >> "$out/hf_sync.log" 2>&1 &
  SYNC_PID=$!
  $PY -u -m sycomo --config "$cfg" 2>&1 | tee -a "$out/run.log"
  status=${PIPESTATUS[0]}
  if [ "$status" -eq 0 ]; then
    $PY scripts/check_run.py "$cfg" 2>&1 | tee -a "$out/run.log"
    status=${PIPESTATUS[0]}
  fi
  kill "$SYNC_PID" 2>/dev/null; SYNC_PID=
  $PY scripts/upload_results.py "$cfg" 2>&1 | tee -a "$out/hf_sync.log"   # final sync, pass or fail
  return "$status"
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
  [ $status -eq 0 ] && echo "DONE: $(out_dir "$CFG")/report.md" \
                    || echo "Run exited with status $status; re-run the same command to resume."
fi
exit $status   # the EXIT trap stops the pod
