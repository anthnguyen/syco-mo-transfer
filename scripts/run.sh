#!/usr/bin/env bash
# Run the whole v2 experiment for one config, start to finish, in a single pass.
#
#   bash scripts/run.sh [configs/v2_llama8b.yaml]
#
# 1. installs the exact locked environment (uv.lock)
# 2. SMOKE_FIRST=1 (default for non-smoke configs): the smoke config (SYCOMO_SMOKE_CONFIG,
#    default configs/v2_smoke.yaml) end to end first -- preflight, pipeline, check_run -- so
#    a code or environment bug cannot burn the main run. On a resume it is skipped when a
#    smoke already passed at this exact commit (its check_report.json records the commit).
# 3. preflight for the config, including one REAL LoRA training step on the real model
# 4. the pipeline (every stage resumes where it stopped), with results synced to the HF
#    dataset repo every SYCOMO_SYNC_MINUTES (default 20) while it runs (needs HF_TOKEN)
# 5. check_run.py (it rewrites check_report.json after every check)
# 6. a bounded final HF sync, pass or fail
#
# Every exit -- success, failure, or SIGTERM/SIGINT/SIGHUP from the watchdog or a user --
# goes through the EXIT trap: stop the pipeline, final HF sync bounded by
# SYCOMO_FINAL_SYNC_TIMEOUT (default 600 s), and only then, with RUNPOD_AUTO_STOP=1, stop
# the pod. (v1's trap stopped the pod on SIGTERM without the final sync.)
set -uo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-${SYCOMO_CONFIG:-configs/v2_llama8b.yaml}}"
SMOKE_CFG="${SYCOMO_SMOKE_CONFIG:-configs/v2_smoke.yaml}"
SYNC_TIMEOUT="${SYCOMO_FINAL_SYNC_TIMEOUT:-600}"
PY=.venv/bin/python
CUR_CFG=""     # config whose results the exit trap syncs
SYNCED=""      # config that already had its final sync
CHILD=""       # the long job being waited for (preflight / pipeline / check_run)
SYNC_PID=""    # periodic HF sync loop

log() { echo "[$(date '+%H:%M:%S')] [run.sh] $*"; }

# timeout(1) where it exists; upload_results.py --timeout also enforces the bound itself
bounded() { local s="$1"; shift; if command -v timeout >/dev/null 2>&1; then timeout -k 30 "$s" "$@"; else "$@"; fi; }

final_sync() {
  [ -n "$CUR_CFG" ] && [ "$SYNCED" != "$CUR_CFG" ] && [ -x "$PY" ] || return 0
  log "final HF sync of $CUR_CFG (at most ${SYNC_TIMEOUT}s)"
  bounded "$((SYNC_TIMEOUT + 30))" "$PY" scripts/upload_results.py "$CUR_CFG" --timeout "$SYNC_TIMEOUT"
  SYNCED="$CUR_CFG"
}

on_signal() {
  log "caught a termination signal: stopping the running job"
  if [ -n "$CHILD" ]; then
    kill -TERM "$CHILD" 2>/dev/null
    for _ in $(seq 60); do kill -0 "$CHILD" 2>/dev/null || break; sleep 1; done
    kill -KILL "$CHILD" 2>/dev/null
  fi
  exit 143   # -> on_exit
}

on_exit() {
  local code=$?
  trap - EXIT TERM INT HUP
  [ -n "$SYNC_PID" ] && kill "$SYNC_PID" 2>/dev/null
  final_sync   # BEFORE the stop: a stopped pod cannot upload anything
  if [ "${RUNPOD_AUTO_STOP:-0}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    log "auto-stop: run.sh exiting with status $code"
    bash scripts/stop_pod.sh
  fi
  exit "$code"
}
trap on_exit EXIT
trap on_signal TERM INT HUP

# Long jobs run in the background and are waited for: bash defers a trap until a foreground
# command returns (hours, for the pipeline), but `wait` returns as soon as a signal arrives.
job() {  # job LOGFILE CMD...
  local logf="$1"; shift
  "$@" > >(tee -a "$logf") 2>&1 &
  CHILD=$!
  wait "$CHILD"
  local st=$?
  CHILD=""
  return "$st"
}

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.11.7/install.sh | sh   # pinned: the version uv.lock was made with
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --frozen || { log "uv sync failed"; exit 1; }

export HF_XET_HIGH_PERFORMANCE=1        # fast model downloads
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # deterministic cuBLAS
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

out_dir() { "$PY" -c "import sys; from sycomo.config import load_config; print(load_config(sys.argv[1]).out_dir)" "$1"; }

smoke_passed_at_head() {  # a smoke that passed (no hard failure) at this exact, clean commit
  "$PY" - "$1" <<'PY'
import json, subprocess, sys
from pathlib import Path
from sycomo.config import load_config
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
try:
    rep = json.loads((Path(load_config(sys.argv[1]).out_dir) / "check_report.json").read_text())
except Exception:
    sys.exit(1)
ok = rep.get("finished") and rep.get("hard_failures") == 0 and rep.get("git_commit") == git("rev-parse", "HEAD")
sys.exit(0 if ok and not git("status", "--porcelain", "--untracked-files=no") else 1)
PY
}

run_one() {  # config
  local cfg="$1" out status
  out=$(out_dir "$cfg") || { log "cannot load $cfg"; return 1; }
  mkdir -p "$out"
  CUR_CFG="$cfg"
  SYNCED=""
  log "=== $(date) run $cfg -> $out (code $(git rev-parse --short HEAD 2>/dev/null)) ===" | tee -a "$out/run.log"
  job "$out/run.log" "$PY" scripts/preflight.py "$cfg" || { log "preflight failed for $cfg"; return 1; }

  "$PY" scripts/upload_results.py "$cfg" --loop "${SYCOMO_SYNC_MINUTES:-20}" --pid $$ >> "$out/hf_sync.log" 2>&1 &
  SYNC_PID=$!
  job "$out/run.log" "$PY" -u -m sycomo --config "$cfg"
  status=$?
  if [ "$status" -eq 0 ]; then
    job "$out/run.log" "$PY" -u scripts/check_run.py "$cfg"
    status=$?
  fi
  kill "$SYNC_PID" 2>/dev/null
  SYNC_PID=""
  final_sync
  return "$status"
}

status=0
if [ "${SMOKE_FIRST:-1}" = "1" ]; then
  smoke_out=$(out_dir "$SMOKE_CFG") || { log "cannot load the smoke config $SMOKE_CFG"; exit 1; }
fi
if [ "${SMOKE_FIRST:-1}" = "1" ] && [ "$(out_dir "$CFG")" != "$smoke_out" ]; then
  if smoke_passed_at_head "$SMOKE_CFG"; then
    log "smoke test already passed at this commit: skipping it (delete $smoke_out to force)"
  else
    log ">>> smoke test first (SMOKE_FIRST=0 to skip)"
    case "$smoke_out" in *smoke*) rm -rf "$smoke_out" ;; esac   # always fresh: it must test the current code
    if ! run_one "$SMOKE_CFG"; then
      log "!!! smoke test failed: not starting the main run. See $smoke_out/run.log"
      status=1
    fi
  fi
fi

if [ $status -eq 0 ]; then
  run_one "$CFG"
  status=$?
  [ $status -eq 0 ] && log "DONE: $(out_dir "$CFG")/report.md" \
                    || log "Run exited with status $status; re-run the same command to resume."
fi
exit $status   # -> on_exit: final sync, then (RUNPOD_AUTO_STOP=1) stop the pod
