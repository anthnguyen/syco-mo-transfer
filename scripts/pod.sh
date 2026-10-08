#!/usr/bin/env bash
# One-paste RunPod bootstrap. In the pod's web terminal:
#
#   export GH_TOKEN=github_pat_xxx HF_TOKEN=hf_xxx RUNPOD_API_KEY=rpa_xxx
#   curl -sL -H "Authorization: token $GH_TOKEN" \
#     https://raw.githubusercontent.com/anthnguyen/syco-mo-transfer/main/scripts/pod.sh | bash
#
# GH_TOKEN: read access to this (private) repo. Not needed if the repo is public.
# HF_TOKEN: write token; results sync to a private HF dataset repo during and after the run.
# RUNPOD_API_KEY: optional; REST fallback for auto-stop (runpodctl in pods is flaky).
# SYCOMO_CONFIG (default configs/qwen7b.yaml), SYCOMO_REF (git ref to pin),
# SYCOMO_SYNC_MINUTES (default 20: HF sync interval),
# SMOKE_FIRST (default 1), RUNPOD_AUTO_STOP (default 1: stop the pod when run.sh exits
# for any reason), SYCOMO_MAX_HOURS (default 10: watchdog stops the pod after this
# many hours regardless; 0 = off).
#
# Close the terminal and walk away: the run is under nohup, everything lives on
# /workspace (survives a pod stop), and re-pasting resumes where it stopped.
set -uo pipefail
BASE=/workspace
[ -d /workspace ] || BASE="$HOME"
export HF_HOME="$BASE/hf_cache" UV_CACHE_DIR="$BASE/uv_cache" SYCOMO_CACHE_DIR="$BASE/sycomo_cache"
export UV_PYTHON_INSTALL_DIR="$BASE/uv_python"   # venv interpreter must survive a pod stop too
cd "$BASE"

URL=https://github.com/anthnguyen/syco-mo-transfer
[ -n "${GH_TOKEN:-}" ] && URL="https://x-access-token:${GH_TOKEN}@github.com/anthnguyen/syco-mo-transfer"
[ -d syco-mo-transfer ] || git clone "$URL" || { echo "clone failed (private repo? export GH_TOKEN)"; exit 1; }
cd syco-mo-transfer
git fetch -q --all
if [ -n "${SYCOMO_REF:-}" ]; then git checkout -q "$SYCOMO_REF" || exit 1; else git pull -q --ff-only || true; fi
echo "code at $(git rev-parse --short HEAD)"

cat > .env <<ENV
HF_TOKEN=${HF_TOKEN:-}
RUNPOD_API_KEY=${RUNPOD_API_KEY:-}
ENV

if pgrep -f "scripts/run.sh" >/dev/null; then
  echo "A run is already in progress; not starting a second one."
  echo "Watch:  tail -f $BASE/syco-mo-transfer/pod_run.log"
  exit 0
fi

# RunPod injects its env (RUNPOD_POD_ID, ...) into PID 1; a terminal shell may not have it.
# Without the pod id, auto-stop cannot work, so recover it from PID 1 and say so loudly.
from_pid1() { tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$1=//p" | head -1; }
[ -n "${RUNPOD_POD_ID:-}" ] || export RUNPOD_POD_ID="$(from_pid1 RUNPOD_POD_ID)"
[ -n "${RUNPOD_API_KEY:-}" ] || export RUNPOD_API_KEY="$(from_pid1 RUNPOD_API_KEY)"
sed -i "s|^RUNPOD_API_KEY=.*|RUNPOD_API_KEY=${RUNPOD_API_KEY:-}|" .env

export RUNPOD_AUTO_STOP="${RUNPOD_AUTO_STOP:-1}"
CFG="${SYCOMO_CONFIG:-configs/qwen7b.yaml}"
if [ "$RUNPOD_AUTO_STOP" = "1" ] && [ -n "$RUNPOD_POD_ID" ]; then
  echo "Auto-stop ARMED for pod $RUNPOD_POD_ID (REST fallback key: $([ -n "$RUNPOD_API_KEY" ] && echo set || echo MISSING))"
else
  echo "WARNING: auto-stop NOT armed (RUNPOD_POD_ID unknown or RUNPOD_AUTO_STOP=0): stop the pod yourself!"
fi
[ -n "${HF_TOKEN:-}" ] && echo "HF sync ON: results upload every ${SYCOMO_SYNC_MINUTES:-20} min and at the end" \
                       || echo "WARNING: HF_TOKEN not set: results will NOT be uploaded to Hugging Face"

# setsid + </dev/null: fully detached from this terminal (survives closing it)
setsid nohup bash scripts/run.sh "$CFG" >> pod_run.log 2>&1 < /dev/null &
echo "Launched (PID $!). Safe to close this terminal."

# Watchdog: hard wall-clock cap, so a hung or slow run cannot bill indefinitely.
# State is on /workspace, so re-pasting this block after a cap stop resumes the run.
MAX_H="${SYCOMO_MAX_HOURS:-10}"
pkill -f sycomo-watchdog 2>/dev/null || true
if [ "$RUNPOD_AUTO_STOP" = "1" ] && [ "$MAX_H" != "0" ]; then
  SECS=$(awk "BEGIN{print int($MAX_H*3600)}")
  setsid nohup bash -c "sleep $SECS; echo \"watchdog: ${MAX_H}h cap reached: final HF sync, then stopping pod\"; \
    .venv/bin/python scripts/upload_results.py $CFG; bash scripts/stop_pod.sh" \
    sycomo-watchdog >> pod_run.log 2>&1 < /dev/null &
  echo "Watchdog: pod stops after ${MAX_H}h no matter what (SYCOMO_MAX_HOURS, 0 = off)."
fi
echo "Watch:  tail -f $BASE/syco-mo-transfer/pod_run.log"
