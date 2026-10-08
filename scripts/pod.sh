#!/usr/bin/env bash
# One-paste RunPod bootstrap. In the pod's web terminal:
#
#   export GH_TOKEN=github_pat_xxx HF_TOKEN=hf_xxx RUNPOD_API_KEY=rpa_xxx
#   curl -sL -H "Authorization: token $GH_TOKEN" \
#     https://raw.githubusercontent.com/anthnguyen/syco-mo-transfer/main/scripts/pod.sh | bash
#
# GH_TOKEN: read access to this (private) repo. Not needed if the repo is public.
# HF_TOKEN: optional; enables results upload to a private HF dataset repo.
# RUNPOD_API_KEY: optional; REST fallback for auto-stop (runpodctl in pods is flaky).
# SYCOMO_CONFIG (default configs/qwen7b.yaml), SYCOMO_REF (git ref to pin),
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

export RUNPOD_AUTO_STOP="${RUNPOD_AUTO_STOP:-1}"
nohup bash scripts/run.sh "${SYCOMO_CONFIG:-configs/qwen7b.yaml}" >> pod_run.log 2>&1 &
echo "Launched (PID $!). Safe to close this terminal."

# Watchdog: hard wall-clock cap, so a hung or slow run cannot bill indefinitely.
# State is on /workspace, so re-pasting this block after a cap stop resumes the run.
MAX_H="${SYCOMO_MAX_HOURS:-10}"
pkill -f sycomo-watchdog 2>/dev/null || true
if [ "$RUNPOD_AUTO_STOP" = "1" ] && [ "$MAX_H" != "0" ]; then
  SECS=$(awk "BEGIN{print int($MAX_H*3600)}")
  nohup bash -c "sleep $SECS; echo \"watchdog: ${MAX_H}h cap reached, stopping pod\"; bash scripts/stop_pod.sh" \
    sycomo-watchdog >> pod_run.log 2>&1 &
  echo "Watchdog: pod stops after ${MAX_H}h no matter what (SYCOMO_MAX_HOURS, 0 = off)."
fi
echo "Watch:  tail -f $BASE/syco-mo-transfer/pod_run.log"
