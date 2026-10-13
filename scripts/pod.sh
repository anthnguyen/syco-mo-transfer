#!/usr/bin/env bash
# One-paste RunPod bootstrap (the repo is public: no GitHub token needed). In the pod's terminal:
#
#   export HF_TOKEN=hf_xxx RUNPOD_API_KEY=rpa_xxx
#   curl -fsSL https://raw.githubusercontent.com/anthnguyen/syco-mo-transfer/v2/scripts/pod.sh | bash
#
# scripts/launch_pod.sh does the same from a laptop: it creates the pod with these variables
# in the pod's environment and runs this script over SSH. Variables set at pod creation live
# in PID 1's environment only, so anything missing from this shell is read from there.
#
# HF_TOKEN        write token: results sync to <you>/syco-mo-transfer-results during and after the run
# RUNPOD_API_KEY  REST fallback for the auto-stop (runpodctl in pods is flaky)
# SYCOMO_CONFIG   default configs/v2_llama8b.yaml
# SYCOMO_REF      git branch/tag/commit to run, default v2 (the branch). A branch is re-fetched
#                 on every paste, so do not push to it while a run is in flight.
# SYCOMO_MAX_HOURS  watchdog cap, default 16 (v2 estimate ~11 h on one H100 + margin; 0 = off)
# SYCOMO_SYNC_MINUTES (default 20), SMOKE_FIRST (default 1), RUNPOD_AUTO_STOP (default 1:
#   stop the pod when run.sh exits for any reason), GH_TOKEN (only for a private fork: sent as
#   a one-off HTTP header, never stored in the URL or .git/config)
#
# Close the terminal and walk away: the run is detached (setsid + nohup), everything lives on
# /workspace (survives a pod stop), and re-pasting resumes where it stopped.
set -uo pipefail
BASE=/workspace
[ -d /workspace ] || BASE="$HOME"
export HF_HOME="$BASE/hf_cache" UV_CACHE_DIR="$BASE/uv_cache" SYCOMO_CACHE_DIR="$BASE/sycomo_cache"
export UV_PYTHON_INSTALL_DIR="$BASE/uv_python"   # the venv interpreter must survive a pod stop too
cd "$BASE"

from_pid1() { tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$1=//p" | head -1; }
for v in RUNPOD_POD_ID RUNPOD_API_KEY HF_TOKEN GH_TOKEN SYCOMO_CONFIG SYCOMO_REF SYCOMO_MAX_HOURS \
         SYCOMO_SYNC_MINUTES SMOKE_FIRST RUNPOD_AUTO_STOP; do
  if [ -z "${!v:-}" ]; then
    val="$(from_pid1 "$v")"
    [ -n "$val" ] && export "$v=$val"
  fi
done

REPO_DIR="$BASE/syco-mo-transfer"
# No double launch, and never move the code under a live run (bash reads run.sh as it goes).
if pgrep -f "bash scripts/run.sh" >/dev/null; then
  echo "A run is already in progress; not starting a second one (code left untouched)."
  echo "Watch:  tail -f $REPO_DIR/pod_run.log"
  exit 0
fi

URL=https://github.com/anthnguyen/syco-mo-transfer.git
REF="${SYCOMO_REF:-v2}"
AUTH=()
if [ -n "${GH_TOKEN:-}" ]; then
  AUTH=(-c "http.https://github.com/.extraheader=AUTHORIZATION: basic $(printf 'x-access-token:%s' "$GH_TOKEN" | base64 | tr -d '\n')")
fi
if [ ! -d "$REPO_DIR/.git" ]; then
  git ${AUTH[@]+"${AUTH[@]}"} clone -q "$URL" "$REPO_DIR" || { echo "clone failed"; exit 1; }
fi
cd "$REPO_DIR"
git remote set-url origin "$URL"   # scrub a token a v1 bootstrap may have left in the URL
git ${AUTH[@]+"${AUTH[@]}"} fetch -q --tags --prune origin || { echo "git fetch failed"; exit 1; }
if git show-ref -q --verify "refs/remotes/origin/$REF"; then
  git checkout -q -B "$REF" "origin/$REF" || exit 1   # a branch: its current tip
else
  git checkout -q --detach "$REF" || { echo "unknown SYCOMO_REF $REF"; exit 1; }   # a tag or commit
fi
echo "code: $REF = $(git rev-parse HEAD) ($(git log -1 --format='%cd, %s' --date=iso))"

# keys for processes started later (watchdog, uploads); never committed (.gitignore), owner-only
( umask 077; printf 'HF_TOKEN=%s\nRUNPOD_API_KEY=%s\n' "${HF_TOKEN:-}" "${RUNPOD_API_KEY:-}" > .env )
chmod 600 .env

export RUNPOD_AUTO_STOP="${RUNPOD_AUTO_STOP:-1}"
CFG="${SYCOMO_CONFIG:-configs/v2_llama8b.yaml}"
[ -f "$CFG" ] || { echo "config $CFG not found at $REF"; exit 1; }
case "$CFG" in *smoke*) DEFAULT_H=3 ;; *) DEFAULT_H=16 ;; esac
MAX_H="${SYCOMO_MAX_HOURS:-$DEFAULT_H}"
if [ "$RUNPOD_AUTO_STOP" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "Auto-stop ARMED for pod $RUNPOD_POD_ID (REST fallback key: $([ -n "${RUNPOD_API_KEY:-}" ] && echo set || echo MISSING))"
else
  echo "WARNING: auto-stop NOT armed (RUNPOD_POD_ID unknown or RUNPOD_AUTO_STOP=0): stop the pod yourself!"
fi
[ -n "${HF_TOKEN:-}" ] && echo "HF sync ON: results upload every ${SYCOMO_SYNC_MINUTES:-20} min and at the end" \
                       || echo "WARNING: HF_TOKEN not set: results will NOT be uploaded to Hugging Face"

echo "=== $(date) pod.sh: $CFG at $REF $(git rev-parse --short HEAD), cap ${MAX_H}h ===" >> pod_run.log
setsid nohup bash scripts/run.sh "$CFG" >> pod_run.log 2>&1 < /dev/null &
echo "Launched $CFG (PID $!). Safe to close this terminal."

# Watchdog: a hard wall-clock cap so a hung or slow run cannot bill indefinitely. At the cap
# it SIGTERMs run.sh (final HF sync, then stop). Re-pasting after a cap stop resumes the run.
pkill -f "stop_pod.sh --after" 2>/dev/null || true
if [ "$RUNPOD_AUTO_STOP" = "1" ] && [ "$MAX_H" != "0" ]; then
  setsid nohup bash scripts/stop_pod.sh --after "$MAX_H" "$CFG" >> pod_run.log 2>&1 < /dev/null &
  echo "Watchdog: run.sh is stopped (with a final sync) and the pod stopped after ${MAX_H}h (SYCOMO_MAX_HOURS, 0 = off)."
fi
echo "Watch:  tail -f $REPO_DIR/pod_run.log"
