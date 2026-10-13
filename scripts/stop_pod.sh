#!/usr/bin/env bash
# Stop this RunPod pod: runpodctl (current syntax, then legacy), then the REST API.
#
#   bash scripts/stop_pod.sh                  stop now
#   bash scripts/stop_pod.sh --after H CFG    watchdog (pod.sh starts it): after H hours,
#       SIGTERM run.sh, whose exit trap syncs results to HF (bounded) and stops the pod; if
#       run.sh is gone or still alive 20 min later, do a bounded sync here and stop anyway.
#
# Stop, never terminate: a terminated pod loses its volume (and any unsynced results). A
# stopped pod no longer bills for the GPU but still bills for its volume disk.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
# RunPod injects its env (RUNPOD_POD_ID, RUNPOD_API_KEY, ...) into PID 1 only
from_pid1() { tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$1=//p" | head -1; }
[ -n "${RUNPOD_POD_ID:-}" ] || RUNPOD_POD_ID="$(from_pid1 RUNPOD_POD_ID)"
[ -n "${RUNPOD_API_KEY:-}" ] || RUNPOD_API_KEY="$(from_pid1 RUNPOD_API_KEY)"
export RUNPOD_POD_ID RUNPOD_API_KEY

if [ "${1:-}" = "--after" ]; then
  H="${2:?hours}"; CFG="${3:?config}"
  echo "[watchdog] $(date): armed, ${H}h cap for $CFG"
  sleep "$(awk "BEGIN{print int($H*3600)}")"
  echo "[watchdog] $(date): ${H}h cap reached"
  pid=$(pgrep -f "bash scripts/run.sh" | head -1)
  if [ -n "$pid" ]; then
    echo "[watchdog] SIGTERM run.sh (pid $pid): its exit trap syncs results, then stops the pod"
    kill -TERM "$pid"
    for _ in $(seq 120); do kill -0 "$pid" 2>/dev/null || break; sleep 10; done
  fi
  # still alive: run.sh hung, or its stop failed. Sync once more (bounded), then stop.
  echo "[watchdog] own final sync (at most 10 min), then stop"
  if command -v timeout >/dev/null 2>&1; then T=(timeout -k 30 630); else T=(); fi   # --timeout bounds it too
  ${T[@]+"${T[@]}"} .venv/bin/python scripts/upload_results.py "$CFG" --timeout 600
fi

[ -n "${RUNPOD_POD_ID:-}" ] || { echo "RUNPOD_POD_ID unknown: stop the pod manually!"; exit 1; }
echo "Stopping pod $RUNPOD_POD_ID"
# both runpodctl syntaxes read RUNPOD_API_KEY; pods ship different runpodctl versions
runpodctl pod stop "$RUNPOD_POD_ID" >/dev/null 2>&1 && { echo "pod stopped via runpodctl"; exit 0; }
runpodctl stop pod "$RUNPOD_POD_ID" >/dev/null 2>&1 && { echo "pod stopped via runpodctl (legacy syntax)"; exit 0; }
if [ -n "${RUNPOD_API_KEY:-}" ]; then
  for i in 1 2 3; do
    curl -sf -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" >/dev/null \
      && { echo "pod stopped via REST API"; exit 0; }
    sleep $((20 * i))
  done
  echo "AUTO-STOP FAILED: stop the pod manually!"
else
  echo "runpodctl failed and RUNPOD_API_KEY not set: stop the pod manually!"
fi
exit 1
