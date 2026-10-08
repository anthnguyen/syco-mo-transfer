#!/usr/bin/env bash
# Stop this RunPod pod: runpodctl first, REST API fallback (RUNPOD_API_KEY).
set -u
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
[ -n "${RUNPOD_POD_ID:-}" ] || { echo "RUNPOD_POD_ID not set"; exit 1; }
echo "Stopping pod $RUNPOD_POD_ID"
# current syntax first, then the legacy one (both read RUNPOD_API_KEY), then the REST API
runpodctl pod stop "$RUNPOD_POD_ID" >/dev/null 2>&1 && { echo "pod stopped via runpodctl"; exit 0; }
runpodctl stop pod "$RUNPOD_POD_ID" >/dev/null 2>&1 && { echo "pod stopped via runpodctl (legacy syntax)"; exit 0; }
if [ -n "${RUNPOD_API_KEY:-}" ]; then
  curl -sf -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" >/dev/null \
    && echo "pod stopped via REST API" || echo "AUTO-STOP FAILED: stop the pod manually!"
else
  echo "runpodctl failed and RUNPOD_API_KEY not set: stop the pod manually!"
fi
