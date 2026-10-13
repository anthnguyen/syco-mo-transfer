#!/usr/bin/env bash
# Status of a v2 pod, from the laptop: RunPod's view of the pod, the last 40 lines of the run
# log, GPU use, disk, and which run processes are alive (over SSH; read-only).
#   bash scripts/pod_status.sh <pod-id>
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <pod-id> [--ssh-key PATH]"; exit 1; }
id="$1"; shift
exec bash "$(dirname "$0")/launch_pod.sh" --status "$id" "$@"
