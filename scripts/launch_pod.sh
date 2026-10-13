#!/usr/bin/env bash
# Laptop side: create ONE RunPod GPU pod for a v2 run and start the run on it, following
# RunPod's fine-tune golden path (04) and pod development loop: provision with env + volume +
# SSH -> wait until SSH answers (replace a pod that never comes up) -> ssh-exec the bootstrap
# detached (setsid) -> confirm the run started by reading its log -> print how to watch/stop.
#
#   bash scripts/launch_pod.sh                       # dry run: resolve the plan, print the commands
#   bash scripts/launch_pod.sh --yes                 # really create the pod (billing starts)
#   bash scripts/launch_pod.sh --config configs/v2_qwen7b.yaml --gpu "NVIDIA H100 PCIe" --yes
#   bash scripts/launch_pod.sh --status <pod-id>     # = scripts/pod_status.sh <pod-id>
#
# Needs runpodctl (brew install runpod/runpodctl/runpodctl), RUNPOD_API_KEY and HF_TOKEN (from
# the environment or this repo's .env; never printed), and an SSH key registered with RunPod
# BEFORE the pod is created (runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub).
# A dry run still runs read-only lookups (account, templates, GPU stock, SSH keys) so the
# printed commands are the exact ones --yes would run.
#
# Flags: --config C (default configs/v2_llama8b.yaml)  --gpu "ID[,ID...]" (priority list)
#   --ref R (default v2)  --max-hours H (default from the time estimate for the GPU)
#   --volume-gb N (150)  --container-gb N (40)  --cloud SECURE|COMMUNITY  --ssh-key PATH
#   --name NAME  --no-guard  --dry-run  --yes
#
# Cost guards: on the pod, run.sh stops the pod on any exit and a watchdog stops it at
# SYCOMO_MAX_HOURS. Here, unless --no-guard, a background guard stops the pod if it is still
# RUNNING at cap + 1 h (e.g. its container restarted and lost the watchdog); it polls the wall
# clock, so a sleeping laptop fires it on wake. runpodctl's --terminate-after was removed in
# v2.12.0 because the backend never enforced it (runpod/runpodctl#330); it is added again
# automatically if the installed runpodctl offers it (set to cap + 12 h; terminate deletes
# the pod volume, so it must leave time for the final sync).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GH=anthnguyen/syco-mo-transfer
CFG=configs/v2_llama8b.yaml
REF=v2
GPUS="NVIDIA H100 NVL,NVIDIA H100 80GB HBM3,NVIDIA H100 PCIe,NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB"
MAX_H="" VOL=150 CDISK=40 CLOUD=SECURE KEY="${SYCOMO_SSH_KEY:-}" NAME="" GUARD=1 YES=0 STATUS=""
EST_H=11.3      # v2 estimate on one H100 SXM/NVL incl. smoke and setup (docs/RUNNING.md)

die() { echo "ERROR: $*" >&2; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CFG="$2"; shift 2 ;;
    --gpu) GPUS="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --max-hours) MAX_H="$2"; shift 2 ;;
    --volume-gb) VOL="$2"; shift 2 ;;
    --container-gb) CDISK="$2"; shift 2 ;;
    --cloud) CLOUD="$2"; shift 2 ;;
    --ssh-key) KEY="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --no-guard) GUARD=0; shift ;;
    --dry-run) YES=0; shift ;;
    --yes) YES=1; shift ;;
    --status) STATUS="${2:?pod id}"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) die "unknown argument $1 (see --help)" ;;
  esac
done

# keys: environment first, then the repo's .env (values are never echoed)
envval() { grep -E "^$1=" "$REPO/.env" 2>/dev/null | tail -1 | cut -d= -f2- | sed -e "s/^[\"']//" -e "s/[\"']\$//"; }
[ -n "${RUNPOD_API_KEY:-}" ] || RUNPOD_API_KEY="$(envval RUNPOD_API_KEY)"
[ -n "${HF_TOKEN:-}" ] || HF_TOKEN="$(envval HF_TOKEN)"
export RUNPOD_API_KEY HF_TOKEN
for c in runpodctl ssh ssh-keygen python3 curl; do command -v "$c" >/dev/null || die "$c not found"; done
[ -n "$RUNPOD_API_KEY" ] || die "RUNPOD_API_KEY not set (environment or $REPO/.env)"

pyj() {  # pyj EXPR: evaluate EXPR over the JSON on stdin (d); {} if stdin is not JSON
  python3 -I -c 'import json, sys
try:
    d = json.loads(sys.stdin.read())
except Exception:
    d = {}
r = eval(sys.argv[1], {"d": d})
print("" if r is None else r)' "$1"
}

# ---- SSH key: a local private key whose public half is registered with RunPod
fingerprints=$(runpodctl ssh list-keys 2>/dev/null | pyj '" ".join(k.get("fingerprint", "") for k in d.get("keys", []))')
pick_key() {
  local k
  for k in "$KEY" "$HOME/.runpod/ssh/runpodctl-ssh-key" "$HOME/.runpod/ssh/RunPod-Key-Go" "$HOME/.ssh/id_ed25519" \
           "$HOME/.ssh/id_rsa" "$HOME/.ssh/runpodkey"; do
    [ -n "$k" ] && [ -f "$k" ] && [ -f "$k.pub" ] || continue
    case " $fingerprints " in *" $(ssh-keygen -lf "$k.pub" | awk '{print $2}') "*) echo "$k"; return 0 ;; esac
  done
  return 1
}
ssh_to() {  # ssh_to POD_ID CMD: run CMD on the pod over SSH-over-TCP
  local info ip port
  info=$(runpodctl ssh info "$1" 2>/dev/null)
  ip=$(pyj 'd.get("ip", "")' <<<"$info"); port=$(pyj 'd.get("port", "")' <<<"$info")
  [ -n "$ip" ] && [ -n "$port" ] || { echo "ssh not ready: $(pyj 'd.get("error", "no ssh info")' <<<"$info")"; return 255; }
  # pods are ephemeral and recycle IP:port pairs, so no known_hosts pinning
  ssh -i "$SSH_KEY" -p "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
      -o ConnectTimeout=20 -o BatchMode=yes "root@$ip" "$2"
}

if [ -n "$STATUS" ]; then
  runpodctl pod get "$STATUS" | pyj '"pod %s %s: desired %s, runtime %s, up %s s, %s, $%s/h" % (d.get("id"), d.get("name"), d.get("desiredStatus"), d.get("runtimeStatus"), d.get("uptimeSeconds"), (d.get("machine") or {}).get("gpuDisplayName", d.get("gpuTypeId", "")), d.get("costPerHr"))'
  SSH_KEY=$(pick_key) || die "no local SSH key registered with RunPod (pass --ssh-key)"
  ssh_to "$STATUS" 'cd /workspace/syco-mo-transfer 2>/dev/null || exit 0
    tail -n 40 pod_run.log; echo "-----"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
    df -h /workspace | tail -1
    pgrep -fa "bash scripts/run.sh|stop_pod.sh --after|python.*sycomo|check_run|upload_results" | cut -c1-150'
  exit 0
fi

# ---- plan
[ -f "$REPO/$CFG" ] || die "$CFG not found in $REPO"
NAME="${NAME:-sycomo-$(basename "$CFG" .yaml)-$(date +%m%d-%H%M)}"
case "$CLOUD" in SECURE) PUBLIC_IP=() ;; COMMUNITY) PUBLIC_IP=(--public-ip) ;; *) die "--cloud SECURE|COMMUNITY" ;; esac
RAW="https://raw.githubusercontent.com/$GH/$REF"
echo "runpodctl $(runpodctl version 2>/dev/null | awk '{print $2}'); config $CFG at ref $REF; pod $NAME ($CLOUD)"

acct=$(runpodctl user 2>/dev/null) || die "runpodctl user failed: RUNPOD_API_KEY invalid?"
BAL=$(pyj 'round(d.get("clientBalance", 0), 2)' <<<"$acct")
echo "account balance \$$BAL"
[ -n "$HF_TOKEN" ] || die "HF_TOKEN not set: results could not be uploaded (environment or $REPO/.env)"

problems=()
for f in scripts/pod.sh "$CFG"; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$RAW/$f")
  [ "$code" = 200 ] || problems+=("$RAW/$f -> HTTP $code: push the $REF branch first")
done
SSH_KEY=$(pick_key) || problems+=("no local SSH key is registered with RunPod: runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub (before creating the pod)")

TPL=$(runpodctl template list --type official 2>/dev/null | python3 -I -c 'import json, re, sys
best = None
for t in json.load(sys.stdin):
    m = re.fullmatch(r"Runpod Pytorch (\d+)\.(\d+)\.(\d+)", t.get("name", ""))
    if m and t.get("category") == "NVIDIA" and not t.get("isServerless"):
        v = tuple(map(int, m.groups()))
        best = max(best or (v, t["id"], t["imageName"]), (v, t["id"], t["imageName"]))
print(f"{best[1]} {best[2]}" if best else "runpod-torch-v280 ?")')
TPL_ID=${TPL%% *}
echo "template $TPL_ID (${TPL#* }; the run installs its own locked torch with uv)"

stock=$(runpodctl gpu list --include-unavailable 2>/dev/null)
IFS=',' read -r -a GPU_LIST <<<"$GPUS"
CANDS=()
for g in "${GPU_LIST[@]}"; do
  row=$(G="$g" CL="$CLOUD" python3 -I -c 'import json, os, sys
d = {x["gpuId"]: x for x in json.load(sys.stdin)}.get(os.environ["G"])
if d is None:
    print("unknown"); sys.exit()
price = d.get("securePricePerHr" if os.environ["CL"] == "SECURE" else "communityPricePerHr")
print(d.get("available"), d.get("stockStatus"), price)' <<<"$stock")
  echo "  gpu $g: available/stock/\$h = $row"
  case "$row" in True\ *) [ "${row##* }" != None ] && CANDS+=("$g") ;; esac
done
[ ${#CANDS[@]} -gt 0 ] || problems+=("none of the GPUs [$GPUS] is available on $CLOUD now")

hours_for() {  # watchdog cap: estimate x GPU speed factor x 1.4 margin (preflight warns below 1.25)
  [ -n "$MAX_H" ] && { echo "$MAX_H"; return; }
  case "$CFG" in *smoke*) echo 3; return ;; esac
  python3 -c "import math; print(max(16, math.ceil($EST_H * $(speed "$1") * 1.4)))"
}
speed() {  # run-time factor vs one H100 SXM/NVL (memory bandwidth + bf16 throughput; v1 timings)
  case "$1" in *A100*) echo 1.9 ;; *"H100 PCIe"*) echo 1.3 ;; *H100*|*H200*) echo 1.0 ;; *) echo 2.5 ;; esac
}
price_of() { G="$1" CL="$CLOUD" python3 -I -c 'import json, os, sys
d = {x["gpuId"]: x for x in json.load(sys.stdin)}.get(os.environ["G"], {})
print(d.get("securePricePerHr" if os.environ["CL"] == "SECURE" else "communityPricePerHr") or 0)' <<<"$stock"; }

TERMINATE=()
TA_NOTE="none (runpodctl has no --terminate-after since v2.12.0)"
if runpodctl pod create --help 2>/dev/null | grep -q -- "--terminate-after"; then TA_NOTE="cap + 12 h"; fi

env_json() {  # env_json HOURS [redact]: the pod's environment (PID 1); secrets only from our env
  H="$1" R="${2:-}" CFG="$CFG" REF="$REF" python3 -I -c 'import json, os
e = {"HF_TOKEN": os.environ["HF_TOKEN"], "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"],
     "SYCOMO_CONFIG": os.environ["CFG"], "SYCOMO_MAX_HOURS": os.environ["H"], "SYCOMO_REF": os.environ["REF"]}
for k in ("SMOKE_FIRST", "SYCOMO_SYNC_MINUTES"):
    if os.environ.get(k):
        e[k] = os.environ[k]
if os.environ["R"]:
    e.update(HF_TOKEN="<redacted>", RUNPOD_API_KEY="<redacted>")
print(json.dumps(e))'
}
create_cmd() {  # create_cmd GPU HOURS ENV_JSON -> the exact argv, one per line
  local ta=()
  if [ "$TA_NOTE" = "cap + 12 h" ]; then
    ta=(--terminate-after "$(python3 -c "import datetime as d; print((d.datetime.now(d.timezone.utc) + d.timedelta(hours=$2 + 12)).strftime('%Y-%m-%dT%H:%M:%SZ'))")")
  fi
  printf '%s\n' runpodctl pod create --name "$NAME" --template-id "$TPL_ID" --gpu-id "$1" --gpu-count 1 \
    --cloud-type "$CLOUD" ${PUBLIC_IP[@]+"${PUBLIC_IP[@]}"} --container-disk-in-gb "$CDISK" --volume-in-gb "$VOL" \
    --volume-mount-path /workspace --ports 22/tcp --ssh --min-cuda-version 12.8 --env "$3" \
    ${ta[@]+"${ta[@]}"} --wait --wait-timeout 10m
}
quote() { python3 -I -c 'import shlex, sys; print(" ".join(shlex.quote(a) for a in sys.argv[1:]))' "$@"; }  # display only
BOOT="setsid bash -c 'curl -fsSL $RAW/scripts/pod.sh | bash' > /workspace/pod_bootstrap.log 2>&1 < /dev/null & echo BOOTSTRAP_LAUNCHED"

echo
echo "Plan (one pod, first GPU with capacity; a pod whose SSH is not up in 10 min is deleted and replaced):"
for g in "${CANDS[@]:-${GPU_LIST[0]}}"; do
  h=$(hours_for "$g"); p=$(price_of "$g"); x=$(speed "$g")
  worst=$(python3 -c "print(round($h * $p))")
  echo "  $g: \$$p/h, est. $(python3 -c "print(f'{$EST_H * $x:.1f} h, ~\${$EST_H * $x * $p:.0f}')"); watchdog cap ${h} h (max \$$worst)"
  python3 -c "import sys; sys.exit(0 if $BAL >= $worst else 1)" \
    || echo "    WARNING: balance \$$BAL < \$$worst: if the account runs dry RunPod stops the pod before the cap"
done
echo "  volume ${VOL} GB at /workspace, container disk ${CDISK} GB, terminate guard: $TA_NOTE"
echo "  laptop guard: $([ $GUARD = 1 ] && echo "stop the pod if still RUNNING at cap + 1 h" || echo off)"
echo
echo "Commands:"
build_cmd() { CMD=(); while IFS= read -r line; do CMD+=("$line"); done < <(create_cmd "$@"); }
build_cmd "${CANDS[0]:-${GPU_LIST[0]}}" "$(hours_for "${CANDS[0]:-${GPU_LIST[0]}}")" "$(env_json "$(hours_for "${CANDS[0]:-${GPU_LIST[0]}}")" redact)"
echo "  $(quote "${CMD[@]}")"
echo "  runpodctl ssh info <pod-id>    # -> ip, port"
echo "  ssh -i ${SSH_KEY:-<registered key>} -p <port> root@<ip> $(quote "$BOOT")"
echo "  ssh ... 'cat /workspace/pod_bootstrap.log; tail /workspace/syco-mo-transfer/pod_run.log'   # confirm the start"
echo "  on SSH timeout: runpodctl pod delete <pod-id>, then create again (next GPU after 2 tries)"
if [ ${#problems[@]} -gt 0 ]; then
  echo; printf 'PROBLEM: %s\n' "${problems[@]}"
  [ $YES = 1 ] && die "fix the problems above first"
fi
if [ $YES != 1 ]; then echo; echo "Dry run: nothing created. Re-run with --yes to create the pod (billing starts)."; exit 0; fi

# ---- provision (billing starts here)
POD_ID="" GPU="" H=""
for g in "${CANDS[@]}"; do
  H=$(hours_for "$g")
  for attempt in 1 2; do
    echo ">>> creating pod on $g (attempt $attempt): waits up to 10 min for SSH"
    errf=$(mktemp)
    build_cmd "$g" "$H" "$(env_json "$H")"
    out=$("${CMD[@]}" 2>"$errf")
    rc=$?
    grep -v '^{' "$errf" | tail -n 3 | sed 's/^/    /' >&2   # last --wait progress lines
    err=$(python3 -I -c 'import json, sys
lines = [l for l in open(sys.argv[1]).read().splitlines() if l.strip().startswith("{")]
print(lines[-1] if lines else "{}")' "$errf"); rm -f "$errf"
    if [ $rc -eq 0 ]; then POD_ID=$(pyj 'd.get("id", "")' <<<"$out"); GPU="$g"; break 2; fi
    stale=$(pyj 'd.get("id", "")' <<<"$err")
    echo "    failed: $(pyj 'd.get("code", "") + ": " + str(d.get("error", ""))[:200]' <<<"$err")"
    if [ -n "$stale" ]; then
      echo "    pod $stale never answered SSH: deleting it (a fresh machine usually works)"
      runpodctl pod delete "$stale" >/dev/null 2>&1 || echo "    DELETE FAILED: remove pod $stale yourself (it bills)!"
    else
      break   # no capacity for this GPU type: next one
    fi
  done
done
[ -n "$POD_ID" ] || die "could not get a pod on any of: ${CANDS[*]}"
echo ">>> pod $POD_ID up on $GPU (cap ${H} h)"

# ---- bootstrap (detached on the pod) and confirm the run started
boot=$(ssh_to "$POD_ID" "$BOOT")
case "$boot" in *BOOTSTRAP_LAUNCHED*) ;; *) echo "bootstrap did not launch: $boot";; esac
started=0
for _ in $(seq 20); do
  sleep "${SYCOMO_LAUNCH_POLL_S:-15}"
  log=$(ssh_to "$POD_ID" 'cat /workspace/pod_bootstrap.log 2>/dev/null; echo ---RUN---; pgrep -f "bash scripts/run.sh" >/dev/null && echo RUN_SH_ALIVE; tail -n 8 /workspace/syco-mo-transfer/pod_run.log 2>/dev/null')
  case "$log" in *RUN_SH_ALIVE*) started=1; break ;; *"clone failed"*|*"unknown SYCOMO_REF"*|*"not found at"*|*"fetch failed"*) break ;; esac
done
echo "$log"
if [ $started != 1 ]; then
  echo "!!! the run did not start; stopping pod $POD_ID so it does not bill (volume kept)"
  runpodctl pod stop "$POD_ID" >/dev/null 2>&1 || echo "STOP FAILED: runpodctl pod stop $POD_ID"
  exit 1
fi
case "$log" in *"Auto-stop ARMED"*) ;; *) echo "!!! WARNING: the pod did not report 'Auto-stop ARMED': watch it and stop it yourself";; esac

if [ $GUARD = 1 ]; then
  mkdir -p "$HOME/.cache/sycomo"
  glog="$HOME/.cache/sycomo/guard-$POD_ID.log"
  deadline=$(python3 -c "import time; print(int(time.time() + ($H + 1) * 3600))")
  # wall-clock polling (not one long sleep): a sleeping laptop fires the guard on wake
  nohup bash -c 'while [ "$(date +%s)" -lt '"$deadline"' ]; do sleep 60; done
    st=$(runpodctl pod get '"$POD_ID"' 2>/dev/null | python3 -I -c "import json,sys; print(json.load(sys.stdin).get(\"desiredStatus\",\"\"))" 2>/dev/null)
    if [ "$st" = RUNNING ]; then echo "$(date): pod '"$POD_ID"' still RUNNING at cap + 1 h: stopping it"; runpodctl pod stop '"$POD_ID"'
    else echo "$(date): pod '"$POD_ID"' is ${st:-gone}: nothing to do"; fi' > "$glog" 2>&1 < /dev/null &
  disown 2>/dev/null || true
  echo "laptop guard: pid $! stops the pod if still RUNNING at $(date -r "$deadline" 2>/dev/null || date -d "@$deadline") (log $glog)"
fi

cat <<EOF

Pod $POD_ID ($GPU, \$$(price_of "$GPU")/h) is running $CFG at $REF; watchdog cap ${H} h.
  status:   bash scripts/pod_status.sh $POD_ID
  log:      ssh into it (runpodctl ssh info $POD_ID) and: tail -f /workspace/syco-mo-transfer/pod_run.log
  results:  https://huggingface.co/datasets/<your HF user>/syco-mo-transfer-results/tree/main/runs
  stop now: runpodctl pod stop $POD_ID     (GPU billing ends; the ${VOL} GB volume keeps billing, ~\$$(python3 -c "print(round($VOL*0.2/30, 2))")/day)
  delete:   runpodctl pod delete $POD_ID   (only after the HF upload is confirmed: deletes the volume)
EOF
