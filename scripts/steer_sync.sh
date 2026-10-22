#!/usr/bin/env bash
# Upload <run>/steer (and steer_preflight) to the HF results repo (PUBLIC); idempotent, bounded.
# The gated HLE set must not be published, so per-item rows of every ays_hle* eval are stripped from
# a staged copy before upload (summaries, i.e. flip rates and n, are kept). The full cells stay on the
# pod's /workspace and are rsynced to the laptop at the end. Also uploads the dispatch run log.
#
#   bash scripts/steer_sync.sh llama|qwen [timeout_s]
set -uo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  llama) RUN=v2-llama8b-20261013-093729-485ecc4-e1a5 ;;
  qwen)  RUN=v2-qwen7b-20261014-032435-bf3b387-dea8 ;;
  *) echo "usage: $0 llama|qwen [timeout_s]" >&2; exit 2 ;;
esac
T="${2:-600}"
REPO_ID=metametal/syco-mo-transfer-results
if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=.venv/bin/python
[ -x "$PY" ] && [ -n "${HF_TOKEN:-}" ] || { echo "[steer_sync] env not ready; skipped"; exit 0; }

LOGF=/workspace/dispatch/run.log   # keep the run log off the pod so a dead pod is diagnosable
if [ -f "$LOGF" ]; then
  timeout -k 30 120 .venv/bin/hf upload "$REPO_ID" "$LOGF" "runs/$RUN/steer_logs/run_${RUNPOD_POD_ID:-pod}.log" --repo-type dataset \
    --commit-message "steer log $RUN" >/dev/null && echo "[steer_sync] uploaded run log" || echo "[steer_sync] run log upload failed"
fi

STAGE=/workspace/steer_public/$RUN
for sub in steer steer_preflight; do
  d="results/hf/runs/$RUN/$sub"
  [ -d "$d" ] || continue
  "$PY" - "$d" "$STAGE/$sub" <<'PY' || { echo "[steer_sync] staging $sub failed; NOT uploading"; continue; }
import json, os, shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
n_strip = 0
for f in src.rglob("*"):
    if not f.is_file():
        continue
    out = dst / f.relative_to(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    if f.suffix == ".json" and f.parent.name == "cells":
        try:
            cell = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue  # being written; next sync
        rows = cell.get("rows") or {}
        for k in [k for k in rows if k.startswith("ays_hle")]:
            del rows[k]
            n_strip += 1
        cell["rows_stripped"] = sorted(set(cell.get("rows_stripped", [])) | {"ays_hle", "ays_hle_neutral"})
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(cell))
        os.replace(tmp, out)
    elif "hle" in f.name.lower() and f.suffix in (".jsonl", ".npz"):
        continue  # never stage per-item HLE side files
    else:
        shutil.copy2(f, out)
# safety net: refuse if any staged file still carries an HLE qid
for f in dst.rglob("*"):
    if f.is_file() and f.suffix in (".json", ".jsonl", ".csv") and '"hle/' in f.read_text(errors="ignore"):
        sys.exit(f"HLE item ids still present in {f}")
print(f"[steer_sync] staged {src} -> {dst} ({n_strip} HLE row sets stripped)")
PY
  timeout -k 30 "$T" .venv/bin/hf upload "$REPO_ID" "$STAGE/$sub" "runs/$RUN/$sub" --repo-type dataset \
    --commit-message "steer sync $RUN/$sub (HLE rows stripped)" >/dev/null && echo "[steer_sync] uploaded $sub" \
    || echo "[steer_sync] upload of $sub failed or timed out"
done
exit 0
