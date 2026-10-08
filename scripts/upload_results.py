#!/usr/bin/env python
"""Sync a run directory to a private HF dataset repo (needs HF_TOKEN with write access).

Usage:
  upload_results.py CONFIG                       one sync
  upload_results.py CONFIG --loop 20 --pid PID   sync every 20 min while PID is alive
  upload_results.py CONFIG --dry-run             list what would be uploaded, upload nothing

Repo <token owner>/syco-mo-transfer-results:
  runs/<run_id>/...    real runs
  smoke/<run_id>/...   smoke-test runs (kept apart so they can never pass for real data)
Uploaded: data/, adapters/ (the reusable MOs), features/, directions/, transfer/, analysis/,
report + figures, manifest, logs. Activation caches (acts/, a few GB, recomputable from
adapters + data) only with SYCOMO_UPLOAD_ACTS=1. Never raises: a failed upload must not
take down a run; it logs and the next sync retries.
"""

import argparse
import fnmatch
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [hf-sync] {msg}", flush=True)


def ignore_patterns() -> list[str]:
    pats = ["*.tmp", "*.tmp.npz", "*.part"]
    if os.environ.get("SYCOMO_UPLOAD_ACTS") != "1":
        pats.append("acts/*")
    return pats


def target(cfg):
    from sycomo.util import read_json

    out = Path(cfg.out_dir)
    man = out / "manifest.json"
    if not man.exists():
        return None, None
    prefix = "smoke" if cfg.run_name.startswith("smoke") else "runs"
    return out, f"{prefix}/{read_json(man)['run_id']}"


def sync(cfg, dry_run=False, attempts=3) -> bool:
    out, dest = target(cfg)
    if out is None:
        log(f"{cfg.out_dir} has no manifest yet; nothing to upload")
        return False
    if dry_run:
        files = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()]
        keep = [f for f in files if not any(fnmatch.fnmatch(f, pat) for pat in ignore_patterns())]
        size = sum((out / f).stat().st_size for f in keep) / 1e6
        log(f"dry run: would upload {len(keep)}/{len(files)} files ({size:.1f} MB) to <user>/syco-mo-transfer-results/{dest}")
        return True
    token = os.environ.get("HF_TOKEN")
    if not token:
        log("HF_TOKEN not set; results stay on disk only")
        return False
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for k in range(attempts):
        try:
            repo_id = f"{api.whoami()['name']}/syco-mo-transfer-results"
            api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
            api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="dataset", path_in_repo=dest,
                              ignore_patterns=ignore_patterns(), commit_message=f"sync {dest}")
            pod_log = REPO / "pod_run.log"
            if pod_log.exists():
                api.upload_file(path_or_fileobj=str(pod_log), path_in_repo=f"{dest}/pod_run.log", repo_id=repo_id,
                                repo_type="dataset", commit_message=f"pod log {dest}")
            log(f"synced {out} -> https://huggingface.co/datasets/{repo_id}/tree/main/{dest}")
            return True
        except Exception as e:  # noqa: BLE001
            log(f"attempt {k + 1}/{attempts} failed: {type(e).__name__}: {e}")
            time.sleep(30 * (k + 1))
    return False


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--loop", type=float, default=0, help="minutes between syncs (0 = sync once)")
    ap.add_argument("--pid", type=int, help="with --loop: stop when this process exits")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from sycomo.config import load_config
    from sycomo.util import load_env

    load_env(REPO)
    cfg = load_config(args.config)
    if not args.loop:
        sync(cfg, args.dry_run)
        return
    log(f"periodic sync every {args.loop:g} min while pid {args.pid} runs")
    while args.pid is None or alive(args.pid):
        deadline = time.time() + args.loop * 60
        while time.time() < deadline:
            if args.pid is not None and not alive(args.pid):
                return  # the owner does the final sync
            time.sleep(min(30, max(1, deadline - time.time())))
        sync(cfg, args.dry_run)


if __name__ == "__main__":
    main()
