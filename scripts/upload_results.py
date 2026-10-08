#!/usr/bin/env python
"""Upload a finished run directory to a private HF dataset repo (needs HF_TOKEN).

Usage: upload_results.py CONFIG
Repo <token owner>/syco-mo-transfer-results, one folder per run_id. Adapters (the
reusable model organisms), data, features, transfer cells, analysis and report are
uploaded; activation caches (acts/, ~2 GB, recomputable from adapters + data) only
with SYCOMO_UPLOAD_ACTS=1.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main():
    from sycomo.config import load_config
    from sycomo.util import load_env, read_json

    load_env(REPO)
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[upload] HF_TOKEN not set; results stay on disk")
        return
    from huggingface_hub import HfApi

    cfg = load_config(sys.argv[1])
    out = Path(cfg.out_dir)
    run_id = read_json(out / "manifest.json")["run_id"]
    api = HfApi(token=token)
    repo_id = f"{api.whoami()['name']}/syco-mo-transfer-results"
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    ignore = ["*.tmp", "*.tmp.npz"] + ([] if os.environ.get("SYCOMO_UPLOAD_ACTS") == "1" else ["acts/*"])
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="dataset", path_in_repo=run_id,
                      ignore_patterns=ignore, commit_message=f"results {run_id}")
    print(f"[upload] {out} -> https://huggingface.co/datasets/{repo_id}/tree/main/{run_id}")


if __name__ == "__main__":
    main()
