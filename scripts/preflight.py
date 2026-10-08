#!/usr/bin/env python
"""Preflight: run before renting a GPU and again on the pod (run.sh does).
Checks everything that could kill a paid run: config, dependencies, GPU, model and
data access at the pinned revisions, disk space. Exits non-zero on any failure.

Usage: .venv/bin/python scripts/preflight.py configs/qwen7b.yaml
"""

import os
import platform
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
ok_all = True


def check(name, fn, hard=True):
    global ok_all
    try:
        print(f"  PASS  {name}: {fn() or 'ok'}", flush=True)
    except Exception as e:  # noqa: BLE001
        ok_all = ok_all and not hard
        print(f"  {'FAIL' if hard else 'WARN'}  {name}: {type(e).__name__}: {e}", flush=True)


def main():
    from sycomo.config import load_config
    from sycomo.util import load_env

    load_env(REPO)
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/qwen7b.yaml"
    cfg = load_config(cfg_path)
    print(f"Preflight {cfg_path}: model {cfg.model.name}@{(cfg.model.revision or 'UNPINNED')[:8]} -> {cfg.out_dir}\n")
    on_pod = platform.system() == "Linux"

    def deps():
        import peft, sklearn, torch, transformers  # noqa: F401
        return f"torch {torch.__version__}, transformers {transformers.__version__}, peft {peft.__version__}"
    check("dependencies", deps)

    def gpu():
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            x = torch.ones(4, device="cuda", dtype=torch.bfloat16).sum().item()  # catches driver/wheel mismatch
            assert x == 4
            return f"{p.name}, {p.total_memory / 1e9:.0f} GB, bf16 ok"
        if torch.backends.mps.is_available():
            return "MPS (Apple Silicon): fine for the smoke config only"
        raise RuntimeError("no GPU visible. If nvidia-smi works, the host driver is older than CUDA 12.8 "
                           "(torch cu128 wheel): pick a pod with CUDA >= 12.8")
    check("gpu", gpu, hard=on_pod)

    def model_access():
        from huggingface_hub import hf_hub_download
        hf_hub_download(cfg.model.name, "config.json", revision=cfg.model.revision)
        return f"{cfg.model.name} downloadable at the pinned revision"
    check("model access", model_access)

    def data_access():
        from sycomo import sources
        from huggingface_hub import HfApi
        sources.fetch_syco_eval()
        api = HfApi()
        for name, (repo, rev, files) in sources.HF_FILES.items():
            info = api.dataset_info(repo, revision=rev, files_metadata=False)
            have = {s.rfilename for s in info.siblings}
            missing = [f for f, _ in files if f not in have]
            assert not missing, f"{repo}@{rev[:8]} lacks {missing}"
        return f"SycophancyEval (sha256 ok) + {len(sources.HF_FILES)} HF datasets at pinned commits"
    check("data access", data_access)

    def disk():
        hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hf.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(hf).free / 1e9
        out = Path(cfg.out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        free_out = shutil.disk_usage(out).free / 1e9
        big = "7B" in cfg.model.name or "8B" in cfg.model.name
        need = 40 if big else 5
        assert min(free, free_out) >= need, f"{free:.0f} GB free at HF_HOME, {free_out:.0f} GB at out_dir; need ~{need} GB"
        return f"{free:.0f} GB free at HF_HOME, {free_out:.0f} GB at {out}"
    check("disk", disk)

    def tokenizer():
        from sycomo.modeling import LETTERS, chat_text, load_tokenizer
        from sycomo.prompts import ANSWER_PREFILL
        tok = load_tokenizer(cfg.model)
        probe = chat_text(tok, [{"role": "user", "content": "Q"}], ANSWER_PREFILL)
        pre = tok(probe, add_special_tokens=False)["input_ids"]
        for L in LETTERS:
            full = tok(probe + L, add_special_tokens=False)["input_ids"]
            assert full[: len(pre)] == pre and len(full) == len(pre) + 1, f"letter {L} merges with '('"
        a = chat_text(tok, [{"role": "user", "content": "x"}])
        assert a == chat_text(tok, [{"role": "user", "content": "x"}]), "chat template not deterministic"
        return "letter readout tokenizes cleanly; template deterministic"
    check("tokenizer", tokenizer)

    print()
    if ok_all:
        print("PREFLIGHT PASSED")
    else:
        print("PREFLIGHT FAILED: fix the FAIL lines before spending GPU time")
        sys.exit(1)


if __name__ == "__main__":
    main()
