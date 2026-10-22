"""python -m sycomo --config configs/qwen7b.yaml [--stages a,b] [--force] [--set key=value ...]

Runs every stage in order in one process. A stage whose completion marker exists is
skipped (so a crashed run resumes where it stopped); --force re-runs the named stages.
Per-cell stages (train, features, transfer, steer) also resume per cell.

Post-hoc stages (steer) are not part of `all`: they run on a finished run directory
(--stages steer), write only their own subdirectory, and may use a config that differs from
the run's (recorded in manifest.json). --set overrides config keys, e.g.
  python -m sycomo --config configs/v2_steer_llama8b.yaml --stages steer --set steer.preflight_items=20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from importlib import metadata

from .config import load_config
from .design import Layout
from .util import REPO_ROOT, load_env, log, read_json, write_json

STAGES = ["import_v1", "prepare", "natural", "generate", "partner", "train", "features", "directions", "transfer",
          "analysis", "report"]
MARKERS = {"import_v1": "items/import.json", "prepare": "items/prepare_v2.json", "natural": "natural/native_v2.json",
           "generate": "data/generate_v2.json", "partner": "data/partner.json", "train": "adapters/.done",
           "features": "features/.done", "directions": "directions/v2.json", "transfer": "transfer/.done",
           "analysis": "analysis/hypotheses.json", "report": "report.md"}
POST_HOC = ["steer"]  # not in `all`; need a finished run dir
POST_HOC_MARKERS = {"steer": "steer/.done"}  # kept out of MARKERS: check_run/preflight read MARKERS as the pipeline
NEEDS_NATURAL_LM = {"natural", "generate"}
PER_CELL = ("train", "features", "transfer", "steer")


def _git() -> dict:
    def run(*a):
        try:
            p = subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
            return p.stdout.strip() if p.returncode == 0 else ""
        except Exception:  # noqa: BLE001
            return ""
    return dict(commit=run("rev-parse", "HEAD"), dirty=bool(run("status", "--porcelain")), branch=run("rev-parse", "--abbrev-ref", "HEAD"))


def _env() -> dict:
    import torch

    from .modeling import pick_device

    pk = {p: _ver(p) for p in ("torch", "transformers", "peft", "accelerate", "numpy", "scikit-learn", "scipy", "huggingface-hub")}
    env = dict(python=sys.version.split()[0], platform=platform.platform(), device=pick_device(), packages=pk)
    if torch.cuda.is_available():
        env.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, n_gpus=torch.cuda.device_count(),
                   driver=_nvidia_driver())
    return env


def _ver(p):
    try:
        return metadata.version(p)
    except metadata.PackageNotFoundError:
        return None


def _nvidia_driver():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def update_manifest(cfg, cfg_path: str, allow_change: bool = False) -> dict:
    """manifest.json: created on first run (fixes run_id), extended on every invocation.
    Refuses to resume a run directory whose config changed, unless allow_change (then the
    new config is recorded on the invocation)."""
    L = Layout(cfg.out_dir)
    path = L.root / "manifest.json"
    cfg_json = json.dumps(cfg.to_dict(), sort_keys=True)
    sha = hashlib.sha256(cfg_json.encode()).hexdigest()
    git = _git()
    if path.exists():
        man = read_json(path)
        prev = man.get("git", {}).get("commit")
        if prev and git["commit"] and prev != git["commit"]:
            log(f"[main] WARNING: resuming a run started at commit {prev[:10]} with code at {git['commit'][:10]}; "
                f"both are recorded in manifest.json invocations")
        if man["config_sha256"] != sha and not allow_change:
            raise SystemExit(f"{path} was produced by a different config (sha {man['config_sha256'][:12]} != {sha[:12]}). "
                             f"Use a new run_name/out_dir, delete the directory, or pass --allow-config-change "
                             f"(e.g. for analysis-only settings).")
    else:
        man = dict(run_id=f"{cfg.run_name}-{time.strftime('%Y%m%d-%H%M%S')}-{(git['commit'] or 'nogit')[:7]}-"
                          f"{uuid.uuid4().hex[:4]}", config=cfg.to_dict(), config_sha256=sha, created=time.time(),
                   invocations=[])
    man["invocations"].append(dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv, config_path=cfg_path,
                                   git=git, env=_env(), config_sha256=sha,
                                   **({"config": cfg.to_dict()} if sha != man["config_sha256"] else {})))
    man["git"], man["env"] = git, man["invocations"][-1]["env"]
    write_json(path, man)
    return man


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", default="all", help=f"comma-separated subset of {STAGES + POST_HOC}, or 'all' "
                                                    f"(= {STAGES})")
    ap.add_argument("--force", action="store_true", help="re-run the selected stages even if complete")
    ap.add_argument("--allow-config-change", action="store_true",
                    help="resume a run dir whose config differs (recorded in the manifest)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config key (YAML value), repeatable, e.g. --set steer.preflight_items=20")
    args = ap.parse_args()

    load_env()
    cfg = load_config(args.config, args.set)
    L = Layout(cfg.out_dir)
    stages = STAGES if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    bad = [s for s in stages if s not in STAGES + POST_HOC]
    if bad:
        raise SystemExit(f"unknown stages {bad}; valid: {STAGES + POST_HOC}")
    stages = [s for s in STAGES + POST_HOC if s in stages]
    post_hoc_only = all(s in POST_HOC for s in stages)
    if post_hoc_only and not (L.root / "manifest.json").exists():
        raise SystemExit(f"{stages} run on a finished run directory, but {L.root}/manifest.json does not exist "
                         f"(download the run first; out_dir must be its directory)")
    L.root.mkdir(parents=True, exist_ok=True)
    man = update_manifest(cfg, args.config, args.allow_config_change or post_hoc_only)
    log(f"[main] run {man['run_id']} -> {L.root}")

    from .util import set_seed

    set_seed(cfg.seed)
    natural_lm = None
    for s in stages:
        marker = L.root / {**MARKERS, **POST_HOC_MARKERS}[s]
        preflight = s == "steer" and cfg.steer.preflight_items is not None  # always runs (fresh)
        if marker.exists() and not args.force and not preflight:
            log(f"[main] skip {s} (done)")
            continue
        if args.force and s in PER_CELL:
            log(f"[main] --force on {s}: per-cell outputs that exist are still reused; delete them to recompute")
        log(f"[main] === {s} ===")
        t0 = time.time()
        if s in NEEDS_NATURAL_LM and natural_lm is None:
            from .modeling import LM

            natural_lm = LM(cfg.model)
        if s not in NEEDS_NATURAL_LM and natural_lm is not None:
            del natural_lm  # free GPU memory before stages that load their own model
            natural_lm = None
            _free()
        _run(s, cfg, natural_lm)
        if s in ("train", "features", "transfer"):
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        _free()
        man = read_json(L.root / "manifest.json")
        man.setdefault("stages", {})[s] = dict(seconds=round(time.time() - t0, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_json(L.root / "manifest.json", man)
        log(f"[main] {s} finished in {(time.time() - t0) / 60:.1f} min")
    log("[main] all requested stages complete")


def _free():
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run(stage, cfg, lm):
    if stage == "import_v1":
        from .data import stage_import_v1
        stage_import_v1(cfg)
    elif stage == "prepare":
        from .data import stage_prepare
        stage_prepare(cfg)
    elif stage == "natural":
        from .data import stage_natural
        stage_natural(cfg, lm)
    elif stage == "generate":
        from .data import stage_generate
        stage_generate(cfg, lm)
    elif stage == "partner":
        from .data import stage_partner
        stage_partner(cfg)
    elif stage == "train":
        from .train import stage_train
        stage_train(cfg)
    elif stage == "features":
        from .features import stage_features
        stage_features(cfg)
    elif stage == "directions":
        from .features import stage_directions
        stage_directions(cfg)
    elif stage == "transfer":
        from .transfer import stage_transfer
        stage_transfer(cfg)
    elif stage == "steer":
        from .steer import stage_steer
        stage_steer(cfg)
    elif stage == "analysis":
        from .analysis import stage_analysis
        stage_analysis(cfg)
    elif stage == "report":
        from .report import stage_report
        stage_report(cfg)


if __name__ == "__main__":
    main()
