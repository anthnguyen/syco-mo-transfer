"""Stage transfer: ablate model i's sycophancy direction (at its L*_i, projected out of
every residual write at every layer) in model j and re-run the eval suite.

Cells per target j:  none (unablated baseline, same code path), every source i,
and n_random random directions. One JSON per cell; resumable per cell. The
baseline cell re-measures the unablated model so every entry of a column is
measured by the identical instrument (and checks determinism against features).
"""

from __future__ import annotations

import time

import numpy as np

from .ablation import random_unit
from .data import k_actual
from .design import Layout, transfer_models
from .evals import run_suite
from .features import direction_of, load_lm, load_sets
from .util import derive_seed, log, read_json, write_json


def random_direction(cfg, target: str, k: int, d: int) -> tuple[np.ndarray, int | None]:
    seed = derive_seed(cfg.seed, "random_direction", target, k)
    if cfg.transfer.random_kind == "isotropic":
        return random_unit(d, seed), None
    L = Layout(cfg.out_dir)
    layer = read_json(L.directions / "probes.json")[target]["best_layer"]
    X = np.load(L.acts / f"{target}_pairs.npz")["acts"][:, layer].astype(np.float32)
    return random_unit(d, seed, X - X.mean(0)), layer


def cell_sources(cfg, models: list[str]) -> list[str]:
    return ["none"] + models + [f"rand{k}" for k in range(cfg.transfer.n_random)]


def stage_transfer(cfg) -> None:
    L = Layout(cfg.out_dir)
    models = transfer_models(cfg, k_actual(cfg))
    jobs = [(t, s) for t in models for s in cell_sources(cfg, models) if not L.cell_file(t, s).exists()]
    total = len(models) * len(cell_sources(cfg, models))
    log(f"[transfer] {total} cells ({len(models)} targets x {len(cell_sources(cfg, models))} sources), "
        f"{total - len(jobs)} already done")
    if not jobs:
        return
    lm = load_lm(cfg, models)
    sets = load_sets(cfg)
    dirs = {m: direction_of(cfg, m) for m in models}
    t0 = time.time()
    for n, (target, source) in enumerate(jobs):
        t1 = time.time()
        if source == "none":
            u, layer = None, None
        elif source.startswith("rand"):
            u, layer = random_direction(cfg, target, int(source[4:]), lm.hidden_size)
        else:
            u, layer = dirs[source]
        summ, rows, _ = run_suite(lm, target, sets, cfg.transfer.evals, cfg.eval.ays_max_new_tokens,
                                  direction=u, desc=f"{source}->{target}:")
        write_json(L.cell_file(target, source), dict(target=target, source=source, layer=layer,
                                                     summaries=summ, rows=rows, wall_s=time.time() - t1))
        eta = (time.time() - t0) / (n + 1) * (len(jobs) - n - 1)
        fr = summ.get("ays_syco", {}).get("flip_rate", float("nan"))
        log(f"[transfer] {n + 1}/{len(jobs)} {source:>10} -> {target:<10} flip={fr:.3f} "
            f"({time.time() - t1:.0f}s, eta {eta / 60:.0f} min)")
