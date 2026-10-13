"""Stage transfer: project a unit direction out of every residual write at every layer of a
target model and re-run the target's eval suite. One JSON per cell; resumable per cell.

v2 computes the entries the questions need instead of v1's full n x n matrix:
  target natural: none (unablated baseline, same code path), u_natural, u_<MO>, u_<seed>,
                  w_<MO> (the MO direction's component orthogonal to the natural one),
                  nat_onpolicy (natural's flipped-vs-held direction), rand0..R-1
  target each MO: none, u_<self>, u_natural, rand0..r-1
Natural-target cells run first (they carry the headline entries), so a capped run still has
them. Directions are read at the shared layer (directions/v2.json).
"""

from __future__ import annotations

import time

import numpy as np

from .ablation import random_unit
from .design import Layout, cell_map
from .evals import run_suite
from .features import load_lm, load_sets, transfer_direction
from .modeling import NATURAL
from .util import derive_seed, log, read_json, write_json


def _group(cfg, roles: list[str]) -> list[str]:
    want = {"mos": "mo", "seeds": "seed", "controls": "control"}
    keep = {want[r] for r in roles if r in want}
    return [c.name for c in cell_map(cfg).values() if c.role in keep]


def plan(cfg) -> list[tuple[str, str]]:
    """Ordered (target, source) cells."""
    t = cfg.transfer
    L = Layout(cfg.out_dir)
    have_on = read_json(L.directions / "v2.json")["nat_onpolicy"]["available"]
    srcs = ["none"]
    if "natural" in t.natural_sources:
        srcs.append(f"u_{NATURAL}")
    for grp in ("mos", "seeds", "controls"):
        if grp in t.natural_sources:
            srcs += [f"u_{m}" for m in _group(cfg, [grp])]
    if "w" in t.natural_sources:
        srcs += [f"w_{m}" for m in _group(cfg, ["mos"])]
    if "nat_onpolicy" in t.natural_sources and have_on:
        srcs.append("nat_onpolicy")
    srcs += [f"rand{k}" for k in range(t.n_random_natural)]
    jobs = [(NATURAL, s) for s in srcs]
    for m in _group(cfg, t.mo_targets):
        jobs += [(m, s) for s in ["none", f"u_{m}", f"u_{NATURAL}"] + [f"rand{k}" for k in range(t.n_random_mo)]]
    return jobs


def _neutral_cell(cfg, source: str) -> bool:
    want = cfg.transfer.natural_neutral_sources
    if source == "none":
        return "none" in want
    if source == f"u_{NATURAL}":
        return "natural" in want
    return source.startswith("u_") and source[2:] in _group(cfg, [g for g in want if g in ("mos", "seeds", "controls")])


def random_direction(cfg, target: str, k: int, d: int, layer: int) -> np.ndarray:
    seed = derive_seed(cfg.seed, "random_direction", target, k)
    if cfg.transfer.random_kind == "isotropic":
        return random_unit(d, seed)
    X = np.load(Layout(cfg.out_dir).acts / f"{target}_pairs.npz")["acts"][:, layer].astype(np.float32)
    return random_unit(d, seed, X - X.mean(0))


def stage_transfer(cfg) -> None:
    L = Layout(cfg.out_dir)
    jobs_all = plan(cfg)
    jobs = [(t, s) for t, s in jobs_all if not L.cell_file(t, s).exists()]
    log(f"[transfer] {len(jobs_all)} cells ({sum(t == NATURAL for t, _ in jobs_all)} on natural), "
        f"{len(jobs_all) - len(jobs)} already done")
    if not jobs:
        return
    targets = sorted({t for t, _ in jobs_all})
    lm = load_lm(cfg, targets)
    sets = load_sets(cfg)
    layer = int(read_json(L.directions / "v2.json")["shared_layer"])
    t0 = time.time()
    for n, (target, source) in enumerate(jobs):
        t1 = time.time()
        if source == "none":
            u, kind = None, "none"
        elif source.startswith("rand"):
            u, kind = random_direction(cfg, target, int(source[4:]), lm.hidden_size, layer), "random"
        else:
            u, _ = transfer_direction(cfg, source)
            kind = source.split("_", 1)[0] if source != "nat_onpolicy" else "nat_onpolicy"
        which = list(cfg.transfer.natural_evals if target == NATURAL else cfg.transfer.mo_evals)
        if target == NATURAL and _neutral_cell(cfg, source) and "ays_syco_full_neutral" not in which:
            which.append("ays_syco_full_neutral")
        summ, rows, _ = run_suite(lm, target, sets, which, cfg.eval.ays_max_new_tokens, direction=u,
                                  desc=f"{source}->{target}:")
        write_json(L.cell_file(target, source), dict(target=target, source=source, kind=kind, layer=layer,
                                                     summaries=summ, rows=rows, wall_s=time.time() - t1))
        eta = (time.time() - t0) / (n + 1) * (len(jobs) - n - 1)
        fr = next((s["flip_rate"] for s in summ.values() if isinstance(s, dict) and "flip_rate" in s), float("nan"))
        log(f"[transfer] {n + 1}/{len(jobs)} {source:>14} -> {target:<12} flip={fr:.3f} "
            f"({time.time() - t1:.0f}s, eta {eta / 60:.0f} min)")
