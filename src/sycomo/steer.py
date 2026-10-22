"""Stage steer (post hoc): do MO sycophancy directions remove the NATURAL model's sycophancy, and how
generally? Runs on an existing v2 run directory (cfg.out_dir), natural target only.

Per source direction u (unit, read at the shared layer; u = sycophantic minus honest, so moving
along -u is away from sycophancy) and op:
  abl    directional ablation (ablation.ablate: u projected out of every residual writer), i.e. the
         v2 transfer intervention, now on the wider steer eval suite
  s<k>   activation steering: add coef * u with coef = -k * sd_u to the OUTPUT of decoder block
         shared_layer - 1 (= residual "layer shared_layer" in modeling's convention: layer 0 is the
         embedding output, layer l the output of block l-1, which is where LM.capture/pooled_acts
         read the pair activations the directions were built from), at every position of the prompt
         and of every generated token. Ported from the spoof branch (spoof/evaluate.py `steer`).
  none   the unsteered baseline through the same code path (one cell)
sd_u = std (ddof 0) of the natural model's pair activations at the shared layer projected on u:
rows = the 2P contrastive pair replies (P sycophantic + P honest, mean over reply tokens, as in
features.stage_features), so the class gap along u is part of sd_u. acts/ is not uploaded to HF, so
the stage recomputes the natural pair activations when acts/natural_pairs.npz is missing (one forward
pass over the pairs) and checks that their diff-in-means reproduces directions' u_natural.

Outputs (<run>/steer/, or <run>/steer_preflight/ with steer.preflight_items set):
  cells/none.json, cells/<source>__<op>.json   transfer-cell schema + op, dose, coef, sd
  sd.json                                      sd_u and the projection means per source
  run_config.json                              steer settings, run id, evals, invocation provenance
Cells run in priority order (none, abl for every source, then s<first_dose>, the other doses
descending) and are skipped when their file exists, so a partial run is usable and resumable.
"""

from __future__ import annotations

import dataclasses
import shutil
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

from .design import Layout
from .modeling import NATURAL
from .util import log, read_json, read_jsonl, save_npz, write_json

NONE = "none"
SET_OF = {"mmlu": "mmlu", "mcq_bias": "mcq_bias", "mcq_bias_suggested": "mcq_bias", "aita": "aita", "wei_add": "wei"}


# ---------------------------------------------------------------- the intervention

def _unit(u) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    n = np.linalg.norm(u)
    if not np.isfinite(n) or n == 0:
        raise ValueError("steering direction has zero or non-finite norm")
    return u / n


@contextmanager
def steer(decoder, layer: int, u, coef: float):
    """Add coef * unit(u) to the residual stream at `layer` (output of decoder.layers[layer - 1]; layer 0 =
    the embedding output) at every position, for every forward pass inside the block (prompt and each
    generated token). Removed on exit, also on an exception."""
    if layer < 0 or layer > len(decoder.layers):
        raise ValueError(f"layer {layer} outside 0..{len(decoder.layers)}")
    mod = decoder.embed_tokens if layer == 0 else decoder.layers[layer - 1]
    p = next(decoder.parameters())
    v = torch.tensor(_unit(u) * float(coef), dtype=torch.float32, device=p.device)

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        h2 = (h.float() + v).to(h.dtype)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    handle = mod.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def projection_stats(X: np.ndarray, u, n_syc: int | None = None) -> dict:
    """sd (ddof 0) of the rows of X projected on unit(u); with n_syc (rows 0..n_syc-1 sycophantic, the rest
    honest) also the class means along u and their gap in sd units."""
    proj = np.asarray(X, dtype=np.float64) @ _unit(u)
    out = dict(sd=float(proj.std()), mean=float(proj.mean()), n=int(len(proj)))
    if n_syc is not None:
        a, b = float(proj[:n_syc].mean()), float(proj[n_syc:].mean())
        out.update(mean_syc=a, mean_honest=b, gap_in_sd=(a - b) / out["sd"] if out["sd"] > 0 else float("nan"))
    return out


# ---------------------------------------------------------------- plan

def dose_op(k: float) -> str:
    return f"s{float(k):g}"


def op_dose(op: str) -> float | None:
    return float(op[1:]) if op.startswith("s") and op != NONE else None


def ordered_ops(st) -> list[str]:
    """abl first, then s<first_dose>, then the remaining doses descending (e.g. abl, s4, s8, s2)."""
    doses = list(dict.fromkeys(float(k) for k in st.doses))
    order = ([st.first_dose] if st.first_dose in doses else []) + sorted((k for k in doses if k != st.first_dose),
                                                                          reverse=True)
    want = set(st.ops)
    ops = ["abl"] if "abl" in want else []
    for k in order:
        if "steer" in want or dose_op(k) in want:
            ops.append(dose_op(k))
    for o in st.ops:  # explicit s<k> not among the doses: last, in the given order
        if o not in ("abl", "steer") and o not in ops:
            ops.append(dose_op(float(o[1:])))
    return ops


def sources(cfg) -> list[str]:
    return list(dict.fromkeys(cfg.steer.sources)) + [f"rand{k}" for k in range(cfg.steer.n_random)]


def cell_id(source: str, op: str) -> str:
    return NONE if source == NONE else f"{source}__{op}"


def plan_cells(srcs: list[str], ops: list[str]) -> list[tuple[str, str]]:
    """(source, op) in run order: the baseline, then op by op (all sources each)."""
    return [(NONE, NONE)] + [(s, op) for op in ops for s in srcs]


def plan(cfg) -> list[tuple[str, str]]:
    cells = plan_cells(sources(cfg), ordered_ops(cfg.steer))
    if cfg.steer.preflight_items is not None:
        want = list(cfg.steer.preflight_cells)
        have = {cell_id(*c) for c in cells}
        miss = [c for c in want if c not in have]
        if miss:
            raise SystemExit(f"[steer] steer.preflight_cells {miss} are not in the plan (cells look like none, "
                             f"<source>__abl, <source>__s<k>)")
        cells = [c for c in cells if cell_id(*c) in want]
    return cells


def out_dir(cfg) -> Path:
    return Layout(cfg.out_dir).root / ("steer" if cfg.steer.preflight_items is None else "steer_preflight")


def cell_path(d: Path, source: str, op: str) -> Path:
    return Path(d) / "cells" / f"{cell_id(source, op)}.json"


def todo(cells: list[tuple[str, str]], d: Path) -> list[tuple[str, str]]:
    return [c for c in cells if not cell_path(d, *c).exists()]


def kind_of(source: str) -> str:
    if source == NONE:
        return "none"
    if source.startswith("rand"):
        return "random"
    if source == "nat_onpolicy":
        return "nat_onpolicy"
    return source.split("_", 1)[0]


# ---------------------------------------------------------------- evals and items

def evals_for(source: str, every: list[str], none_only: list[str]) -> list[str]:
    """Evals of one cell: `every` (steer.evals + neutral_evals, checked) and, on the none cell, `none_only`."""
    return list(dict.fromkeys(every + (none_only if source == NONE else [])))


def set_name(ev: str) -> str | None:
    from .evals import AYS_EVALS

    if ev in AYS_EVALS:
        return AYS_EVALS[ev][0]
    return SET_OF.get(ev)


def check_evals(cfg, sets: dict) -> tuple[list[str], list[str]]:
    """(every-cell evals, none-only evals) that can run. An eval unknown to evals.EVALS, or whose item set
    is missing or empty, stops the stage with the full list, unless steer.allow_missing_evals."""
    from .evals import EVALS

    st = cfg.steer
    every = list(dict.fromkeys(st.evals + st.neutral_evals))
    none_only = [e for e in dict.fromkeys(st.neutral_none_only) if e not in every]
    missing = {}
    for ev in every + none_only:
        if ev not in EVALS:
            missing[ev] = "not in evals.EVALS"
        elif not sets.get(set_name(ev)):
            missing[ev] = f"item set {set_name(ev)!r} missing or empty"
    if missing:
        msg = "; ".join(f"{e}: {why}" for e, why in missing.items())
        if not st.allow_missing_evals:
            raise SystemExit(f"[steer] cannot run evals -> {msg}. Fix the items/evals or set "
                             f"steer.allow_missing_evals: true to drop them.")
        log(f"[steer] WARNING: dropping evals (steer.allow_missing_evals) -> {msg}")
    return [e for e in every if e not in missing], [e for e in none_only if e not in missing]


def truncate_sets(sets: dict, n: int) -> dict:
    """Every item list cut to its first n items; mcq_bias keeps, per dataset, every row (all bias types and
    the unbiased baseline) of its first n question ids, so the switch pairs stay intact."""
    out = {}
    for k, v in sets.items():
        if not isinstance(v, list):
            out[k] = v
        elif k == "mcq_bias":
            keep: dict[str, list] = {}
            for r in v:
                q = keep.setdefault(r["dataset"], [])
                if r["question_id"] not in q and len(q) < n:
                    q.append(r["question_id"])
            out[k] = [r for r in v if r["question_id"] in keep[r["dataset"]]]
        else:
            out[k] = v[:n]
    return out


# ---------------------------------------------------------------- inputs

def check_inputs(cfg) -> None:
    L = Layout(cfg.out_dir)
    need = [L.root / "manifest.json", L.directions / "v2.json", L.directions / "transfer_dirs.npz",
            L.items / "syco.jsonl", L.items / "eval_in.jsonl", L.items / "mmlu.jsonl"]
    if not (L.acts / f"{NATURAL}_pairs.npz").exists():
        need.append(L.data / "pairs.jsonl")
    miss = [str(p.relative_to(L.root)) for p in need if not p.exists()]
    if miss:
        raise SystemExit(f"[steer] {L.root} is missing {miss}: download the v2 run (directions/, items/, natural/, "
                         f"data/pairs.jsonl, manifest.json) first; see the header of configs/v2_steer_llama8b.yaml")


def check_provenance(cfg) -> dict:
    """The run dir must have been made by the same parent and direction settings as this config."""
    man = read_json(Layout(cfg.out_dir).root / "manifest.json")
    run = man.get("config") or {}
    mine = dataclasses.asdict(cfg)  # full (to_dict drops keys still at their defaults)
    keys = [("model", "name"), ("model", "revision"), ("seed",), ("features", "exclude_dims"),
            ("transfer", "random_kind")]

    def get(d, path):
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
        return d

    dflt = {("features", "exclude_dims"): [], ("transfer", "random_kind"): "isotropic"}
    val = lambda d, k: get(d, k) if get(d, k) is not None else dflt.get(k)  # noqa: E731
    bad = {".".join(k): (val(run, k), val(mine, k)) for k in keys}
    bad = {k: v for k, v in bad.items() if v[0] != v[1]}
    if bad:
        raise SystemExit(f"[steer] config does not match the run's manifest (run, config): {bad}")
    return man


def natural_pair_acts(cfg, lm, layer: int, d: Path) -> np.ndarray:
    """[2P, hidden] natural pair activations at `layer` (rows 0..P-1 sycophantic). From acts/natural_pairs.npz
    when present, else recomputed (features.pair_sequences + LM.pooled_acts, same code as the features stage)
    and cached in the steer dir."""
    L = Layout(cfg.out_dir)
    src = L.acts / f"{NATURAL}_pairs.npz"
    cache = d / f"natural_pairs_L{layer}.npz"
    if src.exists():
        return np.load(src)["acts"][:, layer].astype(np.float32)
    if cache.exists():
        return np.load(cache)["acts"].astype(np.float32)
    from .features import pair_sequences

    seqs, spans = pair_sequences(lm, read_jsonl(L.data / "pairs.jsonl"))
    with lm.use(NATURAL):
        X = lm.pooled_acts(seqs, spans, "natural:pairs")[:, layer].astype(np.float32)
    save_npz(cache, acts=X)
    return X


def check_u_natural(cfg, X: np.ndarray, u_nat: np.ndarray, min_cos: float = 0.95) -> float:
    """cos(diff-in-means of X, directions' u_natural): checks the layer convention and the recomputation."""
    from .features import mask_dims

    P = len(X) // 2
    v = X[:P].mean(0) - X[P:].mean(0)
    if cfg.features.exclude_dims:
        v = mask_dims(v, cfg.features.exclude_dims)
    c = float(_unit(v) @ _unit(u_nat))
    if c < min_cos:
        raise SystemExit(f"[steer] the natural pair activations' diff-in-means has cos {c:.4f} with directions' "
                         f"u_natural (< {min_cos}): wrong layer convention or different pairs/model")
    return c


# ---------------------------------------------------------------- stage

def stage_steer(cfg) -> None:
    from .evals import run_suite
    from .features import load_lm, load_sets, transfer_direction
    from .transfer import random_direction

    st = cfg.steer
    L = Layout(cfg.out_dir)
    pre = st.preflight_items is not None
    check_inputs(cfg)
    man = check_provenance(cfg)
    d = out_dir(cfg)
    if pre and (d / "cells").exists():
        shutil.rmtree(d / "cells")  # a preflight always tests the current code end to end
    layer = int(read_json(L.directions / "v2.json")["shared_layer"])
    tdirs = np.load(L.directions / "transfer_dirs.npz")
    srcs = sources(cfg)
    miss = [s for s in srcs if not s.startswith("rand") and s not in tdirs.files]
    if miss:
        raise SystemExit(f"[steer] sources {miss} not in directions/transfer_dirs.npz ({sorted(tdirs.files)})")
    cells = plan(cfg)
    sets = load_sets(cfg)
    if pre:
        sets = truncate_sets(sets, st.preflight_items)
    every, none_only = check_evals(cfg, sets)
    jobs = todo(cells, d)
    write_json(d / "run_config.json", dict(
        run_id=man.get("run_id"), layer=layer, preflight_items=st.preflight_items, steer=cfg.to_dict()["steer"],
        evals=every, none_only_evals=none_only, cells=[cell_id(*c) for c in cells],
        n_items={k: len(v) for k, v in sets.items() if isinstance(v, list)},
        invocation=(man.get("invocations") or [{}])[-1]))
    log(f"[steer] {len(cells)} cells -> {d} ({len(cells) - len(jobs)} already done); layer {layer}; "
        f"evals {every} (+ {none_only} on none)")
    if not jobs:
        (d / ".done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        return

    lm = load_lm(cfg, [NATURAL])
    dirs = {s: (random_direction(cfg, NATURAL, int(s[4:]), lm.hidden_size, layer) if s.startswith("rand")
                else transfer_direction(cfg, s)[0]) for s in srcs}
    sd_path = d / "sd.json"
    sd = read_json(sd_path) if sd_path.exists() else {}
    if any(s not in sd.get("sources", {}) for s in srcs):  # resumed runs keep the sd their earlier cells used
        X = natural_pair_acts(cfg, lm, layer, d)
        cos = check_u_natural(cfg, X, tdirs["u_natural"])
        P = len(X) // 2
        sd = dict(layer=layer, n_rows=len(X), rows="natural pair replies (P syc + P honest), mean over reply tokens",
                  cos_u_natural_recomputed=cos,
                  sources={**sd.get("sources", {}), **{s: projection_stats(X, dirs[s], P) for s in srcs
                                                      if s not in sd.get("sources", {})}})
        write_json(sd_path, sd)
        log(f"[steer] sd at layer {layer} from {len(X)} pair rows (u_natural recomputed: cos {cos:.4f}): "
            + " ".join(f"{s}={v['sd']:.3g}" for s, v in sd["sources"].items()))

    t0 = time.time()
    for n, (source, op) in enumerate(jobs):
        t1 = time.time()
        u = None if source == NONE else dirs[source]
        k = op_dose(op)
        sdu = None if source == NONE else float(sd["sources"][source]["sd"])
        coef = -k * sdu if k is not None else None
        ctx = steer(lm.decoder, layer, u, coef) if k is not None else nullcontext()
        which = evals_for(source, every, none_only)
        with ctx:
            summ, rows, _ = run_suite(lm, NATURAL, sets, which, cfg.eval.ays_max_new_tokens,
                                      direction=u if op == "abl" else None, desc=f"{cell_id(source, op)}:")
        write_json(cell_path(d, source, op), dict(
            target=NATURAL, source=source, kind=kind_of(source), layer=layer, op=op, dose=k, coef=coef, sd=sdu,
            summaries=summ, rows=rows, wall_s=time.time() - t1))
        eta = (time.time() - t0) / (n + 1) * (len(jobs) - n - 1)
        fl = " ".join(f"{e}={summ[e]['flip_rate']:.3f}" for e in ("ays_in", "ays_syco_full") if e in summ)
        acc = (summ.get("mmlu") or {}).get("acc", float("nan"))
        log(f"[steer] {n + 1}/{len(jobs)} {cell_id(source, op):>22} {fl} mmlu={acc:.3f} "
            f"({time.time() - t1:.0f}s, eta {eta / 60:.0f} min)")
    if not todo(cells, d):
        (d / ".done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
