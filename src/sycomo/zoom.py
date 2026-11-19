"""Stage zoom (post hoc): the recursive zoom-in. Where do sycophancy model organisms stop being faithful proxies
for the natural model's sycophancy?

Runs on a finished v2 run directory (cfg.out_dir): its natural model, the k_syc context-distilled capitulations
(data/syc_train.jsonl), the contrastive pairs (data/pairs.jsonl) and the eval items. Iteration k = 1..K:

  train    MO_k = the v2 `base` recipe (rank-8 LoRA on the k_syc capitulations, with v2 `base`'s LoRA init and data
           order at every k, so the MOs differ only in the training-time intervention) under INOCULATION:
           + inoc_dose * sd_j * u_j added to the residual stream at `layer` for every earlier direction j < k, in every
           training forward pass (preventative steering, as in Anthropic's persona-vectors work). The steer supplies
           t_1..t_{k-1}, so the data gives little gradient toward them and training has to express the rest of the
           trait. MO_1 has no inoculation (it reproduces v2 `base`). Every MO is evaluated, and its direction read,
           WITHOUT the steer. inoc_dose = x makes the inoculation the exact negative of the removal op s<x>.
  extract  raw_k = MO_k's diff-in-means at `layer` (sycophantic minus honest pair replies, mean over reply tokens, as
           in features), or with extract=model_diff the mean of MO_k minus natural activations on the same rows;
           masked dims (features.exclude_dims) dropped; u_k = raw_k Gram-Schmidt'd against u_1..u_{k-1}, so
           U = [u_1..u_k] is orthonormal. sd_k = std of the natural pair activations projected on u_k (as in steer).
  apply    t_1..t_k to the NATURAL model with each of remove_ops:
             abl   span(u_1..u_k) projected out of every residual writer (ablation.ablate_span)
             s<x>  - x * sum_j sd_j u_j added at `layer` (steer.steer: every position, prompt and generated tokens)
  cells (cells/<id>.json):
    natural__none                  the natural baseline through the same code path
    k<k>__nat__<op>_cum            natural, u_1..u_k
    k<k>__nat__<op>_rand<s>        natural, u_1..u_{k-1} + a random unit direction orthogonal to them: the control for
                                   u_k's marginal effect (k = 1: a random direction alone)
    k<k>__nat__<op>_solo           natural, u_k alone (k >= 2)
    k<k>__nat__supply              natural + MO_k's inoculation steer: how much of the trait the steer itself supplies
    k<k>__mo__none|abl_prev|abl_cum  MO_k plain; with span(u_<k) ablated (does it route around the earlier slices?);
                                   with span(u_<=k) ablated (does u_k capture what it routed through?)
  control (k in controls): a NAIVE MO_k trained with the removal (naive_op on u_<k) on instead of the inoculation,
           i.e. "fine-tune T minus t_<k on T". The loop's premise is that it relearns t_<k: little of its raw direction
           outside span(u_<k), a positive activation shift along u_<k, and abl_prev removing much of its trait.
           Cells k<k>__naive__none|abl_prev and k<k>__nat__<op>_naive (u_<k + the naive direction orthogonalized).
  stop     (main run only) when MO_k's raw direction lies in span(u_<k) (share of its squared norm outside it <
           min_resid_share: u_k would be noise; no cells for k), or when MO_k is not sycophantic (ays_in flip <
           natural's + min_mo_flip_gain: the inoculated slices already explain what the data teaches; k's MO, control
           and supply cells still run, its natural removal cells do not). stop.json records why.

summary.json (rebuilt after every iteration): per k and op, the natural flip rate under cum, the marginal drop
cum_{k-1} -> cum_k and its excess over the random control with paired item-bootstrap CIs; MO_k's flip rates (none,
prev, cum); supply; geometry; MMLU; the first-batch training loss with and without the inoculation. MO_k stops being a
faithful proxy at the k where u_k still removes MO_k's own trait but its natural excess is ~0.

Resumable: per arm the adapter (train_log.json written last) and the direction (directions/<mo>.json written last),
per cell its file. Changed zoom settings in an existing output directory stop the stage (use another zoom.out);
iterations and n_random may grow. Preflight (zoom.preflight_items=N): fresh <run>/<out>_preflight/, items cut to N,
preflight_iterations iterations of preflight_train_examples-example MOs, stop rules off, hard checks at the end.
"""

from __future__ import annotations

import dataclasses
import gc
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .design import NATURAL, Cell, Layout
from .util import derive_seed, log, read_json, read_jsonl, rng, save_npz, write_json

MO, NAIVE = "mo", "naive"
NAT_NONE = "natural__none"
SUMMARY_EVALS = ("ays_in", "ays_syco_full", "ays_med")
MAY_CHANGE = ("iterations", "n_random", "preflight_items", "preflight_iterations", "preflight_train_examples")
NAT_REMOVAL = ("cum", "rand", "solo", "naive_nat")  # the natural removal cells an MO stop skips


# ---------------------------------------------------------------- names and paths

def out_dir(cfg) -> Path:
    z = cfg.zoom
    return Layout(cfg.out_dir).root / (z.out if z.preflight_items is None else f"{z.out}_preflight")


def mo_name(k: int, arm: str = MO) -> str:
    return f"zoom{k}" if arm == MO else f"zoom{k}-naive"


def arms(cfg, k: int) -> list[str]:
    return [MO] + ([NAIVE] if k in cfg.zoom.controls else [])


def adapter_dir(d: Path, k: int, arm: str = MO) -> Path:
    return Path(d) / "adapters" / mo_name(k, arm)


def direction_files(d: Path, k: int, arm: str = MO) -> tuple[Path, Path]:
    base = Path(d) / "directions" / mo_name(k, arm)
    return base.with_suffix(".npz"), base.with_suffix(".json")


def cell_path(d: Path, cid: str) -> Path:
    return Path(d) / "cells" / f"{cid}.json"


# ---------------------------------------------------------------- directions

def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        raise ValueError("zero or non-finite vector")
    return v / n


def orthogonalize(v, U) -> tuple[np.ndarray, float]:
    """(unit residual of v outside span(U), share of v's squared norm outside span(U)); U = orthonormal rows, possibly
    none. Classical Gram-Schmidt run twice. A zero residual comes back as zeros (share 0): the caller decides."""
    v = np.asarray(v, dtype=np.float64)
    r = v.copy()
    for _ in range(2):
        for u in U:
            r = r - (r @ u) * u
    n2 = float(v @ v)
    share = float(r @ r) / n2 if n2 > 0 else 0.0
    if share < 1e-12:  # v lies in span(U) up to rounding: no direction (normalizing would amplify float noise)
        return np.zeros_like(r), share
    return r / np.linalg.norm(r), share


def _mask(cfg, v) -> np.ndarray:
    from .features import mask_dims

    return mask_dims(v, cfg.features.exclude_dims) if cfg.features.exclude_dims else np.asarray(v, dtype=np.float64)


def random_direction(cfg, k: int, s: int, X_nat: np.ndarray, U_prev) -> np.ndarray:
    """Random unit direction (transfer.random_kind) orthogonal to u_<k, masked like the MO directions."""
    from .ablation import random_unit

    dm, seed = X_nat.shape[1], derive_seed(cfg.seed, "zoom_random", k, s)
    v = random_unit(dm, seed) if cfg.transfer.random_kind == "isotropic" else random_unit(dm, seed, X_nat - X_nat.mean(0))
    return orthogonalize(_mask(cfg, v), U_prev)[0]


def nat_acts_cached(cfg, d: Path, layer: int) -> np.ndarray | None:
    """The natural pair activations at `layer` where steer.natural_pair_acts looks for / caches them, else None."""
    src, cache = Layout(cfg.out_dir).acts / f"{NATURAL}_pairs.npz", Path(d) / f"natural_pairs_L{layer}.npz"
    if src.exists():
        return np.load(src)["acts"][:, layer].astype(np.float64)
    return np.load(cache)["acts"].astype(np.float64) if cache.exists() else None


def registry(cfg, d: Path, k_max: int, X_nat: np.ndarray | None) -> dict[str, tuple[np.ndarray, float]]:
    """{key: (unit direction, sd of the natural pair activations along it)} after iteration k_max: u<k> (main arm,
    orthonormal), n<k> (control arm, orthogonalized against u_<k>), r<k>_<s> (random, orthogonal to u_<k>)."""
    from .steer import projection_stats

    reg: dict = {}
    if k_max < 1:
        return reg
    if X_nat is None:
        raise RuntimeError("[zoom] natural pair activations missing for the direction registry")
    U: list[np.ndarray] = []
    for k in range(1, k_max + 1):
        npz, js = direction_files(d, k)
        if not js.exists():
            raise RuntimeError(f"[zoom] direction of {mo_name(k)} missing ({js})")
        for s in range(cfg.zoom.n_random):
            r = random_direction(cfg, k, s, X_nat, U)
            reg[f"r{k}_{s}"] = (r, projection_stats(X_nat, r)["sd"])
        npz_n, js_n = direction_files(d, k, NAIVE)
        if js_n.exists() and read_json(js_n)["sd"] is not None:
            reg[f"n{k}"] = (np.load(npz_n)["u"].astype(np.float64), float(read_json(js_n)["sd"]))
        rec = read_json(js)
        if rec["sd"] is None:
            raise RuntimeError(f"[zoom] {mo_name(k)} has no usable direction (residual share {rec['resid_share']})")
        u = np.load(npz)["u"].astype(np.float64)
        reg[f"u{k}"] = (u, float(rec["sd"]))
        U.append(u)
    return reg


def extract(cfg, d: Path, k: int, arm: str, lm, seqs, spans, X_nat: np.ndarray, layer: int, U_prev, tdirs: dict) -> dict:
    """MO_k's (arm's) direction, orthogonalized against u_<k, + geometry diagnostics. directions/<mo>.json last."""
    from .steer import projection_stats

    npz, js = direction_files(d, k, arm)
    if js.exists():
        return read_json(js)
    name = mo_name(k, arm)
    P = len(X_nat) // 2
    with lm.use(name):
        X = lm.pooled_acts(seqs, spans, f"{name}:pairs")[:, layer].astype(np.float64)
    raw = _mask(cfg, X[:P].mean(0) - X[P:].mean(0))
    md = _mask(cfg, (X - X_nat).mean(0))
    u, share = orthogonalize(raw if cfg.zoom.extract == "pairs" else md, U_prev)
    ok = bool(np.any(u))
    st = projection_stats(X_nat, u, P) if ok else {}

    def cos(a, b):
        return float(unit(a) @ unit(b))

    def gap(Z, v, sd):
        return float(((Z[:P] @ v).mean() - (Z[P:] @ v).mean()) / sd)

    prev = {f"u{j + 1}": v for j, v in enumerate(U_prev)}
    sd_prev = {j: projection_stats(X_nat, v)["sd"] for j, v in prev.items()}
    cos_prev = {j: cos(raw, v) for j, v in prev.items()}
    cos_ref = {f"cos_raw_{key}": cos(raw, tdirs[key]) for key in ("u_natural", "u_base") if key in tdirs}
    sd = st.get("sd")
    rec = dict(
        k=k, arm=arm, model=name, layer=layer, extract=cfg.zoom.extract, resid_share=share,
        raw_norm=float(np.linalg.norm(raw)), model_diff_norm=float(np.linalg.norm(md)), cos_raw_model_diff=cos(raw, md),
        cos_raw_prev=cos_prev,
        cos_model_diff_prev={j: cos(md, v) for j, v in prev.items()},
        # MO minus natural activation shift along each earlier slice, in that slice's natural sd: > 0 = the MO writes
        # more of t_j (relearned it); < 0 = less (e.g. it learned to cancel the inoculation)
        shift_prev_sd={j: float(((X - X_nat) @ v).mean() / sd_prev[j]) for j, v in prev.items()},
        gap_prev_sd={j: dict(mo=gap(X, v, sd_prev[j]), natural=gap(X_nat, v, sd_prev[j])) for j, v in prev.items()},
        **cos_ref,
        sd=sd, gap_in_sd=st.get("gap_in_sd"), mean_syc=st.get("mean_syc"), mean_honest=st.get("mean_honest"),
        mo_gap_in_sd=gap(X, u, sd) if ok else None)
    save_npz(npz, u=u, raw=raw, model_diff=md)
    write_json(js, rec)
    log(f"[zoom] {name}: share outside span(u_<{k}) {share:.3f}; cos(raw, u_j) "
        + (" ".join(f"{j}={c:.3f}" for j, c in cos_prev.items()) or "-")
        + (f"; cos(raw, u_base) {cos_ref['cos_raw_u_base']:.3f}" if "cos_raw_u_base" in cos_ref else "")
        + f"; sd {'none' if sd is None else f'{sd:.4f}'}")
    return rec


# ---------------------------------------------------------------- interventions

def dose(op: str) -> float:
    if not (op.startswith("s") and op[1:]):
        raise ValueError(f"not a steering op: {op!r}")
    return float(op[1:])


def intervention(decoder, layer: int, op: str, dirs, sds, sign: float = -1.0):
    """Context manager applying `op` with the unit directions `dirs` (orthonormal): abl = project span(dirs) out of
    every residual writer (removal only); s<x> = add sign * x * sum_j sds[j] * dirs[j] at `layer` (sign -1 removes,
    +1 inoculates); `none` or no directions = no-op."""
    from .ablation import ablate_span
    from .steer import steer

    dirs = [np.asarray(u, dtype=np.float64) for u in dirs]
    if op == "none" or not dirs:
        return nullcontext()
    if op == "abl":
        if sign > 0:
            raise ValueError("ablation has no inoculation form; use a steering op s<x>")
        return ablate_span(decoder, np.stack(dirs))
    sds = np.asarray(sds, dtype=np.float64)
    if len(sds) != len(dirs) or not np.all(np.isfinite(sds)) or np.any(sds <= 0):
        raise ValueError(f"need one positive sd per direction, got {sds.tolist()}")
    v = sign * dose(op) * sum(s * unit(u) for u, s in zip(dirs, sds))
    return steer(decoder, layer, v, float(np.linalg.norm(v)))


def train_spec(cfg, k: int, arm: str) -> tuple[str, float, list[str]]:
    """(op, sign, direction keys) active while training MO_k's `arm`: + inoc_dose on u_<k (mo), the removal
    naive_op on u_<k (naive), nothing at k = 1."""
    prev = [f"u{j}" for j in range(1, k)]
    if not prev:
        return "none", 0.0, []
    if arm == MO:
        return f"s{cfg.zoom.inoc_dose:g}", 1.0, prev
    return cfg.zoom.naive_op, -1.0, prev


def train_arm(cfg, d: Path, k: int, arm: str, reg: dict, layer: int, tok, monitor_items, n_syc: int) -> dict:
    from .train import train_cell

    path = adapter_dir(d, k, arm)
    log_path = path / "train_log.json"
    if log_path.exists():
        return read_json(log_path)
    op, sign, keys = train_spec(cfg, k, arm)
    dirs, sds = [reg[key][0] for key in keys], [reg[key][1] for key in keys]
    ctx = (lambda model: intervention(model.get_base_model().model, layer, op, dirs, sds, sign)) if keys else None
    log(f"[zoom] train {mo_name(k, arm)}: {n_syc} sycophantic examples, training intervention "
        f"{op if keys else 'none'} ({'+' if sign > 0 else '-'}) on {keys or '-'}")
    cell = Cell(mo_name(k, arm), cfg.grid.rank, "none", 0, n_syc)
    info = train_cell(cfg, cell, tok, monitor_items, {}, train_ctx=ctx, out_path=path, order_key="base")
    info["zoom"] = dict(k=k, arm=arm, train_op=op, train_sign=sign, train_dirs=keys, train_sds=sds, layer=layer)
    write_json(log_path, info)  # written last: marks the arm trained
    return info


# ---------------------------------------------------------------- plan

@dataclass(frozen=True)
class Job:
    cid: str
    k: int
    group: str          # none | cum | rand | solo | naive_nat | supply | mo_none | mo_prev | mo_cum | naive_none | naive_prev
    target: str         # NATURAL or an MO adapter name
    op: str             # none | abl | s<x>
    dirs: tuple[str, ...]
    sign: float         # -1 removal, +1 inoculation
    evals: tuple[str, ...]


def plan(cfg, k: int) -> list[Job]:
    """Iteration k's cells in run order: MO cells first (MO_k's none cell feeds the stop rule), then the natural ones,
    headline (cum) first."""
    z = cfg.zoom
    prev = tuple(f"u{j}" for j in range(1, k))
    cum = prev + (f"u{k}",)
    me, nv = mo_name(k), mo_name(k, NAIVE)
    nat, aux, moe = tuple(z.nat_evals), tuple(z.aux_evals), tuple(z.mo_evals)
    ctrl = k in z.controls
    jobs = []
    if k == 1:
        jobs.append(Job(NAT_NONE, 0, "none", NATURAL, "none", (), -1.0, tuple(dict.fromkeys(nat + tuple(z.none_only)))))
    jobs.append(Job(f"k{k}__mo__none", k, "mo_none", me, "none", (), -1.0, moe))
    if prev:
        jobs.append(Job(f"k{k}__mo__abl_prev", k, "mo_prev", me, "abl", prev, -1.0, moe))
    jobs.append(Job(f"k{k}__mo__abl_cum", k, "mo_cum", me, "abl", cum, -1.0, moe))
    if ctrl:
        jobs += [Job(f"k{k}__naive__none", k, "naive_none", nv, "none", (), -1.0, moe),
                 Job(f"k{k}__naive__abl_prev", k, "naive_prev", nv, "abl", prev, -1.0, moe)]
    jobs += [Job(f"k{k}__nat__{op}_cum", k, "cum", NATURAL, op, cum, -1.0, nat) for op in z.remove_ops]
    jobs += [Job(f"k{k}__nat__{op}_rand{s}", k, "rand", NATURAL, op, prev + (f"r{k}_{s}",), -1.0, nat)
             for op in z.remove_ops for s in range(z.n_random)]
    if prev:
        jobs += [Job(f"k{k}__nat__{op}_solo", k, "solo", NATURAL, op, (f"u{k}",), -1.0, aux) for op in z.remove_ops]
        if ctrl:
            jobs += [Job(f"k{k}__nat__{op}_naive", k, "naive_nat", NATURAL, op, prev + (f"n{k}",), -1.0, nat)
                     for op in z.remove_ops]
        jobs.append(Job(f"k{k}__nat__supply", k, "supply", NATURAL, f"s{z.inoc_dose:g}", prev, 1.0, aux))
    return jobs


def run_job(cfg, lm, job: Job, reg: dict, layer: int, sets: dict, d: Path) -> dict:
    from .evals import run_suite

    p = cell_path(d, job.cid)
    if p.exists():
        return read_json(p)
    miss = [key for key in job.dirs if key not in reg]
    if miss:
        raise RuntimeError(f"[zoom] {job.cid}: directions {miss} not available")
    dirs, sds = [reg[key][0] for key in job.dirs], [reg[key][1] for key in job.dirs]
    t1 = time.time()
    with intervention(lm.decoder, layer, job.op, dirs, sds, job.sign):
        summ, rows, _ = run_suite(lm, job.target, sets, list(job.evals), cfg.eval.ays_max_new_tokens, desc=f"{job.cid}:")
    cell = dict(**dataclasses.asdict(job), layer=layer, sds=sds, summaries=summ, rows=rows, wall_s=time.time() - t1)
    write_json(p, cell)
    msg = " ".join(f"{e}={s['flip_rate']:.3f}" for e, s in summ.items() if isinstance(s, dict) and s.get("flip_rate") is not None)
    acc = (summ.get("mmlu") or {}).get("acc")
    log(f"[zoom] {job.cid}: {msg}" + (f" mmlu={acc:.3f}" if acc is not None else "") + f" ({time.time() - t1:.0f}s)")
    return cell


def flip(cell: dict | None, ev: str) -> float | None:
    s = ((cell or {}).get("summaries") or {}).get(ev) or {}
    return s.get("flip_rate")


# ---------------------------------------------------------------- inputs and settings

def resolve_layer(cfg) -> int:
    if cfg.zoom.layer is not None:
        return int(cfg.zoom.layer)
    return int(read_json(Layout(cfg.out_dir).directions / "v2.json")["shared_layer"])


def check_inputs(cfg) -> None:
    L = Layout(cfg.out_dir)
    need = [L.root / "manifest.json", L.data / "syc_train.jsonl", L.data / "generate.json", L.data / "pairs.jsonl",
            L.items / "monitor_q.jsonl", L.items / "syco.jsonl", L.items / "eval_in.jsonl", L.items / "mmlu.jsonl"]
    if cfg.zoom.layer is None:
        need.append(L.directions / "v2.json")
    miss = [str(p.relative_to(L.root)) for p in need if not p.exists()]
    if miss:
        raise SystemExit(f"[zoom] {L.root} is missing {miss}: download the v2 run inputs first (scripts/zoom_pod.sh)")


def check_evals(cfg, sets: dict) -> None:
    from .evals import EVALS
    from .steer import set_name

    z = cfg.zoom
    bad = {}
    for ev in dict.fromkeys(z.nat_evals + z.aux_evals + z.mo_evals + z.none_only):
        if ev not in EVALS:
            bad[ev] = "not in evals.EVALS"
        elif not sets.get(set_name(ev)):
            bad[ev] = f"item set {set_name(ev)!r} missing or empty"
    if bad:
        raise SystemExit(f"[zoom] cannot run evals -> {bad}")


def check_settings(cfg, d: Path, layer: int) -> None:
    """An existing output directory must have been made with the same zoom settings (except MAY_CHANGE) and layer."""
    p = Path(d) / "run_config.json"
    if not p.exists():
        return
    old = read_json(p)
    new = dataclasses.asdict(cfg.zoom)
    diff = {k: (old.get("zoom", {}).get(k), v) for k, v in new.items()
            if k not in MAY_CHANGE and old.get("zoom", {}).get(k) != v}
    if old.get("layer") != layer:
        diff["layer (resolved)"] = (old.get("layer"), layer)
    if diff:
        raise SystemExit(f"[zoom] {d} was made with other zoom settings (old, new): {diff}. Use another zoom.out.")


def _free():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- stage

def stage_zoom(cfg) -> None:
    from .data import k_actual
    from .features import load_sets, pair_sequences
    from .modeling import LM, load_tokenizer
    from .steer import check_provenance, check_u_natural, natural_pair_acts, truncate_sets

    z = cfg.zoom
    pre = z.preflight_items is not None
    L = Layout(cfg.out_dir)
    check_inputs(cfg)
    man = check_provenance(cfg)
    d = out_dir(cfg)
    if pre and d.exists():
        shutil.rmtree(d)  # a preflight always tests the current code end to end
    layer = resolve_layer(cfg)
    check_settings(cfg, d, layer)
    K = z.preflight_iterations if pre else z.iterations
    sets = load_sets(cfg)
    if pre:
        sets = truncate_sets(sets, z.preflight_items)
    check_evals(cfg, sets)
    write_json(d / "run_config.json", dict(
        run_id=man.get("run_id"), layer=layer, zoom=dataclasses.asdict(z), iterations=K, preflight=pre,
        n_items={k: len(v) for k, v in sets.items() if isinstance(v, list)},
        invocation=(man.get("invocations") or [{}])[-1]))
    tok = load_tokenizer(cfg.model)
    monitor = read_jsonl(L.items / "monitor_q.jsonl")[: z.preflight_items if pre else None]
    n_syc = min(z.preflight_train_examples, k_actual(cfg)) if pre else k_actual(cfg)
    pairs = read_jsonl(L.data / "pairs.jsonl")
    tpath = L.directions / "transfer_dirs.npz"
    tdirs = {key: np.asarray(v, dtype=np.float64) for key, v in np.load(tpath).items()} if tpath.exists() else {}
    v2 = L.directions / "v2.json"
    shared = int(read_json(v2)["shared_layer"]) if v2.exists() else None
    stop_p = d / "stop.json"
    log(f"[zoom] -> {d}: {K} iterations, layer {layer}, inoculation s{z.inoc_dose:g}, remove {z.remove_ops}, "
        f"controls at {z.controls}, {n_syc} sycophantic examples per MO" + (" (PREFLIGHT)" if pre else ""))

    for k in range(1, K + 1):
        stop = read_json(stop_p) if stop_p.exists() else None
        if stop and (stop["k"] < k or (stop["k"] == k and stop["kind"] == "direction")):
            log(f"[zoom] stopped at iteration {stop['k']}: {stop['reason']}")
            break

        def wanted(job):  # after an MO stop: the iteration's diagnostic cells only (MO, control arm, supply)
            return not (stop and stop["k"] == k and job.group in NAT_REMOVAL)

        reg = registry(cfg, d, k - 1, nat_acts_cached(cfg, d, layer))
        for arm in arms(cfg, k):
            train_arm(cfg, d, k, arm, reg, layer, tok, monitor, n_syc)
        jobs = plan(cfg, k)
        if (all(direction_files(d, k, a)[1].exists() for a in arms(cfg, k))
                and all(cell_path(d, j.cid).exists() for j in jobs if wanted(j))):
            summarize(cfg, d)
            if stop and stop["k"] <= k:
                break
            continue
        lm = LM(cfg.model, adapters={mo_name(k, a): str(adapter_dir(d, k, a)) for a in arms(cfg, k)})
        try:
            if not 1 <= layer <= lm.n_layers:
                raise SystemExit(f"[zoom] layer {layer} outside 1..{lm.n_layers}")
            X_nat = natural_pair_acts(cfg, lm, layer, d).astype(np.float64)
            if k == 1 and "u_natural" in tdirs and shared == layer:
                check_u_natural(cfg, X_nat, tdirs["u_natural"])  # layer convention and recomputation
            seqs, spans = pair_sequences(lm, pairs)
            U_prev = [reg[f"u{j}"][0] for j in range(1, k)]
            rec = {a: extract(cfg, d, k, a, lm, seqs, spans, X_nat, layer, U_prev, tdirs) for a in arms(cfg, k)}
            if rec[MO]["sd"] is None or (not pre and rec[MO]["resid_share"] < z.min_resid_share):
                if pre:
                    raise SystemExit(f"[zoom] preflight: {mo_name(k)} has no direction outside span(u_<{k})")
                stop = dict(k=k, kind="direction", reason=f"{mo_name(k)}'s direction has share "
                            f"{rec[MO]['resid_share']:.4f} outside span(u_<{k}) (< {z.min_resid_share})")
                write_json(stop_p, stop)
                log(f"[zoom] stop: {stop['reason']}")
            else:
                reg = registry(cfg, d, k, X_nat)
                for job in jobs:
                    if not wanted(job):
                        continue
                    cell = run_job(cfg, lm, job, reg, layer, sets, d)
                    if job.group == "mo_none" and not pre and not stop:
                        nat_f, mo_f = flip(read_json(cell_path(d, NAT_NONE)), "ays_in"), flip(cell, "ays_in")
                        if nat_f is not None and mo_f is not None and mo_f < nat_f + z.min_mo_flip_gain:
                            stop = dict(k=k, kind="mo", reason=f"{mo_name(k)} is not sycophantic: ays_in flip "
                                        f"{mo_f:.3f} < natural {nat_f:.3f} + {z.min_mo_flip_gain}")
                            write_json(stop_p, stop)
                            log(f"[zoom] stop: {stop['reason']}; running this iteration's MO, control and supply "
                                f"cells, not its natural removal cells")
        finally:
            del lm
            _free()
        summarize(cfg, d)
        if stop and stop["k"] <= k:
            break
    summarize(cfg, d)
    if pre:
        preflight_checks(cfg, d, K)
    (d / ".done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


# ---------------------------------------------------------------- summary

def _flip_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    c = np.array([bool(r.get("correct1")) for r in rows], dtype=bool)
    f = np.array([bool(r.get("flipped")) and bool(r.get("correct1")) for r in rows], dtype=bool)
    return c, f


def boot_rates(conds: dict[str, list[dict]], diffs: dict[str, tuple[str, str]], B: int, g: np.random.Generator) -> dict:
    """Flip rates F = flips / turn-1 correct (each condition its own denominator: turn 1 can change under an
    intervention) of conditions run on the same items, and F[a] - F[b] for each named pair with a paired item
    bootstrap (items aligned by qid, resampled jointly)."""
    names = list(conds)
    qids = sorted(set.intersection(*[{r["qid"] for r in conds[n]} for n in names])) if names else []
    out: dict = dict(n_items=len(qids), F={}, diff={})
    if not qids:
        return out
    arr = {}
    for n in names:
        by = {r["qid"]: r for r in conds[n]}
        arr[n] = _flip_arrays([by[q] for q in qids])
    idx = g.integers(0, len(qids), size=(B, len(qids)))
    Fb = {}
    for n, (c, f) in arr.items():
        out["F"][n] = float(f.sum() / c.sum()) if c.sum() else float("nan")
        with np.errstate(divide="ignore", invalid="ignore"):
            Fb[n] = f[idx].sum(1) / c[idx].sum(1)
    for key, (a, b) in diffs.items():
        dv = Fb[a] - Fb[b]
        dv = dv[np.isfinite(dv)]
        lo, hi = np.quantile(dv, [0.025, 0.975]) if len(dv) > 10 else (float("nan"), float("nan"))
        out["diff"][key] = dict(est=out["F"][a] - out["F"][b], lo=float(lo), hi=float(hi))
    return out


def _brief(cell: dict | None) -> dict:
    out = {}
    for ev, s in ((cell or {}).get("summaries") or {}).items():
        if isinstance(s, dict):
            v = s.get("flip_rate", s.get("acc"))
            if v is not None:
                out[ev] = v
    if "ays_in" in out and "ays_in_neutral" in out:
        out["S_in"] = out["ays_in"] - out["ays_in_neutral"]
    return out


def _train_brief(d: Path, k: int, arm: str) -> dict | None:
    p = adapter_dir(d, k, arm) / "train_log.json"
    if not p.exists():
        return None
    t = read_json(p)
    return dict(steps=t["steps"], loss0=t.get("loss0"), loss_first=t["loss"][0]["loss"] if t["loss"] else None,
                loss_last=t["loss"][-1]["loss"] if t["loss"] else None, quick_flip_final=t["monitor"][-1]["p_switch"],
                steps_to_target=t["steps_to_target"], skipped_steps=t["skipped_steps"], wall_s=t["wall_s"],
                train=t.get("zoom"))


def summarize(cfg, d: Path) -> dict:
    """summary.json from the cells and logs on disk (CPU only)."""
    z, B = cfg.zoom, cfg.analysis.bootstrap_n
    cdir = Path(d) / "cells"
    cells = {p.stem: read_json(p) for p in sorted(cdir.glob("*.json"))} if cdir.exists() else {}

    def rows(cid, ev):
        return ((cells.get(cid) or {}).get("rows") or {}).get(ev)

    def boot(names, diffs, ev, *key):
        conds: dict = {n: rows(n, ev) for n in names if rows(n, ev)}
        diffs = {k: ab for k, ab in diffs.items() if ab[0] in conds and ab[1] in conds}
        return boot_rates(conds, diffs, B, rng(cfg.seed, "zoom_boot", ev, *key)) if diffs else None

    stop = Path(d) / "stop.json"
    out: dict = dict(layer=(read_json(Path(d) / "run_config.json") if (Path(d) / "run_config.json").exists() else {}).get("layer"),
               zoom=dataclasses.asdict(z), stop=read_json(stop) if stop.exists() else None,
               natural=_brief(cells.get(NAT_NONE)), iterations=[])
    k = 1
    while direction_files(d, k)[1].exists():
        it = dict(k=k, direction=read_json(direction_files(d, k)[1]), train=_train_brief(d, k, MO),
                  cells={cid: _brief(c) for cid, c in cells.items() if cid.startswith(f"k{k}__")})
        if direction_files(d, k, NAIVE)[1].exists():
            it["naive"] = dict(direction=read_json(direction_files(d, k, NAIVE)[1]), train=_train_brief(d, k, NAIVE))
        nat, mo, nv, sup = {}, {}, {}, {}
        for ev in SUMMARY_EVALS:
            for op in z.remove_ops:
                prev, cum = (NAT_NONE if k == 1 else f"k{k - 1}__nat__{op}_cum"), f"k{k}__nat__{op}_cum"
                rands = [f"k{k}__nat__{op}_rand{s}" for s in range(z.n_random)]
                naive = f"k{k}__nat__{op}_naive"
                diffs = {"marginal": (prev, cum), "naive_marginal": (prev, naive), "excess_over_naive": (naive, cum)}
                for s, r in enumerate(rands):
                    diffs[f"marginal_rand{s}"] = (prev, r)
                    diffs[f"excess_rand{s}"] = (r, cum)  # > 0: u_k removes more than a random direction would
                b = boot([prev, cum, naive, *rands], diffs, ev, k, op)
                if b:
                    nat[f"{op}/{ev}"] = b
            m0, mp, mc = f"k{k}__mo__none", f"k{k}__mo__abl_prev", f"k{k}__mo__abl_cum"
            b = boot([m0, mp, mc], ({"prev_drop": (m0, mp), "self_drop": (mp, mc)} if k > 1 else {"self_drop": (m0, mc)}),
                     ev, k, "mo")
            if b:
                mo[ev] = b
            b = boot([f"k{k}__naive__none", f"k{k}__naive__abl_prev"],
                     {"prev_drop": (f"k{k}__naive__none", f"k{k}__naive__abl_prev")}, ev, k, "naive")
            if b:
                nv[ev] = b
            b = boot([NAT_NONE, f"k{k}__nat__supply"], {"supply_gain": (f"k{k}__nat__supply", NAT_NONE)}, ev, k, "supply")
            if b:
                sup[ev] = b
        it.update(natural_marginal=nat, mo=mo, naive_cells=nv, supply=sup)
        out["iterations"].append(it)
        k += 1
    write_json(Path(d) / "summary.json", out)
    _log_table(out, z)
    return out


def _fmt(b: dict | None, key: str) -> str:
    v = ((b or {}).get("diff") or {}).get(key)
    if not v or v.get("est") is None or not np.isfinite(v["est"]):
        return "-"
    return f"{v['est']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]"


def _log_table(out: dict, z) -> None:
    ev = "ays_syco_full"
    for it in out["iterations"]:
        k, dr, c = it["k"], it["direction"], it["cells"]
        mo = [c.get(f"k{k}__mo__{g}", {}).get("ays_in") for g in ("none", "abl_prev", "abl_cum")]
        parts = [f"[zoom] k={k} share {dr['resid_share']:.3f}",
                 "MO ays_in none/prev/cum " + "/".join("-" if v is None else f"{v:.3f}" for v in mo)]
        for op in z.remove_ops:
            b = it["natural_marginal"].get(f"{op}/{ev}")
            F = (b or {}).get("F", {}).get(f"k{k}__nat__{op}_cum")
            mm = c.get(f"k{k}__nat__{op}_cum", {}).get("mmlu")
            parts.append(f"{op}: nat {ev} {'-' if F is None else f'{F:.3f}'} marginal {_fmt(b, 'marginal')} "
                         f"excess {_fmt(b, 'excess_rand0')} mmlu {'-' if mm is None else f'{mm:.3f}'}")
        log(" | ".join(parts))
    if out.get("stop"):
        log(f"[zoom] stopped: {out['stop']['reason']}")


# ---------------------------------------------------------------- preflight

def preflight_checks(cfg, d: Path, K: int) -> None:
    """Hard checks on a preflight directory; SystemExit listing every failure."""
    fails = []
    U = []
    for k in range(1, K + 1):
        for a in arms(cfg, k):
            tl = adapter_dir(d, k, a) / "train_log.json"
            if not tl.exists():
                fails.append(f"{mo_name(k, a)}: no train_log.json")
                continue
            t = read_json(tl)
            if t["skipped_steps"] or any(x["loss"] is None or not np.isfinite(x["loss"]) for x in t["loss"]):
                fails.append(f"{mo_name(k, a)}: skipped steps or non-finite losses")
            if k > 1:
                l0 = t.get("loss0") or {}
                if l0.get("plain") is None or l0.get("ctx") is None or abs(l0["plain"] - l0["ctx"]) < 1e-6:
                    fails.append(f"{mo_name(k, a)}: the training intervention did not change the loss ({l0})")
            if not direction_files(d, k, a)[1].exists():
                fails.append(f"{mo_name(k, a)}: no direction")
            elif a == MO:
                U.append(np.load(direction_files(d, k, a)[0])["u"].astype(np.float64))
        for job in plan(cfg, k):
            p = cell_path(d, job.cid)
            if not p.exists():
                fails.append(f"cell {job.cid} missing")
                continue
            for ev, s in read_json(p)["summaries"].items():
                if not isinstance(s, dict):
                    continue
                n_ok = s.get("n_correct1", s.get("n", 1))
                for key in ("flip_rate", "acc"):
                    if key in s and n_ok and (s[key] is None or not np.isfinite(s[key])):
                        fails.append(f"cell {job.cid}: {ev}.{key} = {s[key]}")
    if U:
        G = np.stack(U) @ np.stack(U).T
        err = float(np.abs(G - np.eye(len(U))).max())
        if err > 1e-6:
            fails.append(f"u_1..u_{len(U)} not orthonormal (max |UU^T - I| = {err:.2e})")
    if fails:
        raise SystemExit("[zoom] preflight FAILED:\n  " + "\n  ".join(fails))
    log(f"[zoom] preflight passed: {K} iterations, {sum(len(plan(cfg, k)) for k in range(1, K + 1))} cells, "
        f"directions orthonormal")
