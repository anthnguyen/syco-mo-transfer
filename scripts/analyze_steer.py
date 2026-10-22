#!/usr/bin/env python
"""Analyze the steer stage (src/sycomo/steer.py) of one run: every cell against the `none` cell.

    .venv/bin/python scripts/analyze_steer.py <run dir> [--B 2000] [--preflight]

Writes <run>/steer/analysis/ (or steer_preflight/analysis/):
  steer.csv          one row per (cell, eval):
                       trait evals (ays_*, mcq_bias, aita): T = relative drop of the trait vs `none` with a paired
                         item bootstrap CI (analysis.transfer_entry: AYS flip rate, mcq-bias net switch toward the
                         hint, AITA both-NTA); mcq_bias also per bias type (eval "mcq_bias/<type>": the
                         "different biases" axis); "S:ays_in" = T on S = flip(pushback) - flip(neutral re-ask)
                       *_neutral: the neutral re-ask flip rate (base, abl; no T): steering must not make the model
                         flip without pushback
                       mmlu: accuracy (base, abl); damage = base - cell with a paired bootstrap CI
                     every row also carries the cell's damage, the random-direction null for its (op, eval)
                     (mean and max T over rand*) and beats_random = T > max random T
  steer_groups.json  per op x trait eval: dT = mean T(conditional) - mean T(leaky) over the MO directions
                     (leaky: u_base, u_U-self, u_U-self-5k; conditional: u_A-self, u_C-self, u_C-self-1800, as in
                     docs/results_v2/leaky_cond_stats.py) with an item bootstrap (all cells resampled on the same
                     items; MOs fixed), p_boot Holm-adjusted across the evals of each op, and the exact cell-level
                     permutation p over the 3/3 splits (min 0.10); plus the baseline S per AYS eval.
The mentor's hypothesis (narrow MOs -> narrow vectors) predicts dT ~ 0 on ays_in (trained domain) and dT < 0
off-domain (other datasets, biases, AITA).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sycomo.analysis import (aita_both_units, align, flip_arrays, holm, mcq_net_switch_units,  # noqa: E402
                             s_drop_boot, transfer_entry)

LEAKY = ["u_base", "u_U-self", "u_U-self-5k"]
CONDL = ["u_A-self", "u_C-self", "u_C-self-1800"]
COLS = ["source", "op", "kind", "group", "dose", "coef", "sd", "eval", "metric", "T", "lo", "hi", "base", "abl",
        "abs_drop", "n", "mmlu_base", "mmlu_cell", "damage", "damage_lo", "damage_hi", "T_rand_mean", "T_rand_max",
        "beats_random"]


def group_of(source: str) -> str:
    if source in LEAKY:
        return "leaky"
    if source in CONDL:
        return "conditional"
    if source.startswith("rand"):
        return "random"
    return "natural" if source in ("u_natural", "nat_onpolicy") else "other"


def is_trait(ev: str) -> bool:
    return (ev.startswith("ays") and not ev.endswith("_neutral")) or ev.startswith("mcq_bias") or ev == "aita"


def bias_types(rows: list[dict]) -> list[str]:
    return sorted({r.get("bias_type") for r in rows} - {None, "unbiased"})


def rows_of(cell: dict, ev: str) -> list[dict] | None:
    """Rows of eval ev; "mcq_bias/<type>" = the mcq_bias rows of that bias type plus the unbiased baselines."""
    if "/" in ev:
        base, bt = ev.split("/", 1)
        r = (cell.get("rows") or {}).get(base)
        return None if r is None else [x for x in r if x.get("bias_type") in ("unbiased", bt)]
    return (cell.get("rows") or {}).get(ev)


def sub_cell(cell: dict, ev: str) -> dict:
    return dict(rows={ev.split("/")[0]: rows_of(cell, ev)}, summaries={})


def trait_evals(base: dict, cells: list[dict]) -> list[str]:
    evs = [e for e in (base.get("rows") or {}) if is_trait(e) and any(e in (c.get("rows") or {}) for c in cells)]
    out = []
    for e in evs:
        out.append(e)
        if e == "mcq_bias":
            out += [f"mcq_bias/{bt}" for bt in bias_types(base["rows"][e])]
    return out


def entry(ev: str, base: dict, cell: dict, B: int, g) -> dict | None:
    if "/" in ev:
        return transfer_entry(ev.split("/")[0], sub_cell(base, ev), sub_cell(cell, ev), B, g)
    return transfer_entry(ev, base, cell, B, g)


def mmlu_damage(base: dict, cell: dict, B: int, g) -> dict:
    br, cr = rows_of(base, "mmlu"), rows_of(cell, "mmlu")
    if not br or not cr:
        return dict(base=np.nan, abl=np.nan, damage=np.nan, lo=np.nan, hi=np.nan, n=0)
    br, cr = align(br, cr)
    b = np.array([bool(r["correct"]) for r in br], float)
    a = np.array([bool(r["correct"]) for r in cr], float)
    idx = g.integers(0, len(b), size=(B, len(b)))
    d = b[idx].mean(1) - a[idx].mean(1)
    return dict(base=float(b.mean()), abl=float(a.mean()), damage=float(b.mean() - a.mean()),
                lo=float(np.quantile(d, 0.025)), hi=float(np.quantile(d, 0.975)), n=len(b))


def flip_rate(rows) -> float:
    if not rows:
        return np.nan
    c, f = flip_arrays(rows)
    return float(f.sum() / c.sum()) if c.sum() else np.nan


def cell_table(base: dict, cells: list[dict], B: int, g) -> pd.DataFrame:
    evs = trait_evals(base, cells)
    out = []
    for c in cells:
        meta = dict(source=c["source"], op=c["op"], kind=c.get("kind"), group=group_of(c["source"]),
                    dose=c.get("dose"), coef=c.get("coef"), sd=c.get("sd"))
        dm = mmlu_damage(base, c, B, g)
        dmg = dict(mmlu_base=dm["base"], mmlu_cell=dm["abl"], damage=dm["damage"], damage_lo=dm["lo"],
                   damage_hi=dm["hi"])
        for ev in evs:
            if rows_of(c, ev) is None:
                continue
            r = entry(ev, base, c, B, g)
            if r is not None:
                out.append({**meta, "eval": ev, "metric": "T", **{k: r.get(k) for k in ("T", "lo", "hi", "base",
                                                                                        "abl", "abs_drop", "n")}, **dmg})
        if all(rows_of(x, e) for x in (base, c) for e in ("ays_in", "ays_in_neutral")):
            r = s_drop_boot(base, c, "ays_in", B, g)
            if r is not None:
                out.append({**meta, "eval": "S:ays_in", "metric": "T_S", **r, **dmg})
        for ev in (c.get("rows") or {}):
            if ev.endswith("_neutral") and rows_of(base, ev):
                fb, fa = flip_rate(rows_of(base, ev)), flip_rate(rows_of(c, ev))
                out.append({**meta, "eval": ev, "metric": "neutral_flip", "T": np.nan, "lo": np.nan, "hi": np.nan,
                            "base": fb, "abl": fa, "abs_drop": fb - fa, "n": len(rows_of(c, ev)), **dmg})
        if np.isfinite(dm["base"]):
            out.append({**meta, "eval": "mmlu", "metric": "acc", "T": np.nan, "lo": dm["lo"], "hi": dm["hi"],
                        "base": dm["base"], "abl": dm["abl"], "abs_drop": dm["damage"], "n": dm["n"], **dmg})
    df = pd.DataFrame(out, columns=[c for c in COLS if c not in ("T_rand_mean", "T_rand_max", "beats_random")])
    rnd = df[(df.group == "random") & df["T"].notna()].groupby(["op", "eval"])["T"]
    df = df.merge(rnd.mean().rename("T_rand_mean").reset_index(), on=["op", "eval"], how="left")
    df = df.merge(rnd.max().rename("T_rand_max").reset_index(), on=["op", "eval"], how="left")
    df["beats_random"] = (df["T"] > df["T_rand_max"]).where(df["T"].notna() & df["T_rand_max"].notna())
    return df[COLS]


# ---------------------------------------------------------------- leaky vs conditional (leaky_cond_stats.py)

def unit_arrays(ev, base_rows, cell_rows):
    """Per-item arrays over the items shared by the baseline and every cell, in one order."""
    if ev.startswith("ays"):
        keys = set(r["qid"] for r in base_rows)
        for rows in cell_rows.values():
            keys &= set(r["qid"] for r in rows)
        keys = sorted(keys)

        def arr(rows):
            d = {r["qid"]: r for r in rows}
            c, f = flip_arrays([d[k] for k in keys])
            return np.stack([c, f]).astype(float)   # (2, n): initially correct, flipped
        return arr(base_rows), {m: arr(r) for m, r in cell_rows.items()}
    units = mcq_net_switch_units if ev.startswith("mcq") else aita_both_units
    ub = units(base_rows)
    uc = {m: units(r) for m, r in cell_rows.items()}
    keys = sorted(set(ub).intersection(*[set(u) for u in uc.values()]), key=str)
    return np.array([ub[k] for k in keys]), {m: np.array([u[k] for k in keys]) for m, u in uc.items()}


def T_of(ev, b, a, idx):
    """T for each bootstrap row of item indices idx (B, n)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if ev.startswith("ays"):
            Fb = b[1][idx].sum(-1) / b[0][idx].sum(-1)
            Fa = a[1][idx].sum(-1) / a[0][idx].sum(-1)
        else:
            Fb, Fa = b[idx].mean(-1), a[idx].mean(-1)
        return (Fb - Fa) / Fb


def group_contrast(ev: str, base: dict, by_src: dict, B: int, g) -> dict | None:
    leaky = [m for m in LEAKY if m in by_src and rows_of(by_src[m], ev)]
    cond = [m for m in CONDL if m in by_src and rows_of(by_src[m], ev)]
    if not leaky or not cond or not rows_of(base, ev):
        return None
    b, a = unit_arrays(ev, rows_of(base, ev), {m: rows_of(by_src[m], ev) for m in leaky + cond})
    n = b.shape[-1]
    if n == 0:
        return None
    Tpt = {m: float(T_of(ev, b, a[m], np.arange(n)[None])[0]) for m in leaky + cond}
    d = float(np.mean([Tpt[m] for m in cond]) - np.mean([Tpt[m] for m in leaky]))
    idx = g.integers(0, n, size=(B, n))
    Tb = {m: T_of(ev, b, a[m], idx) for m in leaky + cond}
    db = np.mean([Tb[m] for m in cond], 0) - np.mean([Tb[m] for m in leaky], 0)
    db = db[np.isfinite(db)]
    ok = len(db) > 10 and np.isfinite(d)
    p_boot = float(min(1.0, 2 * min((db <= 0).mean(), (db >= 0).mean()))) if ok else np.nan
    allm, k = leaky + cond, len(leaky)
    null = [np.mean([Tpt[m] for m in allm if m not in s]) - np.mean([Tpt[m] for m in s])
            for s in itertools.combinations(allm, k)]
    p_perm = float(np.mean(np.abs(null) >= abs(d) - 1e-12)) if np.isfinite(d) else np.nan
    return dict(T=Tpt, leaky=leaky, conditional=cond, dT=d,
                lo=float(np.quantile(db, 0.025)) if ok else np.nan, hi=float(np.quantile(db, 0.975)) if ok else np.nan,
                p_boot=p_boot, p_perm_cells=p_perm, n_items=int(n))


def groups(base: dict, cells: list[dict], B: int, g) -> dict:
    evs = trait_evals(base, cells)
    out = {}
    for op in dict.fromkeys(c["op"] for c in cells):
        by_src = {c["source"]: c for c in cells if c["op"] == op}
        res = {ev: r for ev in evs if (r := group_contrast(ev, base, by_src, B, g)) is not None}
        for key in ("p_boot", "p_perm_cells"):
            adj = holm([res[e][key] for e in res])
            for e, p in zip(res, adj):
                res[e][f"{key}_holm"] = float(p)
        out[op] = res
    return out


def baseline_S(base: dict) -> dict:
    out = {}
    for ev, rows in (base.get("rows") or {}).items():
        if ev.startswith("ays") and not ev.endswith("_neutral"):
            f, fn = flip_rate(rows), flip_rate((base["rows"]).get(f"{ev}_neutral"))
            out[ev] = dict(flip=f, flip_neutral=fn, S=f - fn if np.isfinite(fn) else np.nan,
                           n_correct1=int(flip_arrays(rows)[0].sum()))
    return out


def load_cells(d: Path) -> tuple[dict, list[dict]]:
    files = sorted((d / "cells").glob("*.json"))
    cells = [json.loads(f.read_text()) for f in files]
    base = [c for c in cells if c.get("source") == "none"]
    if not base:
        raise SystemExit(f"no none.json in {d / 'cells'}")
    return base[0], [c for c in cells if c.get("source") != "none"]


def analyze(d: Path, B: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    g = np.random.default_rng(seed)
    base, cells = load_cells(d)
    df = cell_table(base, cells, B, g)
    gr = dict(B=B, seed=seed, leaky=LEAKY, conditional=CONDL, baseline=baseline_S(base), groups=groups(base, cells, B, g),
              cells=sorted(f"{c['source']}__{c['op']}" for c in cells))
    out = d / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "steer.csv", index=False)
    (out / "steer_groups.json").write_text(json.dumps(gr, indent=1, default=float))
    return df, gr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preflight", action="store_true", help="analyze steer_preflight/ instead of steer/")
    a = ap.parse_args()
    d = Path(a.run) / ("steer_preflight" if a.preflight else "steer")
    df, gr = analyze(d, a.B, a.seed)
    print(f"{len(df)} rows -> {d / 'analysis' / 'steer.csv'}")
    t = df[df.metric == "T"].pivot_table(index=["op", "source"], columns="eval", values="T")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 2):
        print(t)
    for op, res in gr["groups"].items():
        for ev, r in res.items():
            print(f"{op:4s} {ev:34s} dT(cond-leaky)={r['dT']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] "
                  f"p_boot={r['p_boot']:.4f} holm={r['p_boot_holm']:.3f} p_cells={r['p_perm_cells']:.2f} n={r['n_items']}")


if __name__ == "__main__":
    main()
