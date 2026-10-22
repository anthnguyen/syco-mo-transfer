"""Report for up_scan + block_ko.

Q1: upstream responsiveness R = (flips removed in MO m by dir<b0) / (flips removed in natural by dir<b0),
    absolute flip-rate drops; paired bootstrap over items. Spearman(R, T_to_natural) across MOs.
Q-spoof: T = (F(m|none) - F(m|whole:src)) / F(m|none) for each MO and whole-model ablated direction.
Q2: per-block adapter knockout in suffix MOs.

Usage: python spoof/scan_report.py D   (D has up_scan/*.json, block_ko/*.json, transfer_features.csv)
Prints markdown tables, saves D/scan_report.json and three figures into D.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAT = "natural"
B = 2000
EVALS = ("ays_in", "ays_syco")
DIRS = ("u_base", "u_natural")
B0S = (16, 31)
MAIN = ("u_base", 31)
SEEDS = {"base": ["base", "base-s1", "base-s2"], "C-self": ["C-self", "C-self-s1", "C-self-s2"]}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
GRID, TXT = "#e4e3df", "#3a3a38"

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": TXT, "axes.labelcolor": TXT,
    "xtick.color": TXT, "ytick.color": TXT, "text.color": TXT, "axes.titlecolor": TXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9, "lines.linewidth": 1.0, "axes.linewidth": 0.6,
})


def load_dir(d):
    res = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        for k, v in json.load(open(f)).items():
            res.setdefault(k, v)  # natural repeats in every shard
    return res


def rows_arr(rows, qids):
    m = {r["qid"]: (r["c"], r["f"]) for r in rows}
    return (np.array([m[q][0] for q in qids], float), np.array([m[q][1] for q in qids], float))


def flip(c, f, I=None):
    """Flip rate; with I [B, n] bootstrap indices returns a [B] vector."""
    if I is None:
        return (c * f).sum() / max(c.sum(), 1e-9)
    cs = c[I].sum(1)
    return (c[I] * f[I]).sum(1) / np.maximum(cs, 1e-9)


def ranks(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):  # average ties
        t = x == v
        r[t] = r[t].mean()
    return r


def spearman(x, y, nperm=10000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        return dict(rho=float("nan"), p=float("nan"), n=n)
    rx, ry = ranks(x), ranks(y)
    rho = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(nperm)])
    p = (1 + (np.abs(perm) >= abs(rho) - 1e-12).sum()) / (nperm + 1)
    return dict(rho=float(rho), p=float(p), n=n)


def f3(x):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.3f}"


def ci(v, lo, hi):
    return f"{f3(v)} [{f3(lo)}, {f3(hi)}]"


def md_table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()


def read_features(path):
    T = {}
    if not os.path.exists(path):
        print(f"[warn] {path} missing; no transfer correlation\n")
        return T
    for r in csv.DictReader(open(path)):
        def g(k):
            try:
                return float(r.get(k, "nan"))
            except ValueError:
                return float("nan")
        T[r["model"]] = (g("T_to_natural"), g("T_to_natural_lo"), g("T_to_natural_hi"))
    return T


def models_of(R):
    seen = []
    for k in R:
        m = k.split("|")[0]
        if m not in seen:
            seen.append(m)
    return seen


# ---------------- Q1 ----------------

def upstream(R, out):
    models = [m for m in models_of(R) if m != NAT]
    rng = np.random.default_rng(0)
    resp = {}  # (dir, b0, eval) -> model -> dict
    for ev in EVALS:
        nat_rows = R.get(f"{NAT}|none", {}).get(ev)
        if nat_rows is None:
            continue
        for dn in DIRS:
            for b0 in B0S:
                kd = f"{dn}<{b0}"
                if ev not in R.get(f"{NAT}|{kd}", {}):
                    continue
                spec = {}
                for m in models:
                    keys = [f"{m}|none", f"{m}|{kd}", f"{NAT}|none", f"{NAT}|{kd}"]
                    if not all(k in R and ev in R[k] for k in keys):
                        continue
                    qs = set.intersection(*[{r["qid"] for r in R[k][ev]["rows"]} for k in keys])
                    qids = sorted(qs, key=str)
                    A = [rows_arr(R[k][ev]["rows"], qids) for k in keys]
                    I = rng.integers(0, len(qids), size=(B, len(qids)))
                    pt = [flip(*a) for a in A]
                    bs = [flip(*a, I) for a in A]
                    dm, dn_ = pt[0] - pt[1], pt[2] - pt[3]
                    bm, bn = bs[0] - bs[1], bs[2] - bs[3]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rb = bm / bn
                    rb = rb[np.isfinite(rb)]
                    spec[m] = dict(F_none=pt[0], F_dir=pt[1], dF=dm, dF_lo=np.percentile(bm, 2.5),
                                   dF_hi=np.percentile(bm, 97.5), dF_nat=dn_,
                                   R=dm / dn_ if dn_ != 0 else float("nan"),
                                   R_lo=np.percentile(rb, 2.5) if len(rb) else float("nan"),
                                   R_hi=np.percentile(rb, 97.5) if len(rb) else float("nan"), n_items=len(qids))
                resp[(dn, b0, ev)] = spec
    out["upstream"] = {f"{dn}<{b0}|{ev}": {m: {k: float(v) for k, v in d.items()} for m, d in s.items()}
                       for (dn, b0, ev), s in resp.items()}
    for (dn, b0, ev), spec in resp.items():
        if not spec:
            continue
        nat = next(iter(spec.values()))["dF_nat"]
        print(f"### Upstream responsiveness: {dn}<{b0}, {ev} (natural dF = {f3(nat)})\n")
        md_table(["model", "F none", "F dir", "dF [95% CI]", "R [95% CI]", "n items"],
                 [[m, f3(d["F_none"]), f3(d["F_dir"]), ci(d["dF"], d["dF_lo"], d["dF_hi"]),
                   ci(d["R"], d["R_lo"], d["R_hi"]), d["n_items"]] for m, d in spec.items()])
    return resp


def seed_avg(vals):
    """vals: model -> (R, T). Averages seed replicates into one point."""
    out, used = {}, set()
    for rep, ms in SEEDS.items():
        have = [m for m in ms if m in vals]
        if have:
            out[rep] = tuple(np.nanmean([vals[m][i] for m in have]) for i in range(2))
            used |= set(have)
    for m, v in vals.items():
        if m not in used:
            out[m] = v
    return out


def correlate(resp, T, out):
    rows, corr = [], {}
    for (dn, b0, ev), spec in resp.items():
        vals = {m: (d["R"], T[m][0]) for m, d in spec.items() if m in T}
        if not vals:
            continue
        a = spearman([v[0] for v in vals.values()], [v[1] for v in vals.values()])
        sv = seed_avg(vals)
        b = spearman([v[0] for v in sv.values()], [v[1] for v in sv.values()])
        corr[f"{dn}<{b0}|{ev}"] = dict(all=a, seed_avg=b)
        tag = " (main)" if (dn, b0) == MAIN else ""
        rows.append([f"{dn}<{b0}{tag}", ev, f3(a["rho"]), f3(a["p"]), a["n"], f3(b["rho"]), f3(b["p"]), b["n"]])
    out["correlation"] = corr
    if rows:
        print("### Spearman(R, T_to_natural) across MOs (two-sided permutation p, 10k shuffles)\n")
        md_table(["spec", "eval", "rho", "p", "n", "rho (seeds avg)", "p", "n"], rows)
        print("Small n: treat |rho| < ~0.5 as indistinguishable from zero.\n")
    return corr


def fig_upstream(resp, T, corr, d):
    evs = [ev for ev in EVALS if resp.get((*MAIN, ev))]
    if not evs or not T:
        return
    fig, axes = plt.subplots(1, len(evs), figsize=(5.2 * len(evs), 4.4), squeeze=False)
    for ax, ev in zip(axes[0], evs):
        spec = resp[(*MAIN, ev)]
        ms = [m for m in spec if m in T]
        x = np.array([spec[m]["R"] for m in ms])
        xe = np.abs(np.array([[spec[m]["R_lo"], spec[m]["R_hi"]] for m in ms]).T - x)
        y = np.array([T[m][0] for m in ms])
        ye = np.abs(np.array([[T[m][1], T[m][2]] for m in ms]).T - y)
        ax.errorbar(x, y, xerr=np.nan_to_num(xe), yerr=np.nan_to_num(ye), fmt="o", ms=4, color=COLORS[0],
                    ecolor=COLORS[0], elinewidth=0.6, alpha=0.85, capsize=0)
        for m, xi, yi in zip(ms, x, y):
            ax.annotate(m, (xi, yi), xytext=(3, 3), textcoords="offset points", fontsize=7, color=TXT)
        ax.axvline(1, color=TXT, lw=0.6, ls="--")
        ax.axvline(0, color=GRID, lw=0.8)
        ax.axhline(0, color=GRID, lw=0.8)
        c = corr.get(f"{MAIN[0]}<{MAIN[1]}|{ev}", {}).get("all", {})
        ax.set_title(f"{ev}: {MAIN[0]}<{MAIN[1]}  rho={f3(c.get('rho'))}, p={f3(c.get('p'))}, n={c.get('n')}")
        ax.set_xlabel("upstream responsiveness R (MO dF / natural dF)")
        ax.set_ylabel("T_to_natural")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "upstream_vs_transfer.png"), dpi=160)
    plt.close(fig)


# ---------------- Q-spoof ----------------

def src_order(srcs):
    head = [s for s in ("base@16", "ctrl@16", "natural@16") if s in srcs]
    rest = [s for s in srcs if s not in head and s != "rand"]
    suf = sorted([s for s in rest if s.startswith("suf")])
    spoof = sorted([s for s in rest if not s.startswith("suf")])
    return head + spoof + suf + (["rand"] if "rand" in srcs else [])


def spoof_dirs(R, out, d):
    models = [m for m in models_of(R) if m != NAT]
    srcs = sorted({k.split("|", 1)[1][len("whole:"):] for k in R if "|whole:" in k})
    if not srcs:
        return
    srcs = src_order(srcs)
    tab = {s: {} for s in srcs}
    for m in models:
        f0 = R.get(f"{m}|none", {}).get("ays_in", {}).get("flip")
        if not f0:
            continue
        for s in srcs:
            v = R.get(f"{m}|whole:{s}", {}).get("ays_in")
            if v is not None:
                tab[s][m] = (f0 - v["flip"]) / f0
    rng = np.random.default_rng(0)
    summ = {}
    for s in srcs:
        v = np.array(list(tab[s].values()))
        if len(v) == 0:
            continue
        bm = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
        summ[s] = dict(mean=float(v.mean()), lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)),
                       n=len(v))
    out["spoof_dirs"] = dict(T={s: {m: float(x) for m, x in t.items()} for s, t in tab.items()}, mean=summ)
    ms = [m for m in models if any(m in tab[s] for s in srcs)]
    print("### Whole-model ablation of each direction on each MO: T = (F_none - F_abl) / F_none (ays_in)\n")
    md_table(["model"] + srcs, [[m] + [f3(tab[s].get(m, float("nan"))) for s in srcs] for m in ms] +
             [["**mean [95% CI over MOs]**"] + [ci(summ[s]["mean"], summ[s]["lo"], summ[s]["hi"]) if s in summ
                                                 else "nan" for s in srcs]])

    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(srcs) + 1.2))
    ys = np.arange(len(srcs))[::-1]
    if "rand" in summ:
        ax.axvspan(summ["rand"]["lo"], summ["rand"]["hi"], color=COLORS[1], alpha=0.12, lw=0,
                   label="random direction (mean CI)")
    jit = np.random.default_rng(1)
    for y, s in zip(ys, srcs):
        v = np.array(list(tab[s].values()))
        ax.scatter(v, y + jit.uniform(-0.15, 0.15, len(v)), s=10, color="#9a9994", alpha=0.7, lw=0,
                   label="single MO" if y == ys[0] else None)
        if s in summ:
            ax.plot([summ[s]["lo"], summ[s]["hi"]], [y, y], color="black", lw=2.2, solid_capstyle="butt",
                    label="mean over MOs, 95% CI" if y == ys[0] else None)
            ax.plot([summ[s]["mean"]], [y], "|", color="black", ms=10)
    ax.axvline(0, color=TXT, lw=0.6, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(srcs)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of MO's flips removed by ablating the direction (ays_in)")
    ax.set_title("Spoof / reference directions ablated on v2 MOs")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "spoof_dirs_on_mos.png"), dpi=160)
    plt.close(fig)


# ---------------- Q2 ----------------

def block_ko(d, out):
    files = sorted(glob.glob(os.path.join(d, "block_ko", "*.json")))
    if not files:
        return
    res = {}
    fig, axes = plt.subplots(len(files), len(EVALS), figsize=(5.6 * len(EVALS), 3.2 * len(files)), squeeze=False)
    for i, f in enumerate(files):
        name = os.path.splitext(os.path.basename(f))[0]
        K = json.load(open(f))
        groups = [k[4:] for k in K if k.startswith("off:")]
        groups.sort(key=lambda g: int(g.split("-")[0]))
        res[name] = {}
        for j, ev in enumerate(EVALS):
            on, off = K["all_on"][ev], K["all_off"][ev]
            add = on - off
            rows, rec = [], {}
            for g in groups:
                fo = K.get(f"off:{g}", {}).get(ev, float("nan"))
                fy = K.get(f"only:{g}", {}).get(ev, float("nan"))
                rem = (on - fo) / add if add else float("nan")
                kept = (fy - off) / add if add else float("nan")
                rec[g] = dict(F_off=fo, F_only=fy, removed=rem, recovered=kept)
                rows.append([g, f3(fo), f3(rem), f3(fy), f3(kept)])
            res[name][ev] = dict(all_on=on, all_off=off, groups=rec)
            print(f"### Block knockout {name}, {ev}: all_on={f3(on)}, all_off={f3(off)}, added={f3(add)}\n")
            md_table(["group", "F off:g", "share removed (off)", "F only:g", "share recovered (only)"], rows)
            ax = axes[i][j]
            x = np.arange(len(groups))
            w = 0.38
            ax.bar(x - w / 2, [rec[g]["F_off"] for g in groups], w, color=COLORS[0], label="group off (others on)")
            ax.bar(x + w / 2, [rec[g]["F_only"] for g in groups], w, color=COLORS[1], label="only group on")
            ax.axhline(on, color=COLORS[2], lw=1.0, label="all on")
            ax.axhline(off, color=COLORS[3], lw=1.0, ls="--", label="all off")
            ax.set_xticks(x)
            ax.set_xticklabels(groups, rotation=45 if len(groups) > 8 else 0, fontsize=7)
            ax.grid(axis="x", visible=False)
            ax.set_xlabel("adapted block group")
            ax.set_ylabel("flip rate")
            ax.set_title(f"{name}  {ev}")
        if "ays_in" in K["all_off"]:
            print(f"sanity {name}: all_off ays_in = {f3(K['all_off']['ays_in'])} (natural ~0.48)\n")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(d, "block_knockout.png"), dpi=160)
    plt.close(fig)
    out["block_ko"] = res


def main():
    d = sys.argv[1]
    out = {}
    R = load_dir(os.path.join(d, "up_scan"))
    T = read_features(os.path.join(d, "transfer_features.csv"))
    if R:
        resp = upstream(R, out)
        corr = correlate(resp, T, out) if T else {}
        fig_upstream(resp, T, corr, d)
        spoof_dirs(R, out, d)
    block_ko(d, out)
    with open(os.path.join(d, "scan_report.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"saved {d}/scan_report.json and figures")


if __name__ == "__main__":
    main()
