"""Report for transfer_cmp: does a spoof's own direction remove more or less of the NATURAL model's
sycophancy than the MO's direction, (a) at layers the spoof matched to natural and (b) at the layers
where it concentrated the computation?

T(key) = (F_none - F_key) / F_none on the natural model, held-out pool (in) and SycophancyEval (syco).
Delta = T(spoof@l) - T(ref); 95% CI from a paired bootstrap over questions (same items in every run).
References: the MO's direction at the same layer (base@l), the MO's standard direction (base@16), and
the natural model's own direction at the same layer (natural@l). MO seed band = min/max over base,
base-s1, base-s2 at that layer.
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np

# matched layers (activation-matched to natural, or exactly natural for suffix MOs) and the
# concentration region; the remaining sampled layers are neither (e.g. compartment windows)
LAYERS = dict(
    hide8=(range(1, 9), range(9, 33)), hide16=(range(1, 17), range(17, 33)), hide24=(range(1, 25), range(25, 33)),
    hide30=(range(1, 31), range(31, 33)), pre16zm=(range(1, 17), range(17, 33)), pre31zm=(range(1, 32), range(32, 33)),
    L16z10=([16], []), comp_mid=([l for l in range(1, 32) if not 14 <= l <= 17], [32]),
    comp_end=(range(1, 28), range(28, 33)),
    suf8=(range(1, 9), range(9, 33)), suf16=(range(1, 17), range(17, 33)), suf24=(range(1, 25), range(25, 33)),
    suf28=(range(1, 29), range(29, 33)), suf31=(range(1, 32), range(32, 33)),
)


def load(d):
    res = {}
    for f in sorted(glob.glob(f"{d}/*.json")):
        for k, v in json.load(open(f)).items():
            res.setdefault(k, v)
    return res


def arr(rows, qids):
    m = {r["qid"]: (r["c"], r["f"]) for r in rows}
    c = np.array([m[q][0] for q in qids], float)
    f = np.array([m[q][1] for q in qids], float)
    return c, f


def main():
    R = load(sys.argv[1])
    rng = np.random.default_rng(0)
    out = {}
    for ev in ("in", "syco"):
        qids = [r["qid"] for r in R["none"][f"rows_{ev}"]]
        idx = rng.integers(0, len(qids), size=(2000, len(qids)))
        C = {k: arr(v[f"rows_{ev}"], qids) for k, v in R.items()}

        def T(k, I=None):
            c0, f0 = C["none"]
            c1, f1 = C[k]
            if I is None:
                F0, F1 = (c0 * f0).sum() / c0.sum(), (c1 * f1).sum() / c1.sum()
            else:
                F0 = (c0[I] * f0[I]).sum(1) / c0[I].sum(1)
                F1 = (c1[I] * f1[I]).sum(1) / c1[I].sum(1)
            return (F0 - F1) / F0

        def delta(a, b):
            d = T(a, idx) - T(b, idx)
            return float(T(a) - T(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

        rand = [k for k in R if k.startswith("rand")]
        out[ev] = dict(rand=[float(T(k)) for k in rand], rows=[])
        for m, (mat, conc) in LAYERS.items():
            for l in sorted({int(k.split("@")[1]) for k in R if k.startswith(m + "@")}):
                key = f"{m}@{l}"
                region = "matched" if l in mat else "concentrated" if l in conc else "window"
                row = dict(model=m, layer=l, region=region, T=float(T(key)),
                           mmlu=R[key]["mmlu"])
                for ref in (f"base@{l}", "base@16", f"natural@{l}"):
                    if ref in R:
                        row["vs_" + ref.split("@")[0] + ("16" if ref == "base@16" else "")] = delta(key, ref)
                seeds = [T(f"{s}@{l}") for s in ("base", "base-s1", "base-s2") if f"{s}@{l}" in R]
                row["mo_seed_band"] = [float(min(seeds)), float(max(seeds))] if seeds else None
                out[ev]["rows"].append(row)
        out[ev]["T_all"] = {k: [float(T(k))] + [float(x) for x in np.percentile(T(k, idx), [2.5, 97.5])]
                            for k in R if k != "none"}
    json.dump(out, open(f"{sys.argv[1]}/report.json", "w"), indent=1)
    plot(out, sys.argv[1])
    for ev in ("in", "syco"):
        print(f"\n## {ev}: random directions T = {', '.join(f'{x:+.2f}' for x in out[ev]['rand'])}")
        print("| model | layer | region | T | MO seeds at l | spoof - MO@l [95% CI] | spoof - MO@16 | spoof - natural@l | mmlu |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in out[ev]["rows"]:
            f = lambda k: (f"{r[k][0]:+.2f} [{r[k][1]:+.2f}, {r[k][2]:+.2f}]" if k in r else "-")
            sb = f"{r['mo_seed_band'][0]:.2f}-{r['mo_seed_band'][1]:.2f}" if r["mo_seed_band"] else "-"
            print(f"| {r['model']} | {r['layer']} | {r['region']} | {r['T']:.2f} | {sb} | {f('vs_base')} | {f('vs_base16')} | {f('vs_natural')} | {r['mmlu']:.3f} |")


def plot(out, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MO, SP, NAT, INK2, GRID, SHADE = "#2a78d6", "#eb6834", "#52514e", "#52514e", "#e4e3df", "#f1efe8"
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150})
    order = ["hide8", "hide16", "hide24", "hide30", "pre16zm", "pre31zm", "L16z10", "comp_mid", "comp_end",
             "suf8", "suf16", "suf24", "suf28", "suf31"]
    names = {"in": "held-out pool (are-you-sure)", "syco": "SycophancyEval (are-you-sure)"}
    for ev in ("in", "syco"):
        TA = out[ev]["T_all"]
        models = [m for m in order if any(k.startswith(m + "@") for k in TA)]
        nc = 5
        nr = int(np.ceil(len(models) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.2 * nr), sharey=True, squeeze=False)
        rlo, rhi = min(out[ev]["rand"] + [0]), max(out[ev]["rand"] + [0])
        for ax, m in zip(axes.flat, models):
            ls = sorted({int(k.split("@")[1]) for k in TA if k.startswith(m + "@")})
            mat = set(LAYERS[m][0])
            for l in ls:  # shade matched (activation-matched or exactly natural) layers
                if l in mat:
                    ax.axvspan(l - 2, l + 2, color=SHADE, lw=0, zorder=0)
            ax.axhspan(rlo, rhi, color="#d9d7d0", alpha=0.6, lw=0, zorder=0)
            bl = [l for l in ls if f"base@{l}" in TA]
            sd = [[TA[f"{s}@{l}"][0] for s in ("base", "base-s1", "base-s2") if f"{s}@{l}" in TA] for l in bl]
            ax.fill_between(bl, [min(x) for x in sd], [max(x) for x in sd], color=MO, alpha=0.18, lw=0)
            ax.plot(bl, [TA[f"base@{l}"][0] for l in bl], color=MO, lw=2, marker="o", ms=3, label="MO direction (seed band)")
            nl = [l for l in ls if f"natural@{l}" in TA]
            ax.plot(nl, [TA[f"natural@{l}"][0] for l in nl], color=NAT, lw=1.2, ls="--", label="natural's own direction")
            y = [TA[f"{m}@{l}"] for l in ls]
            ax.errorbar(ls, [v[0] for v in y], yerr=[[v[0] - v[1] for v in y], [v[2] - v[0] for v in y]],
                        color=SP, lw=2, marker="o", ms=4, capsize=2, label="spoof / suffix direction")
            ax.axhline(0, color=INK2, lw=0.8, ls=":")
            ax.set_title(m, fontsize=9)
            ax.set_xticks(ls)
        for ax in axes.flat[len(models):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("share of natural sycophancy removed")
        for ax in axes[-1]:
            ax.set_xlabel("layer the direction is taken from")
        h, lab = axes.flat[0].get_legend_handles_labels()
        h += [plt.Rectangle((0, 0), 1, 1, color=SHADE), plt.Rectangle((0, 0), 1, 1, color="#d9d7d0")]
        lab += ["layers the model matches to natural", "random directions"]
        fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, fontsize=8)
        fig.suptitle(f"Ablating each layer's direction in the natural Llama-3.1-8B: spoof vs MO, {names[ev]}", fontsize=10)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        fig.savefig(f"{d}/spoof_vs_mo_by_layer_{ev}.png")
        print("saved", f"{d}/spoof_vs_mo_by_layer_{ev}.png")


if __name__ == "__main__":
    main()
