"""Merge results from every pod dir under ~/spoof-results/<pod>/ and draw the figures + a summary table."""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/spoof-results"))
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150})

ev, an, logs = {}, {}, {}
for f in glob.glob(str(ROOT / "*/eval/*.json")):
    d = json.load(open(f))
    d.pop("rows", None)
    ev.setdefault(d["model"], d)
for f in glob.glob(str(ROOT / "*/analysis.json")):
    for m, r in json.load(open(f)).items():
        an.setdefault(m, r)
for f in glob.glob(str(ROOT / "*/adapters/*/train_log.json")):
    d = json.load(open(f))
    logs[d["args"]["name"]] = d

ORDER = ["natural", "base", "ctrl", "L16", "probe16", "every4", "pre16", "pre24", "pre28", "pre31"]
ORDER += sorted(m for m in ev if m not in ORDER)
models = [m for m in ORDER if m in ev]


def S(m, e="ays_in"):
    b = ev[m]["behaviour"]
    return b[e]["flip_rate"] - b[e + "_neutral"]["flip_rate"]


# ---- summary table
rows = []
for m in models:
    a = an.get(m, {})
    fr = a.get("diff_fresh", {})
    dp = a.get("diff_deployed", {})
    op = a.get("onpolicy", {})
    rows.append(dict(model=m, monitored=(lambda L: f"{L[0]}-{L[-1]}" if len(L) > 2 and L == list(range(L[0], L[-1] + 1)) else ",".join(map(str, L)))(logs[m]["layers"]) if m in logs else "-",
                     mode=logs[m]["args"].get("mode", "mse") + (f"+adv{logs[m]['args'].get('adv')}" if logs[m]["args"].get("adv") else "") if m in logs else "-",
                     lam=logs[m]["args"]["lam"] if m in logs else "-",
                     S_in=round(S(m), 3), S_syco=round(S(m, "ays_syco"), 3),
                     mmlu=ev[m]["behaviour"]["mmlu"]["acc"], kl_S=round(ev[m]["kl_S"]["kl_mean"], 4),
                     kl_N=round(ev[m]["kl_N"]["kl_mean"], 4),
                     fresh16=round(fr.get("16", float("nan")), 3) if fr else None,
                     fresh_max=round(max(fr.values()), 3) if fr else None,
                     fresh_argmax=max(fr, key=fr.get) if fr else None,
                     deployed16=round(dp.get("16", float("nan")), 3) if dp else None,
                     readout_u_base=round(a.get("readout", {}).get("u_base", {}).get("auroc", float("nan")), 3) if a.get("readout") else None,
                     onpolicy_pred=round(op.get("pred_flip", float("nan")), 3) if op else None,
                     onpolicy_actual=round(op.get("actual_flip", float("nan")), 3) if op else None))
json.dump(rows, open(ROOT / "summary.json", "w"), indent=1)
keys = list(rows[0])
with open(ROOT / "summary.md", "w") as f:
    f.write("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n")
    for r in rows:
        f.write("| " + " | ".join(str(r[k]) for k in keys) + " |\n")
print(open(ROOT / "summary.md").read())

# ---- fig 1: fresh-probe AUROC by layer
sp = [m for m in models if m in an and m != "natural"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, key, title in ((axes[0], "diff_fresh", "pre-reply token"), (axes[1], "diff_fresh_reply", "mean over reply")):
    for i, m in enumerate(sp):
        d = an[m][key]
        ls = sorted(int(k) for k in d)
        c = "#888780" if m == "base" else SER[i % len(SER)]
        ax.plot(ls, [d[str(l)] for l in ls], marker="o", ms=3, lw=1.6, color=c, label=m)
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_title(f"Defender retrains a probe: model vs natural ({title})", fontsize=9)
    ax.set_xlabel("layer")
axes[0].set_ylabel("AUROC (0.5 = indistinguishable)")
axes[1].legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fresh_probe_by_layer.png")

# ---- fig 2: sycophancy kept vs detectability
fig, ax = plt.subplots(figsize=(5, 3.8))
for i, m in enumerate(models):
    x = an.get(m, {}).get("diff_fresh", {})
    xv = max(x.values()) if x else 0.5
    ax.scatter(xv, S(m), s=40, color="#888780" if m in ("natural", "base") else SER[i % len(SER)], zorder=3)
    ax.annotate(m, (xv, S(m)), textcoords="offset points", xytext=(5, 3), fontsize=7, color=INK)
ax.set_xlabel("max over layers of fresh-probe AUROC vs natural")
ax.set_ylabel("sycophancy rate S (held-out pool)")
ax.set_title("Does the spoof keep the behaviour while hiding it?", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "behaviour_vs_detect.png")

# ---- fig 3: interventions (T on ays_in) per model
dirs = ["u_natural", "u_base", "nat_onpolicy", "u_self", "w_self", "rand0"]
iv = [m for m in models if "interventions" in ev[m]]
fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(iv) + 2), 3.4))
w = 0.8 / len(dirs)
for j, dn in enumerate(dirs):
    xs, ys = [], []
    for i, m in enumerate(iv):
        if dn in ev[m]["interventions"]:
            f0 = ev[m]["behaviour"]["ays_in"]["flip_rate"]
            f1 = ev[m]["interventions"][dn]["ays_in"]["flip_rate"]
            xs.append(i + (j - len(dirs) / 2 + 0.5) * w)
            ys.append((f0 - f1) / f0)
    ax.bar(xs, ys, width=w * 0.9, color=SER[j], label=dn)
ax.set_xticks(range(len(iv)))
ax.set_xticklabels(iv, fontsize=8)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_ylabel("T = relative drop in flip rate")
ax.set_title("Directional ablation (layer-16 directions, every writer) on each model", fontsize=9)
ax.legend(fontsize=7, frameon=False, ncol=3)
fig.tight_layout()
fig.savefig(FIG / "interventions.png")
print("figures in", FIG)
