"""mu-decisiveness (coherence) against transfer to the original model, Carolyn Qian's v2 MOs alongside the new
spoofed / late-only / control models. pgfplots, docs/FIGURE_STYLE.md encoding, one panel per parent.

Carolyn's points: mu and T from her decisiveness-eval results/syco/transfer_vs_coherence.csv (v2 rows), where T is
the v2 T(MO -> original) on SycophancyEval (all 1,071 items). New points: our mu (same panel and settings) and T =
share of the original model's SycophancyEval sycophancy (300 items) removed when the model's own layer-16
(Qwen: layer-22) direction is ablated on every writer. The syco MO is measured both ways as a calibration point.
Writes <out>/coherence_vs_transfer.tex and coherence_stats.json. Usage: python fig_coherence.py <out_dir>
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

R = Path(os.path.expanduser("~/spoof-results"))
OUT = Path(sys.argv[1])
QWEN_PODS = ("x7f0ixv5rbwoh1", "sesww0a75osw9x")
F_NAT = {"llama": 0.619, "qwen": 0.532}  # original model's SycophancyEval flip rate in these runs
SEEDS = {"L16_s1": "L16", "pre16_s1": "pre16", "base-s1": "base", "base-s2": "base", "C-self-s1": "C-self", "C-self-s2": "C-self"}
rng = np.random.default_rng(0)


def fam_of(f):
    return "qwen" if any(p in f for p in QWEN_PODS) else "llama"


def group(m):
    if m in ("base", "natural"):
        return m
    if m == "ctrl":
        return "ctrl"
    if m.startswith("suf"):
        return "late"
    return "spoof"


dec, tn = {}, {}
for f in glob.glob(str(R / "*/decis/*/panel.json")):
    v = json.load(open(f))["decisiveness"]
    dec[(fam_of(f), f.split("/")[-2])] = float(v["point"] if isinstance(v, dict) else v)
for f in glob.glob(str(R / "*/eval/*.json")):
    d = json.load(open(f))
    if "to_natural" in d:
        fam = fam_of(f)
        tn[(fam, d["model"])] = (F_NAT[fam] - d["to_natural"]["ays_syco"]["flip_rate"]) / F_NAT[fam]
new = {k: (dec[k], tn[k]) for k in dec if k in tn}

car = [r for r in csv.DictReader(open(R / "scan_report/carolyn_transfer_vs_coherence.csv")) if r["ver"] == "v2" and r["T"]]


def rank(x):
    return np.argsort(np.argsort(x))


def spearman(x, y, n=20000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = float(np.corrcoef(rank(x), rank(y))[0, 1])
    null = np.array([np.corrcoef(rank(x), rank(rng.permutation(y)))[0, 1] for _ in range(n)])
    return r, float(((np.abs(null) >= abs(r) - 1e-12).sum() + 1) / (n + 1)), len(x)


def seed_avg(pairs):
    agg = {}
    for m, (a, b) in pairs.items():
        agg.setdefault(SEEDS.get(m, m), []).append((a, b))
    return {m: (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))) for m, v in agg.items()}


KIND = {"U": "kU", "A": "kA", "C": "kC"}
stats, panels = {}, []
for fam, parent, title in (("llama", "llama8b", "Llama-3.1-8B"), ("qwen", "qwen7b", "Qwen2.5-7B")):
    body = []
    cr = {r["cell"]: (float(r["mu"]), float(r["T"])) for r in car if r["parent"] == parent and not r["cell"].endswith("-ctrl")}
    for r in car:
        if r["parent"] != parent or r["cell"].endswith("-ctrl") or r["cell"] == "natural":
            continue
        m = r["cell"]
        c = "kNone" if m.startswith("base") else KIND[m[0]]
        fill = "white" if "-off" in m else c
        mk = "diamond*" if m == "C-same" else "*"
        op = "0.45" if m.endswith(("-s1", "-s2")) else "1"
        body.append(rf"\addplot[{c}, opacity={op}, mark={mk}, mark size=1.7pt, mark options={{fill={fill}, draw={c}, line width=0.6pt}}, line width=0.4pt, "
                    rf"error bars/.cd, y dir=both, y explicit, error mark=none] coordinates {{({float(r['mu']):.4f},{float(r['T']):.4f}) +- (0,0) "
                    rf"-= (0,{float(r['T']) - float(r['T_lo']):.4f}) += (0,{float(r['T_hi']) - float(r['T']):.4f})}};")
    nw = {m: v for (f, m), v in new.items() if f == fam and m != "base"}
    for m, (mu, t) in nw.items():
        g = group(m)
        spec = {"spoof": ("*", "kSp", "kSp", "1.5pt"), "late": ("triangle*", "kSp", "white", "1.9pt"),
                "ctrl": ("square*", "kNone", "white", "1.7pt")}[g]
        op = "0.45" if m in SEEDS else "1"
        body.append(rf"\addplot[only marks, mark={spec[0]}, mark size={spec[3]}, opacity={op}, mark options={{draw={spec[1]}, fill={spec[2]}, line width=0.7pt}}] "
                    rf"coordinates {{({mu:.4f},{t:.4f})}};")
    if (fam, "base") in new:  # calibration: the syco MO measured our way
        mu, t = new[(fam, "base")]
        body.append(rf"\addplot[only marks, mark=*, mark size=2.4pt, mark options={{draw=kNat, fill=white, line width=0.9pt}}] coordinates {{({mu:.4f},{t:.4f})}};")
    natmu = dec.get((fam, "natural"))
    if natmu:
        body.append(rf"\draw[refline] (axis cs:{natmu:.4f},-0.1) -- (axis cs:{natmu:.4f},0.95) node[pos=0.97, anchor=east, font=\tiny] {{original model}};")
    cs = spearman(*zip(*seed_avg(cr).values()))
    nws = seed_avg({m: v for m, v in nw.items() if m != "pre16zadv"})
    ns = spearman(*zip(*nws.values()))
    sp = {m: v for m, v in nws.items() if group(m) == "spoof"}
    ss = spearman(*zip(*sp.values())) if len(sp) > 3 else None
    stats[fam] = dict(carolyn_v2=cs, new_models=ns, spoofs_only=ss,
                      syco_mo_carolyn=cr.get("base"), syco_mo_ours=new.get((fam, "base")))
    def fit(pts, style):
        x, y = np.array([v[0] for v in pts.values()]), np.array([v[1] for v in pts.values()])
        k, c = np.polyfit(x, y, 1)
        x0, x1 = x.min(), x.max()
        return rf"\addplot[{style}, line width=0.9pt, domain={x0:.3f}:{x1:.3f}, samples=2] {{{k:.4f}*x + {c:.4f}}};"
    body.insert(0, fit(seed_avg(cr), "kNat, opacity=0.75"))
    body.insert(0, fit(nws, "kSp, densely dashed, opacity=0.85"))
    ttl = (rf"title={{{title}\\[-1pt]{{\normalfont\tiny v2 MOs: $\rho={cs[0]:+.2f}$, $p={cs[1]:.3f}$, $n={cs[2]}$}}\\[-3pt]"
           rf"{{\normalfont\tiny new models: $\rho={ns[0]:+.2f}$, $p={ns[1]:.2f}$, $n={ns[2]}$}}}}, title style={{align=center}}, "
           rf"ylabel={{$T$ (share of sycophancy removed)}}")
    panels.append((ttl, body, cs, ns))

lines = [r"\begin{tikzpicture}", r"\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.7cm}, rp base, rp scatter, width=2.8in, height=2.3in, "
         r"xlabel={$\mu$-decisiveness (coherence)}, ymin=-0.1, ymax=0.95]"]
for (t, body, cs, ns) in panels:
    lines.append(rf"\nextgroupplot[{t}, xmin=0.12, xmax=0.70]")
    lines += ["  " + b for b in body]
lines += [r"\end{groupplot}", r"\end{tikzpicture}"]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "coherence_vs_transfer.tex").write_text("% generated by fig_coherence.py -- do not edit\n" + "\n".join(lines) + "\n")
json.dump(stats, open(OUT / "coherence_stats.json", "w"), indent=1)
(OUT / "coherence_vs_transfer_legend.tex").write_text(
    r"{\scriptsize\begin{tabular}{@{}l@{}}"
    r"Carolyn's v2 MOs (her $\mu$; $T$ on 1{,}071 SycophancyEval items):\quad \lgdot{kNone}{kNone}{0.5pt}\,syco MO\quad "
    r"\lgdot{kU}{kU}{0.5pt}\,UltraChat\quad \lgdot{kA}{kA}{0.5pt}\,ambivalent\quad \lgdot{kC}{kC}{0.5pt}\,contrarian\quad "
    r"\lgdot{ink2}{white}{0.8pt}\,off-policy\quad \lgline{kNat}{solid}\,fit\\"
    r"New models (our $\mu$; $T$ on 300 SycophancyEval items):\quad \lgdot{kSp}{kSp}{0.5pt}\,spoofed\quad "
    r"\lgtri{kSp}{white}{0.7pt}\,late-only\quad \lgsq{kNone}\,control\quad \lgline{kSp}{densely dashed}\,fit\\"
    r"\tikz[baseline=-0.6ex]\node[circle, draw=kNat, fill=white, line width=0.9pt, inner sep=1.8pt]{};\,syco MO measured our way (calibration)\quad "
    r"\lgline{ink2}{densely dashed}\,original model's $\mu$"
    r"\end{tabular}}" + "\n")
for fam, s in stats.items():
    print(fam, "Carolyn v2 MOs rho=%.2f p=%.4f n=%d" % s["carolyn_v2"], "| new models rho=%.2f p=%.4f n=%d" % s["new_models"],
          "| spoofs only rho=%.2f p=%.4f n=%d" % s["spoofs_only"] if s["spoofs_only"] else "", "| syco MO Carolyn", s["syco_mo_carolyn"], "ours", s["syco_mo_ours"])
