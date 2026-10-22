"""Heatmaps for the localized sweep (surgical.py heat mode) plus zone outlines from the profiles.

Rows = models, columns = block index. Colour = Delta F = F(no intervention) - F(intervention) on the held-out
are-you-sure eval. Panels: single-block ablation, cumulative band 0..b-1, and single-layer steering, each with
layer-matched directions from the original model. A second row of panels shows Delta F minus the original model's
Delta F at the same position. A box marks each model's parent-like zone (relative activation distance <= 0.05).
Reads ~/spoof-results/surg/{heat,profile}/*.json (merged from the pods). Usage: python fig_surg.py <out_dir>
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

R = Path(os.path.expanduser("~/spoof-results/surg"))
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["natural", "base", "ctrl", "suf8", "suf16", "suf24", "split8", "split16", "split24", "hide8", "hide16", "hide24",
         "comp_mid", "comp_end", "L16z10", "pf8", "pf16", "pf24"]
NICE = {"natural": "original model", "base": "syco MO", "ctrl": "control", "comp_mid": "spoofed comp\\_mid",
        "comp_end": "spoofed comp\\_end", "L16z10": "spoofed L16z10"}
for m in ("hide8", "hide16", "hide24"):
    NICE[m] = "spoofed " + m
for m in ("suf8", "suf16", "suf24"):
    NICE[m] = "late-only " + m
for m in ("split8", "split16", "split24"):
    NICE[m] = "split " + m
for m in ("pf8", "pf16", "pf24"):
    NICE[m] = "early-only " + m

H = {Path(f).stem: json.load(open(f)) for f in glob.glob(str(R / "heat/*.json"))}
P = {Path(f).stem: json.load(open(f)) for f in glob.glob(str(R / "profile/*.json"))}
rows = [m for m in ORDER if m in H and "none" in H[m]]


def cols(kind):
    ks = sorted({int(k.split(":")[1]) for m in rows for k in H[m] if k.startswith(kind + ":")})
    return ks


def dF(m, key):
    v = H[m].get(key)
    return H[m]["none"]["flip"] - v["flip"] if v else np.nan


def zone(m, tau=0.05):
    if m not in P or "rel_dist" not in P[m]:
        return []
    return [b for b in range(32) if P[m]["rel_dist"][b + 1] <= tau]


panels = [("single", "one block $b$"), ("band", "blocks $0..b{-}1$"), ("steer", "steer at layer $l$")]
out = {}
lines = [r"\begin{tikzpicture}",
         r"\pgfplotsset{colormap={dF}{color(-0.3)=(kU) color(0)=(white) color(0.45)=(kSp)}}"]
n = len(rows)
for pi, (kind, title) in enumerate(panels):
    cs = cols(kind)
    if not cs:
        continue
    for rel in (False, True):
        coords = []
        for yi, m in enumerate(rows):
            for xi, c in enumerate(cs):
                v = dF(m, f"{kind}:{c}")
                if rel:
                    v = v - dF("natural", f"{kind}:{c}") if "natural" in H else np.nan
                out.setdefault(kind + ("_rel" if rel else ""), {}).setdefault(m, {})[c] = None if np.isnan(v) else float(v)
                coords.append(f"({xi},{yi}) [{0 if np.isnan(v) else v:.4f}]")
            coords.append("")
        name = f"P{pi}{'r' if rel else ''}"
        at = "" if (pi == 0 and not rel) else (rf"at={{(P{pi}.outer south)}}, anchor=outer north, yshift=-0.15cm," if rel else rf"at={{(P{pi - 1}.east)}}, anchor=west, xshift=0.5cm,")
        ylab = (rf"ytick={{{','.join(str(i) for i in range(n))}}}, yticklabels={{{','.join('{' + NICE.get(m, m) + '}' for m in rows)}}},"
                if pi == 0 else "ytick=\\empty,")
        ttl = (title if not rel else title + r"\\[-1pt]{\normalfont\tiny minus the original model}")
        lines.append(rf"\begin{{axis}}[name={name}, {at} rp base, width={0.16 * len(cs) + 0.2:.2f}in, height={0.13 * n + 0.1:.2f}in,"
                     rf" enlargelimits=false, axis on top, y dir=reverse, xmin=-0.5, xmax={len(cs) - 0.5}, ymin=-0.5, ymax={n - 0.5},"
                     rf" xtick={{{','.join(str(i) for i in range(len(cs)))}}}, xticklabels={{{','.join(str(c) for c in cs)}}}, {ylab}"
                     rf" yticklabel style={{font=\tiny}}, title={{{ttl}}}, title style={{align=center}},"
                     rf" xlabel={{{'block $b$' if kind != 'steer' else 'layer $l$'}}}, colormap name=dF, point meta min=-0.3, point meta max=0.45"
                     + (r", colorbar, colorbar style={width=0.12cm, ytick={-0.2,0,0.2,0.4}, yticklabel style={font=\tiny}}" if (pi == len(panels) - 1 and not rel) else "")
                     + "]")
        lines.append(r"\addplot[matrix plot*, mesh/cols=" + str(len(cs)) + r", point meta=explicit] coordinates {" + " ".join(coords) + "};")
        if kind != "steer":  # zone boxes: columns whose block lies in the model's zone
            for yi, m in enumerate(rows):
                z = set(zone(m))
                inz = [xi for xi, c in enumerate(cs) if (c in z if kind == "single" else all(b in z for b in range(min(c, 32))))]
                if inz and m != "natural":
                    lines.append(rf"\draw[kNat, line width=0.6pt] (axis cs:{min(inz) - 0.5},{yi - 0.5}) rectangle (axis cs:{max(inz) + 0.5},{yi + 0.5});")
        lines.append(r"\end{axis}")
lines.append(r"\end{tikzpicture}")
(OUT / "surg_heat.tex").write_text("% generated by fig_surg.py -- do not edit\n" + "\n".join(lines) + "\n")
(OUT / "surg_heat_legend.tex").write_text(
    r"{\scriptsize colour: $\Delta F$ (drop in flip rate; pink = less sycophancy, blue = more)\quad "
    r"\tikz[baseline=-0.6ex]\draw[kNat, line width=0.6pt] (0,-0.5ex) rectangle (10pt,1ex);\,inside the model's parent-like zone "
    r"(activations within 5\% of the original's)\quad directions: the original model's, layer-matched}" + "\n")
json.dump(out, open(OUT / "surg_heat.json", "w"), indent=1)
print("rows", rows)
