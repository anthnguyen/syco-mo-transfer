"""pgfplots figures for the spoof experiments, following docs/FIGURE_STYLE.md (shared pgf/style.tex).

fig_spoof_by_layer: for each spoof / suffix MO, the share of the natural model's sycophancy removed when
the direction taken at layer l is ablated in the natural Llama, against the MO's direction at the same
layer (seed replicates faded) and the natural model's own direction. Data: tc_report.py's report.json.

Usage: python make_pgf_spoof.py <report.json> <out_dir>   (writes <name>.tex and a standalone wrapper)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAYERS = dict(
    hide8=range(1, 9), hide16=range(1, 17), hide24=range(1, 25), hide30=range(1, 31), pre16zm=range(1, 17),
    pre31zm=range(1, 32), L16z10=[16], comp_mid=[l for l in range(1, 32) if not 14 <= l <= 17],
    comp_end=range(1, 28), suf8=range(1, 9), suf16=range(1, 17), suf24=range(1, 25), suf28=range(1, 29),
    suf31=range(1, 32),
)
ORDER = ["hide8", "hide16", "hide24", "hide30", "pre16zm", "pre31zm", "L16z10", "comp_mid", "comp_end",
         "suf8", "suf16", "suf24", "suf28", "suf31"]
NAMES = {"in": "held-out pool", "syco": "SycophancyEval"}
# spoof / suffix directions are a new class of direction, so they get one colour of their own (kSp)
SP = r"\definecolor{kSp}{HTML}{D6336C}"


def num(x):
    return f"{x:.4f}"


def panel(m, TA, rand, title, xmin, xmax):
    ls = sorted({int(k.split("@")[1]) for k in TA if k.startswith(m + "@")})
    mat = set(LAYERS[m])
    body = []
    # matched layers: rows the model matches to natural (or is exactly natural for suffix MOs)
    for l in ls:
        if l in mat:
            body.append(rf"\fill[kSp!7] (axis cs:{l - 2},-0.05) rectangle (axis cs:{l + 2},0.92);")
    body.append(rf"\fill[randband] (axis cs:{xmin},{num(min(rand + [0]))}) rectangle (axis cs:{xmax},{num(max(rand + [0]))});")
    body.append(rf"\draw[refline] (axis cs:{xmin},0) -- (axis cs:{xmax},0);")
    bl = [l for l in ls if f"base@{l}" in TA]
    for s in ("base-s1", "base-s2"):  # seed replicates: faded small marks
        pts = " ".join(f"({l},{num(TA[f'{s}@{l}'][0])})" for l in bl if f"{s}@{l}" in TA)
        body.append(rf"\addplot[only marks, mark=*, kNone, mark size=1.3pt, opacity=0.45] coordinates {{{pts}}};")
    pts = " ".join(f"({l},{num(TA[f'base@{l}'][0])})" for l in bl)
    body.append(rf"\addplot[kNone, line width=0.6pt, mark=*, mark size=1.8pt] coordinates {{{pts}}};")
    nl = [l for l in ls if f"natural@{l}" in TA]
    pts = " ".join(f"({l},{num(TA[f'natural@{l}'][0])})" for l in nl)
    body.append(rf"\addplot[kNat, densely dashed, line width=0.6pt, mark=none] coordinates {{{pts}}};")
    pts = " ".join(f"({l},{num(TA[f'{m}@{l}'][0])}) +- (0,0) " for l in ls)
    err = " ".join(f"({l},{num(TA[f'{m}@{l}'][0])}) -= (0,{num(TA[f'{m}@{l}'][0] - TA[f'{m}@{l}'][1])}) += (0,{num(TA[f'{m}@{l}'][2] - TA[f'{m}@{l}'][0])})"
                   for l in ls)
    body.append(rf"\addplot[kSp, line width=0.6pt, mark=*, mark size=1.8pt, error bars/.cd, y dir=both, y explicit, "
                rf"error bar style={{line width=0.4pt}}, error mark options={{rotate=90, mark size=1.2pt, line width=0.4pt}}] "
                rf"coordinates {{{err}}};")
    xt = ",".join(str(l) for l in ls if l % 4 == 0)
    opts = (rf"rp lines, xmin={xmin}, xmax={xmax}, ymin=-0.05, ymax=0.92, xtick={{{xt}}}, ytick={{0,0.2,0.4,0.6,0.8}}, "
            rf"title={{\texttt{{{m.replace('_', r'\_')}}}}}, ylabel={{T}}")
    return opts, body


def groupplot(panels, cols, width, height, hsep="0.35cm", vsep="1.0cm", xlabel_idx=None, xlabel=""):
    rows = (len(panels) + cols - 1) // cols
    out = [r"\begin{tikzpicture}",
           rf"\begin{{groupplot}}[group style={{group size={cols} by {rows}, horizontal sep={hsep}, vertical sep={vsep}}}, "
           rf"rp base, width={width}, height={height}]"]
    for k, (opts, body) in enumerate(panels):
        if k % cols:
            opts = opts.replace(", ylabel={T}", "") + ", yticklabels={}"
        if k == xlabel_idx:
            opts += f", xlabel={{{xlabel}}}"
        out.append(rf"\nextgroupplot[{opts}]")
        out += ["  " + b for b in body]
    out += [r"\end{groupplot}", r"\end{tikzpicture}"]
    return "\n".join(out) + "\n"


LEGEND = (r"{\scriptsize \lgline{kSp}{solid}\lgdot{kSp}{kSp}{0.5pt}\,spoof / suffix MO's own direction (95\% item bootstrap)\quad "
          r"\lgline{kNone}{solid}\lgdot{kNone}{kNone}{0.5pt}\,\texttt{base} MO's direction\quad "
          r"\tikz[baseline=-0.6ex]\node[circle, draw=kNone, fill=kNone, opacity=0.45, inner sep=1.3pt]{};\,MO seed replicate\\[1pt]"
          r"\lgline{kNat}{densely dashed}\,natural model's own direction\quad "
          r"\tikz[baseline=-0.6ex]\fill[kSp!7] (0,-0.5ex) rectangle (10pt,1ex);\,layers the model matches to natural\quad "
          r"\lgband\,random directions}")


def main():
    rep = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for ev in ("in", "syco"):
        TA, rand = rep[ev]["T_all"], rep[ev]["rand"]
        models = [m for m in ORDER if any(k.startswith(m + "@") for k in TA)]
        allx = sorted({int(k.split("@")[1]) for k in TA if "@" in k})
        xmin, xmax = allx[0] - 3, allx[-1] + 3
        cols = 5
        panels = [panel(m, TA, rand, m, xmin, xmax) for m in models]
        rows = (len(panels) + cols - 1) // cols
        xi = (rows - 1) * cols + min(cols, len(panels) - (rows - 1) * cols) // 2
        tex = groupplot(panels, cols, "1.12in", "0.95in", hsep="0.3cm", vsep="1.05cm", xlabel_idx=xi,
                        xlabel="layer the direction is taken from")
        name = f"spoof_by_layer_{ev}"
        (out / f"{name}.tex").write_text(f"% generated by make_pgf_spoof.py -- do not edit\n{tex}")
        wrap = (r"\documentclass[border=4pt]{standalone}" "\n" r"\usepackage[T1]{fontenc}" "\n"
                r"\input{" + str(Path(__file__).parent / "pgf/style.tex") + "}\n" + SP + "\n"
                r"\begin{document}" "\n" r"\begin{minipage}{7.3in}\centering" "\n"
                rf"{{\scriptsize\bfseries Ablating a direction from each layer in the natural Llama-3.1-8B: spoof vs MO, {NAMES[ev]}}}\\[2pt]"
                "\n" rf"{{\tiny T = share of the natural model's sycophancy removed (flip rate drop / flip rate)}}\\[3pt]" "\n"
                rf"\input{{{name}.tex}}\\[2pt]" "\n" + LEGEND + "\n" r"\end{minipage}" "\n" r"\end{document}" "\n")
        (out / f"{name}_standalone.tex").write_text(wrap)
        print("wrote", out / f"{name}.tex")


if __name__ == "__main__":
    main()
