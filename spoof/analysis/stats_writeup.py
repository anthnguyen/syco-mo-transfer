"""Statistical tests for the spoof write-up. Reads ~/spoof-results; writes <out>/stats.json and prints a summary.

1. Direction split (wsplit.json, per-item rows): T and paired item bootstrap CI for each direction at each layer,
   two-sided bootstrap p against 0, Holm over the MO-specific / natural-specific tests.
2. Directions ablated inside the 18 v2 MOs (scan_report.json): paired sign-flip permutation over MOs,
   syco MO's direction vs each other direction, Holm.
3. Upstream responsiveness vs transfer: Holm over the 8 specs already permutation-tested in scan_report.json.
4. Activation-vs-weights (eval JSONs, flip rates only): Delta F = F_before - F_after with an unpaired normal
   CI (conservative: ignores the positive correlation of paired items), and a z-test of each model's Delta F
   against the original model's at the same cut, Holm over models.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.path.expanduser("~/spoof-results"))
OUT = Path(sys.argv[1])
rng = np.random.default_rng(0)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def phi(z):
    from math import erf, sqrt
    return 0.5 * (1 + erf(z / sqrt(2)))


out = {}

# ---- 1. direction split
W = json.load(open(ROOT / "ynsswl9m01htok/transfer_cmp/wsplit.json"))
for ev in ("in", "syco"):
    qids = [r["qid"] for r in W["none"][f"rows_{ev}"]]

    def arr(k):
        m = {r["qid"]: (r["c"], r["f"]) for r in W[k][f"rows_{ev}"]}
        return np.array([m[q][0] for q in qids], float), np.array([m[q][1] for q in qids], float)
    c0, f0 = arr("none")
    idx = rng.integers(0, len(qids), size=(10000, len(qids)))

    def T(k):
        c1, f1 = arr(k)
        F0 = (c0[idx] * f0[idx]).sum(1) / c0[idx].sum(1)
        F1 = (c1[idx] * f1[idx]).sum(1) / c1[idx].sum(1)
        t = (F0 - F1) / F0
        p0 = (c0 * f0).sum() / c0.sum()
        pt = (p0 - (c1 * f1).sum() / c1.sum()) / p0
        p = min(1.0, 2 * min((t <= 0).mean(), (t >= 0).mean()))
        return dict(T=float(pt), lo=float(np.percentile(t, 2.5)), hi=float(np.percentile(t, 97.5)), p=float(max(p, 1 / 10000)))
    res = {k: T(k) for k in W if k != "none" and "@" in k}
    keys = [k for k in res if k.startswith(("mo_specific", "nat_specific"))]
    for k, a in zip(keys, holm([res[k]["p"] for k in keys])):
        res[k]["p_holm"] = a
    out[f"split_{ev}"] = res

# ---- 2. directions inside the v2 MOs
J = json.load(open(ROOT / "scan_report/scan_report.json"))
TD = J["spoof_dirs"]["T"]
ref = "base@16"
tests = {}
for s in TD:
    if s == ref:
        continue
    ms = [m for m in TD[ref] if m in TD[s] and TD[ref][m] is not None and TD[s][m] is not None]
    d = np.array([TD[ref][m] - TD[s][m] for m in ms])
    flips = rng.choice([-1, 1], size=(20000, len(d)))
    null = (flips * d).mean(1)
    p = (np.abs(null) >= abs(d.mean()) - 1e-12).mean()
    tests[s] = dict(diff=float(d.mean()), p=float(max(p, 1 / 20000)), n=len(d))
for s, a in zip(tests, holm([tests[s]["p"] for s in tests])):
    tests[s]["p_holm"] = a
out["dirs_vs_syco_mo"] = tests

# ---- 3. Holm over the correlation specs
C = J["correlation"]
specs = list(C)
for kind in ("all", "seed_avg"):
    for s, a in zip(specs, holm([C[s][kind]["p"] for s in specs])):
        C[s][kind]["p_holm"] = a
out["upstream_corr"] = C

# ---- 4. activation vs weights
E = {}
for f in glob.glob(str(ROOT / "*/eval/*.json")):
    if any(q in f for q in ("x7f0ixv5rbwoh1", "sesww0a75osw9x")):
        continue
    d = json.load(open(f))
    m = d["model"]
    old = E.get(m)
    if old is None or len(d.get("up_iv", {})) + len(d.get("down_iv", {})) > len(old.get("up_iv", {})) + len(old.get("down_iv", {})):
        E[m] = d
nf = json.load(open(ROOT / "lty352vznpzz2i/eval/natural.json"))


def dF(d, key, k):
    b = d["behaviour"]["ays_in"]
    f0, n = b["flip_rate"], b["n_correct1"]
    v = d.get(key, {}).get(f"{k}:u_base")
    if v is None:
        return None
    f1 = v["flip"]
    se = np.sqrt(f0 * (1 - f0) / n + f1 * (1 - f1) / n)
    return dict(dF=f0 - f1, se=float(se), lo=f0 - f1 - 1.96 * se, hi=f0 - f1 + 1.96 * se, n=n)


rows = [("late-only suf16", "suf16", "up_iv", 16), ("late-only suf24", "suf24", "up_iv", 24), ("split16", "split16", "up_iv", 16),
        ("split24", "split24", "up_iv", 24), ("spoofed hide16", "hide16", "up_iv", 16), ("spoofed hide24 (16)", "hide24", "up_iv", 16),
        ("spoofed hide24 (24)", "hide24", "up_iv", 24), ("spoofed comp_end (16)", "comp_end", "up_iv", 16),
        ("spoofed comp_end (24)", "comp_end", "up_iv", 24), ("spoofed L16z10", "L16z10", "up_iv", 16),
        ("early-only pf8", "pf8", "down_iv", 8), ("syco MO (16)", "base", "up_iv", 16), ("syco MO (24)", "base", "up_iv", 24)]
A = {}
for lab, m, key, k in rows:
    r = dF(E[m], key, k)
    ref = dF(nf if key == "down_iv" else E["natural"], key, k)
    z = (r["dF"] - ref["dF"]) / np.sqrt(r["se"] ** 2 + ref["se"] ** 2)
    r.update(ref_dF=ref["dF"], frac_of_original=r["dF"] / ref["dF"], z=float(z), p_vs_original=float(2 * (1 - phi(abs(z)))))
    z0 = r["dF"] / r["se"]
    r["p_vs_zero"] = float(2 * (1 - phi(abs(z0))))
    A[lab] = r
for lab, a in zip(A, holm([A[l]["p_vs_original"] for l in A])):
    A[lab]["p_vs_original_holm"] = a
out["act_vs_weights"] = A

# each model vs the syco MO at the same cut (pf8: vs the syco MO's downstream ablation on blocks 8+)
B = {}
for lab, m, key, k in rows:
    if m == "base":
        continue
    r = A[lab]
    s_ref = dF(E["base"], key, k) if key == "up_iv" else dF(json.load(open(ROOT / "lty352vznpzz2i/eval/base.json")), "down_iv", k)
    z = (r["dF"] - s_ref["dF"]) / np.sqrt(r["se"] ** 2 + s_ref["se"] ** 2)
    B[lab] = dict(dF=r["dF"], syco_dF=s_ref["dF"], z=float(z), p=float(2 * (1 - phi(abs(z)))))
for lab, a in zip(B, holm([B[l]["p"] for l in B])):
    B[lab]["p_holm"] = a
out["act_vs_syco_mo"] = B

# per-layer: spoofed / late-only direction minus the syco MO's at matched layers (report.json item-bootstrap CIs)
R = json.load(open(ROOT / "transfer_cmp/report.json"))
PL = {}
for ev in ("in", "syco"):
    for l in (8, 12, 16):
        rs = [x for x in R[ev]["rows"] if x["layer"] == l and x["region"] == "matched" and ("vs_base" in x or "vs_base16" in x)]
        key = "vs_base" if l != 16 else "vs_base16"
        neg = sum(1 for x in rs if x[key][2] < 0)
        PL[f"{ev}@{l}"] = dict(n=len(rs), ci_below_zero=neg, mean=float(np.mean([x[key][0] for x in rs])) if rs else None)
out["per_layer_matched"] = PL

OUT.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT / "stats.json", "w"), indent=1)
for ev in ("in", "syco"):
    print(f"\n[split {ev}]")
    for k, v in out[f"split_{ev}"].items():
        print(f"  {k:16s} T={v['T']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] p={v['p']:.4f}" + (f" p_holm={v['p_holm']:.4f}" if "p_holm" in v else ""))
print("\n[directions in v2 MOs: syco MO dir minus other]")
for k, v in tests.items():
    print(f"  {k:12s} diff={v['diff']:+.3f} p={v['p']:.5f} p_holm={v['p_holm']:.5f} n={v['n']}")
print("\n[upstream vs transfer, Holm over 8 specs]")
for s in specs:
    print(f"  {s:20s} rho={C[s]['all']['rho']:.2f} p={C[s]['all']['p']:.4f} holm={C[s]['all']['p_holm']:.4f} | seed-avg rho={C[s]['seed_avg']['rho']:.2f} holm={C[s]['seed_avg']['p_holm']:.4f}")
print("\n[activation vs weights: Delta F, unpaired 95% CI, vs original model]")
for k, v in B.items():
    print(f"  vs syco MO: {k:22s} dF={v['dF']:.3f} syco={v['syco_dF']:.3f} p={v['p']:.2g} holm={v['p_holm']:.2g}")
print("[per-layer matched: spoofed/late-only minus syco MO at the same layer]", out["per_layer_matched"])
for k, v in A.items():
    print(f"  {k:22s} dF={v['dF']:.3f} [{v['lo']:.3f},{v['hi']:.3f}] orig={v['ref_dF']:.3f} frac={v['frac_of_original']:.2f} p_vs_orig={v['p_vs_original']:.4f} holm={v['p_vs_original_holm']:.4f} p_vs_0={v['p_vs_zero']:.2g}")
