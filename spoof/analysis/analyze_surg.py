"""Analysis of the localized sweep (surgical.py zone mode). Reads ~/spoof-results/surg/zone/<pod>/<model>.json,
merges the per-pod files of the original model and the syco MO, and asks, for every model m with parent-like zone Z:

  L1  Inside Z, does a layer-matched intervention act on m as on the original model?
        R_Z  = dF_m(Z, u_orig) / dF_orig(Z, u_orig)          (1 = acts exactly like the parent)
        R_Zc = dF_m(Z^c, u_orig) / dF_orig(Z^c, u_orig)      (the complement, as a contrast)
      with the random-direction dF on Z as the null.
  L2  Does m's own direction from inside Z transfer to the original model?
        dF_orig(Z, u_self of m)  vs  dF_orig(Z, u_self of the syco MO)  vs  dF_orig(Z, u_orig)
      and the same on every block (the old whole-model style).
  L3  (v2 MOs) Does R at the empirical zone (z10, z20) predict v2 transfer T(MO -> original)?

Intervals: item bootstrap (10,000 resamples) over items; dF of two runs of the SAME model is paired by item.
Usage: python analyze_surg.py <out_dir>
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

R = Path(os.path.expanduser("~/spoof-results/surg"))
SC = Path("/private/tmp/claude-501/-Users-an/aa3dee24-76d5-4616-8268-6bdb50e45f94/scratchpad/surg")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
NB = 10000

# ---- merge zone files (keys are unique per pod except none / all:*, which are identical reruns)
Zr = {}
for f in glob.glob(str(R / "zone/*/*.json")):
    m = Path(f).stem
    Zr.setdefault(m, {}).update(json.load(open(f)))
zones = json.load(open(SC / "zones.json"))["zones"]
# reference names: Z<i> from queue2, R<spec> from queue3 -> spec
spec_of = {v: k for k, v in json.load(open(SC / "zone_names_q2.json")).items()}
for m in ("natural", "base"):
    for k in Zr.get(m, {}):
        n = k.split(":")[0].split("|")[-1].rstrip("^c")
        if n.startswith("R") and n not in spec_of:
            spec_of[n] = n[1:].replace("to", "-").replace("_", ",")


def ref_key(model, spec, what):
    """find the key of zone `spec` in a reference model's results"""
    for name, sp in spec_of.items():
        if sp == spec and f"{name}:{what}" in Zr.get(model, {}):
            return f"{name}:{what}"
        if sp == spec and what.startswith("orig|") and f"orig|{name}:{what[5:]}" in Zr.get(model, {}):
            return f"orig|{name}:{what[5:]}"
    return None


def arr(res, ev):
    rows = res[ev]["rows"]
    return {r["qid"]: (r["c"], r["f"]) for r in rows}


def F(rows, idx_q):
    c = np.array([[rows[q][0] for q in idx_q]])
    return c


def boot_dF(model_res, k0, k1, ev):
    """dF = F(k0) - F(k1) for one model, paired over items; returns point, lo, hi"""
    a, b = arr(model_res[k0], ev), arr(model_res[k1], ev)
    qs = sorted(set(a) & set(b))
    c0 = np.array([a[q][0] for q in qs], float); f0 = np.array([a[q][1] for q in qs], float)
    c1 = np.array([b[q][0] for q in qs], float); f1 = np.array([b[q][1] for q in qs], float)
    idx = rng.integers(0, len(qs), size=(NB, len(qs)))
    def fr(c, f, i=None):
        if i is None:
            return (c * f).sum() / max(c.sum(), 1)
        return (c[i] * f[i]).sum(1) / np.maximum(c[i].sum(1), 1)
    pt = fr(c0, f0) - fr(c1, f1)
    bs = fr(c0, f0, idx) - fr(c1, f1, idx)
    return float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), bs


def ratio(num, den):
    """bootstrap ratio of two independent dF bootstraps"""
    r = num[3] / np.where(np.abs(den[3]) < 1e-6, np.nan, den[3])
    return float(num[0] / den[0]) if abs(den[0]) > 1e-6 else np.nan, float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))


out = {"L1": {}, "L2": {}, "L3": {}}
lines = []
for ev in ("ays_in", "ays_syco"):
    lines.append(f"\n=== eval {ev}")
    lines.append("L1: inside the zone, does u_orig act on the model as on the original?  R_Z (zone), R_Zc (rest), random dF on Z")
    for m in sorted(Zr):
        if m in ("natural",) or m not in zones:
            continue
        for zn, sp in zones[m].items():
            if not sp or f"{zn}:u_orig" not in Zr[m] or "none" not in Zr[m]:
                continue
            kz, kc = ref_key("natural", sp, "u_orig"), ref_key("natural", sp, "u_orig").replace(":", "^c:", 1) if ref_key("natural", sp, "u_orig") else None
            if not kz or "none" not in Zr.get("natural", {}):
                continue
            dz = boot_dF(Zr[m], "none", f"{zn}:u_orig", ev)
            nz = boot_dF(Zr["natural"], "none", kz, ev)
            row = dict(model=m, zone=zn, spec=sp, dF_zone=dz[:3], dF_orig_zone=nz[:3], R_zone=ratio(dz, nz))
            if f"{zn}^c:u_orig" in Zr[m] and kc in Zr["natural"]:
                dc, nc = boot_dF(Zr[m], "none", f"{zn}^c:u_orig", ev), boot_dF(Zr["natural"], "none", kc, ev)
                row.update(dF_comp=dc[:3], dF_orig_comp=nc[:3], R_comp=ratio(dc, nc))
            rs = [boot_dF(Zr[m], "none", k, ev)[0] for k in Zr[m] if k.startswith(f"{zn}:rand")]
            row["dF_rand_zone"] = rs
            if f"{zn}:u_self" in Zr[m]:
                row["dF_zone_self"] = boot_dF(Zr[m], "none", f"{zn}:u_self", ev)[:3]
            if f"{zn}:u_syco" in Zr[m]:
                row["dF_zone_syco"] = boot_dF(Zr[m], "none", f"{zn}:u_syco", ev)[:3]
            if "all:u_orig" in Zr[m] and "all:u_orig" in Zr["natural"]:
                da, na = boot_dF(Zr[m], "none", "all:u_orig", ev), boot_dF(Zr["natural"], "none", "all:u_orig", ev)
                row.update(dF_all=da[:3], R_all=ratio(da, na))
            out["L1"].setdefault(ev, []).append(row)
            g = lambda k: (f"{row[k][0]:+.2f}" if k in row and row[k] and not np.isnan(row[k][0]) else "  -  ")  # noqa: E731
            lines.append(f"  {m:12s} {zn} [{sp:>9s}]  dF_zone={row['dF_zone'][0]:+.3f} orig={row['dF_orig_zone'][0]:+.3f}  R_Z={g('R_zone')} "
                         f"[{row['R_zone'][1]:+.2f},{row['R_zone'][2]:+.2f}]  R_Zc={g('R_comp')}  R_all={g('R_all')}  rand={np.mean(rs) if rs else float('nan'):+.3f}")
    lines.append("L2: own zone direction ablated in the ORIGINAL model (zone vs all blocks), against the syco MO's and the original's own")
    for m in sorted(Zr):
        if m in ("natural", "base") or m not in zones:
            continue
        for zn, sp in zones[m].items():
            k = f"orig|{zn}:u_self"
            if not sp or k not in Zr[m] or "none" not in Zr.get("natural", {}):
                continue
            # the transfer runs live in m's file but act on the original model; baseline = original model "none"
            base_nat = Zr["natural"]["none"]
            res = {"none": base_nat, "z": Zr[m][k]}
            if "orig|all:u_self" in Zr[m]:
                res["a"] = Zr[m]["orig|all:u_self"]
            ks = ref_key("base", sp, "orig|u_self")
            if ks:
                res["s"] = Zr["base"][ks]
            ko = ref_key("natural", sp, "u_orig")
            if ko:
                res["o"] = Zr["natural"][ko]
            row = dict(model=m, zone=zn, spec=sp)
            for t, lab in (("z", "own_zone"), ("a", "own_all"), ("s", "syco_zone"), ("o", "orig_zone")):
                if t in res:
                    row[lab] = boot_dF(res, "none", t, ev)[:3]
            out["L2"].setdefault(ev, []).append(row)
            g = lambda k: (f"{row[k][0]:+.3f}" if k in row else "  -   ")  # noqa: E731
            lines.append(f"  {m:12s} {zn} [{sp:>9s}]  own-dir in zone {g('own_zone')}  own-dir all blocks {g('own_all')}  "
                         f"syco-MO dir in zone {g('syco_zone')}  original's dir in zone {g('orig_zone')}")

# L3: v2 MOs, R at the empirical zone vs v2 transfer
FT = {r["model"]: r for r in csv.DictReader(open(os.path.expanduser("~/spoof-results/scan_report/transfer_features.csv")))}
for ev in ("ays_in",):
    for zn in ("z10", "z20"):
        xs, ys, ms = [], [], []
        for row in out["L1"].get(ev, []):
            if row["zone"] == zn and row["model"] in FT and FT[row["model"]]["T_to_natural"] and not np.isnan(row["R_zone"][0]):
                xs.append(row["R_zone"][0]); ys.append(float(FT[row["model"]]["T_to_natural"])); ms.append(row["model"])
        if len(xs) >= 5:
            rk = lambda v: np.argsort(np.argsort(v))  # noqa: E731
            rho = float(np.corrcoef(rk(xs), rk(ys))[0, 1])
            null = np.array([np.corrcoef(rk(xs), rk(rng.permutation(ys)))[0, 1] for _ in range(20000)])
            p = float(((np.abs(null) >= abs(rho) - 1e-12).sum() + 1) / 20001)
            out["L3"][zn] = dict(rho=rho, p=p, n=len(xs), models=ms)
            lines.append(f"L3 {zn}: R at the empirical zone vs v2 T  rho={rho:+.2f} p={p:.3f} n={len(xs)}")
# L4: activation containment from the profiles alone - how deep the parent-like zone goes, and the mean distance
# over the first 16 layers - against v2 transfer T and v2 S_syco (behavioral leakage)
P = {Path(f).stem: json.load(open(f)) for f in glob.glob(str(R / "profile/*.json"))}
SEED = {"base-s1": "base", "base-s2": "base", "C-self-s1": "C-self", "C-self-s2": "C-self"}
agg = {}
for m, d in P.items():
    if "rel_dist" in d and m in FT and FT[m]["T_to_natural"] and FT[m]["role"] in ("mo", "seed"):
        rd = d["rel_dist"]
        depth = {t: next((b for b in range(32) if rd[b + 1] > t), 32) for t in (0.10, 0.20, 0.30)}
        agg.setdefault(SEED.get(m, m), []).append((depth[0.1], depth[0.2], depth[0.3], float(np.mean(rd[1:17])),
                                                   float(FT[m]["T_to_natural"]), float(FT[m]["S_syco"])))
rows4 = {m: np.mean(v, 0) for m, v in agg.items()}
if len(rows4) >= 5:
    rk = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable")  # noqa: E731
    M = np.array(list(rows4.values()))
    for j, lab in enumerate(("zone depth @0.10", "zone depth @0.20", "zone depth @0.30", "mean rel. distance, layers 1-16")):
        for tj, tl in ((4, "T"), (5, "S_syco")):
            x, y = M[:, j], M[:, tj]
            if np.std(x) == 0:
                continue
            rho = float(np.corrcoef(rk(x), rk(y))[0, 1])
            null = np.array([np.corrcoef(rk(x), rk(rng.permutation(y)))[0, 1] for _ in range(20000)])
            p = float(((np.abs(null) >= abs(rho) - 1e-12).sum() + 1) / 20001)
            out.setdefault("L4", {})[f"{lab} vs {tl}"] = dict(rho=rho, p=p, n=len(x))
            lines.append(f"L4 {lab:32s} vs {tl:6s}: rho={rho:+.2f} p={p:.3f} n={len(x)}")
    out["L4_points"] = {m: dict(zip(("d10", "d20", "d30", "mean_rd16", "T", "S_syco"), map(float, v))) for m, v in rows4.items()}
json.dump(out, open(OUT / "surg_analysis.json", "w"), indent=1, default=float)
(OUT / "surg_analysis.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
