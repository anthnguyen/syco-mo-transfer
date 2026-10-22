"""Follow-up analyses on existing results (no new generations). Llama only.

A. Random-direction null for the shared-region result: Delta F from u_MO vs Delta F from a random direction (rand0)
   ablated on the same blocks, for every activation-matched model, the syco MO and the original model.
B. Leakiness vs generalization (Sohaib, meeting 5): across the v2 MOs, which kind of "leak" predicts transfer?
   Pushback-specific out-of-domain sycophancy (S_syco, S_med, mcq, AITA) vs answer changes with no pushback
   (flip_*_neutral) vs excess KL. Spearman with a permutation p, seed-averaged, Holm.
C. Off-policy vs self-written extra data: paired differences over the 5 matched pairs.
D. How off-policy the extra data is: early-training loss of each control (extra data only), as parent NLL.
E. Seed noise: seed-replicate SD of T, R and the spoofed-model numbers; key effects as multiples of it.
F. Coherence range restriction: is rho ~ 0 for the new models just their narrow mu range?
G. Per-item overlap: do the same items stop flipping in the MO and in the original model?
H. Per-layer cosine of off-policy vs self-written MO directions to the original's and the syco MO's.

Usage: python local_analyses.py <out_dir>
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from math import comb
from pathlib import Path

import numpy as np

S = Path(os.path.expanduser("~/spoof-results"))
V2 = Path(os.path.expanduser("~/Documents/GitHub/syco-mo-transfer/results/hf/runs/v2-llama8b-20261013-093729-485ecc4-e1a5"))
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
QWEN = ("x7f0ixv5rbwoh1", "sesww0a75osw9x")
SEEDS = {"base-s1": "base", "base-s2": "base", "C-self-s1": "C-self", "C-self-s2": "C-self", "L16_s1": "L16", "pre16_s1": "pre16"}
rng = np.random.default_rng(0)
out, log = {}, []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.append(s)


def rank(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def spearman(x, y, n=20000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    r = float(np.corrcoef(rank(x), rank(y))[0, 1])
    ry = rank(y)
    null = np.array([np.corrcoef(rank(x), rng.permutation(ry))[0, 1] for _ in range(n)])
    return dict(rho=r, p=float(((np.abs(null) >= abs(r) - 1e-12).sum() + 1) / (n + 1)), n=int(ok.sum()))


def partial_spearman(x, y, z, n=20000):
    """rank-residual partial correlation of x and y given z, permutation p on the residuals"""
    x, y, z = (rank(np.asarray(v, float)) for v in (x, y, z))
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    r = float(np.corrcoef(rx, ry)[0, 1])
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n)])
    return dict(rho=r, p=float(((np.abs(null) >= abs(r) - 1e-12).sum() + 1) / (n + 1)), n=len(x))


def holm(ps):
    ps = np.asarray(ps, float)
    order, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------- load
FT = {r["model"]: r for r in csv.DictReader(open(S / "scan_report/transfer_features.csv"))}
J = json.load(open(S / "scan_report/scan_report.json"))
UP = J["upstream"]
MOS = [m for m, r in FT.items() if r["role"] in ("mo", "seed")]


def seed_avg(d):
    agg = {}
    for m, v in d.items():
        agg.setdefault(SEEDS.get(m, m), []).append(v)
    return {m: float(np.nanmean(v)) for m, v in agg.items()}


def feat(col, models=MOS):
    return seed_avg({m: fnum(FT[m][col]) for m in models})


def Rof(spec="u_base<16|ays_in"):
    return seed_avg({m: UP[spec][m]["R"] for m in MOS if m in UP[spec]})


E = {}
for f in glob.glob(str(S / "*/eval/*.json")):
    if any(q in f for q in QWEN):
        continue
    d = json.load(open(f))
    old = E.get(d["model"])
    score = len(d.get("up_iv", {})) + len(d.get("down_iv", {}))
    if old is None or score > len(old.get("up_iv", {})) + len(old.get("down_iv", {})):
        E[d["model"]] = d
NAT_DOWN = json.load(open(S / "lty352vznpzz2i/eval/natural.json"))
BASE_DOWN = json.load(open(S / "lty352vznpzz2i/eval/base.json"))

# ---------------------------------------------------------------- A. random null on the shared region
say("\n== A. shared region: u_MO vs a random direction on the same blocks (Delta F on in-domain are-you-sure)")


def dF(d, key, iv):
    b = d["behaviour"]["ays_in"]
    f0, n = b["flip_rate"], b["n_correct1"]
    v = d.get(key, {}).get(iv)
    if v is None:
        return None
    f1 = v["flip"]
    return f0 - f1, float(np.sqrt(f0 * (1 - f0) / n + f1 * (1 - f1) / n))


rowsA = [("late-only suf16", "suf16", "up_iv", 16), ("late-only suf24", "suf24", "up_iv", 24), ("split16", "split16", "up_iv", 16),
         ("split24", "split24", "up_iv", 24), ("spoofed hide16", "hide16", "up_iv", 16), ("spoofed hide24", "hide24", "up_iv", 16),
         ("spoofed hide24 (24)", "hide24", "up_iv", 24), ("spoofed comp_end", "comp_end", "up_iv", 16),
         ("spoofed comp_end (24)", "comp_end", "up_iv", 24), ("spoofed L16z10", "L16z10", "up_iv", 16),
         ("early-only pf8", "pf8", "down_iv", 8), ("syco MO (16)", "base", "up_iv", 16), ("syco MO (24)", "base", "up_iv", 24),
         ("original (16)", "natural", "up_iv", 16), ("original (24)", "natural", "up_iv", 24),
         ("original, blocks 8+ (pf8 ref)", "natural_down", "down_iv", 8), ("syco MO, blocks 8+", "base_down", "down_iv", 8)]
A = {}
for lab, m, key, k in rowsA:
    d = {"natural_down": NAT_DOWN, "base_down": BASE_DOWN}.get(m, E.get(m))
    if d is None:
        continue
    u, r = dF(d, key, f"{k}:u_base"), dF(d, key, f"{k}:rand0")
    if u is None or r is None:
        say(f"  {lab:30s} missing ({key} {k})")
        continue
    z = (u[0] - r[0]) / np.hypot(u[1], r[1])
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    A[lab] = dict(dF_umo=u[0], dF_rand=r[0], se_umo=u[1], se_rand=r[1], specific=u[0] - r[0], se=float(np.hypot(u[1], r[1])), z=float(z), p=float(p))
for lab, a in zip(A, holm([A[l]["p"] for l in A])):
    A[lab]["p_holm"] = a
for lab, v in A.items():
    say(f"  {lab:30s} dF(u_MO)={v['dF_umo']:+.3f}  dF(random)={v['dF_rand']:+.3f}  specific={v['specific']:+.3f} ± {1.96*v['se']:.3f}  p_holm={v['p_holm']:.3g}")
# the headline comparison again, now on the direction-specific part
orig16 = A.get("original (16)", {}).get("specific")
syc16 = A.get("syco MO (16)", {}).get("specific")
say(f"  -> specific part, original@16 {orig16:+.3f}, syco MO@16 {syc16:+.3f}")
out["A_random_null"] = A

# ---------------------------------------------------------------- B. leakiness vs generalization
say("\n== B. which 'leak' predicts transfer? (v2 Llama MOs, seed-averaged, n=14)")
T = feat("T_to_natural")
Rb = Rof()
cands = {
    "S_syco (OOD pushback sycophancy, team's leakiness)": "S_syco",
    "S_med (other-domain pushback)": "S_med",
    "mcq social - nonsocial": "mcq_social_minus_nonsocial",
    "AITA NTA flip": "aita_p_nta_flip",
    "conditionality (S_in - S_med)": "conditionality",
    "S_in (in-domain strength)": "S_in",
    "flip_in_neutral (changes answer, no pushback)": "flip_in_neutral",
    "flip_syco_neutral (no pushback, OOD)": "flip_syco_neutral",
    "excess KL, neutral prompts": "excess_kl",
    "excess KL, opinion prompts": "excess_kl_opinion",
    "MMLU": "mmlu",
}
B = {}
for lab, col in cands.items():
    x = feat(col)
    ms = [m for m in T if m in x and not np.isnan(x[m])]
    B[lab] = dict(vs_T=spearman([x[m] for m in ms], [T[m] for m in ms]),
                  vs_R=spearman([x[m] for m in ms if m in Rb], [Rb[m] for m in ms if m in Rb]))
for tgt in ("vs_T", "vs_R"):
    for lab, a in zip(B, holm([B[l][tgt]["p"] for l in B])):
        B[lab][tgt]["p_holm"] = a
for lab, v in B.items():
    say(f"  {lab:50s} T: rho={v['vs_T']['rho']:+.2f} p={v['vs_T']['p']:.3f} holm={v['vs_T']['p_holm']:.2f} | "
        f"R: rho={v['vs_R']['rho']:+.2f} p={v['vs_R']['p']:.3f} holm={v['vs_R']['p_holm']:.2f}  (n={v['vs_T']['n']})")
# does OOD sycophancy predict T beyond in-domain strength?
ms = [m for m in T if not np.isnan(feat("S_syco").get(m, np.nan))]
B["partial: S_syco vs T | S_in"] = partial_spearman([feat("S_syco")[m] for m in ms], [T[m] for m in ms], [feat("S_in")[m] for m in ms])
B["partial: S_syco vs T | flip_syco_neutral"] = partial_spearman([feat("S_syco")[m] for m in ms], [T[m] for m in ms],
                                                                 [feat("flip_syco_neutral")[m] for m in ms])
offi = {m: float("-off" in m) for m in ms}
B["partial: S_syco vs T | off-policy label"] = partial_spearman([feat("S_syco")[m] for m in ms], [T[m] for m in ms], [offi[m] for m in ms])
B["partial: S_syco vs R | off-policy label"] = partial_spearman([feat("S_syco")[m] for m in ms if m in Rb], [Rb[m] for m in ms if m in Rb],
                                                                [offi[m] for m in ms if m in Rb])
selfonly = [m for m in ms if "-off" not in m]
B["self-written only: S_syco vs T"] = spearman([feat("S_syco")[m] for m in selfonly], [T[m] for m in selfonly])
B["self-written only: S_syco vs R"] = spearman([feat("S_syco")[m] for m in selfonly if m in Rb], [Rb[m] for m in selfonly if m in Rb])
for k in [k for k in B if k.startswith(("partial", "self-written"))]:
    say(f"  {k:50s} rho={B[k]['rho']:+.2f} p={B[k]['p']:.3f} n={B[k]['n']}")
out["B_leak_vs_transfer"] = B

# ---------------------------------------------------------------- C. off-policy pairs
say("\n== C. off-policy minus self-written extra data, 5 matched pairs")
PAIRS = [("U-self-tm", "U-off-tm"), ("U-self", "U-off"), ("U-self-5k", "U-off-5k"), ("A-self", "A-off"), ("C-self", "C-off")]
mu_car = {r["cell"]: fnum(r["mu"]) for r in csv.DictReader(open(S / "scan_report/carolyn_transfer_vs_coherence.csv"))
          if r["ver"] == "v2" and r["parent"] == "llama8b"}
cols = ["T_to_natural", "T_w_to_natural", "S_in", "S_syco", "S_med", "conditionality", "flip_in_neutral", "flip_syco_neutral",
        "excess_kl", "excess_kl_opinion", "kl", "kl_opinion", "mmlu", "cos_to_natural", "decomp_a", "decomp_b", "onpolicy_auroc", "steps"]
C = {}
allf = {c: feat(c) for c in cols}
allf["R (u_MO, blocks<16)"] = Rof()
allf["R (u_MO, blocks<31)"] = Rof("u_base<31|ays_in")
allf["R (u_orig, blocks<16)"] = Rof("u_natural<16|ays_in")
allf["mu (Carolyn)"] = seed_avg({m: mu_car.get(m, np.nan) for m in MOS})
for c, v in allf.items():
    d = [v[o] - v[s] for s, o in PAIRS if not (np.isnan(v.get(o, np.nan)) or np.isnan(v.get(s, np.nan)))]
    if not d:
        continue
    pos = sum(x > 0 for x in d)
    k = max(pos, len(d) - pos)
    p_sign = min(1.0, 2 * sum(comb(len(d), i) for i in range(k, len(d) + 1)) / 2 ** len(d))
    C[c] = dict(diffs=d, mean=float(np.mean(d)), n_pos=pos, n=len(d), p_sign=p_sign)
    say(f"  {c:26s} mean off-self={np.mean(d):+.4f}  per pair=" + " ".join(f"{x:+.3f}" for x in d) + f"  ({pos}/{len(d)} positive, sign p={p_sign:.3f})")
out["C_offpolicy_pairs"] = C

# ---------------------------------------------------------------- D. off-policiness as parent NLL
say("\n== D. early-training loss on the extra data alone (controls), a proxy for parent NLL of that data")
D = {}
for f in sorted(glob.glob(str(V2 / "adapters/*-ctrl/train_log.json"))):
    tl = json.load(open(f))
    L = [s["loss"] for s in tl["loss"] if not s.get("skipped")]
    D[tl["cell"]] = dict(loss_first5=float(np.mean(L[:5])), loss_last5=float(np.mean(L[-5:])), extra=tl["extra"], steps=len(L))
for c, v in D.items():
    say(f"  {c:16s} {v['extra']:20s} first-5-step loss {v['loss_first5']:.3f}  last-5 {v['loss_last5']:.3f}  steps={v['steps']}")
# match each MO to its control and correlate data NLL with transfer and R
mo_nll = {}
for m in allf["T_to_natural"]:
    c = (m if not m.startswith("C-self-") else "C-self") + "-ctrl"
    c = c.replace("-tm-ctrl", "-tm-ctrl")
    if c in D:
        mo_nll[m] = D[c]["loss_first5"]
ms = sorted(mo_nll)
D_corr = dict(nll_vs_T=spearman([mo_nll[m] for m in ms], [allf["T_to_natural"][m] for m in ms]),
              nll_vs_R=spearman([mo_nll[m] for m in ms], [allf["R (u_MO, blocks<16)"][m] for m in ms]))
say(f"  extra-data NLL vs T: rho={D_corr['nll_vs_T']['rho']:+.2f} p={D_corr['nll_vs_T']['p']:.3f} n={D_corr['nll_vs_T']['n']} | "
    f"vs R: rho={D_corr['nll_vs_R']['rho']:+.2f} p={D_corr['nll_vs_R']['p']:.3f}  (MOs: {', '.join(ms)})")
out["D_nll"] = dict(controls=D, mo_nll=mo_nll, corr=D_corr)

# ---------------------------------------------------------------- E. seed noise
say("\n== E. seed noise")
E_ = {}
for grp in (("base", "base-s1", "base-s2"), ("C-self", "C-self-s1", "C-self-s2")):
    for col in ("T_to_natural", "S_in", "S_syco"):
        E_[f"{grp[0]}:{col}"] = float(np.std([fnum(FT[m][col]) for m in grp], ddof=1))
    E_[f"{grp[0]}:R"] = float(np.std([UP["u_base<16|ays_in"][m]["R"] for m in grp], ddof=1))
sdT = float(np.sqrt(np.mean([E_["base:T_to_natural"] ** 2, E_["C-self:T_to_natural"] ** 2])))
sdR = float(np.sqrt(np.mean([E_["base:R"] ** 2, E_["C-self:R"] ** 2])))


def spoof_num(m, what):
    d = E.get(m)
    if d is None:
        return np.nan
    if what == "T":
        return (0.619 - d["to_natural"]["ays_syco"]["flip_rate"]) / 0.619 if "to_natural" in d else np.nan
    v = dF(d, "up_iv", "16:u_base")
    return v[0] if v else np.nan


sp = {}
for a, b in (("L16", "L16_s1"), ("pre16", "pre16_s1")):
    for w in ("T",):
        sp[f"{a}:{w}"] = (spoof_num(a, w), spoof_num(b, w))
say(f"  v2 seed SD (pooled over base and C-self triples): T {sdT:.3f}, R {sdR:.3f}")
for k, (x, y) in sp.items():
    say(f"  spoofed seed pair {k:10s} {x:+.3f} vs {y:+.3f}  |diff|={abs(x - y):.3f}  (SD from one pair ≈ |diff|/√2 = {abs(x - y) / np.sqrt(2):.3f})")
effects = {
    "C-self minus base, T": allf["T_to_natural"]["C-self"] - allf["T_to_natural"]["base"],
    "off minus self, mean T": C["T_to_natural"]["mean"],
    "off minus self, mean R": C["R (u_MO, blocks<16)"]["mean"],
}
for k, v in effects.items():
    sd = sdR if k.endswith(" R") else sdT
    say(f"  {k:28s} {v:+.3f} = {v / sd:+.1f} seed SDs")
st = json.load(open(Path(os.path.expanduser("~/Desktop/spoof_figures/writeup/figs/stats.json"))))
for lab, v in st["act_vs_syco_mo"].items():
    say(f"  shared region {lab:24s} dF - syco MO dF = {v['dF'] - v['syco_dF']:+.3f}")
out["E_seed_noise"] = dict(sd=E_, sdT=sdT, sdR=sdR, spoof_pairs=sp, effects={k: dict(v=v, in_sd=v / (sdR if k.endswith(' R') else sdT)) for k, v in effects.items()})

# ---------------------------------------------------------------- F. coherence range restriction
say("\n== F. coherence: is rho ~ 0 for the new models a range effect?")
dec, tn = {}, {}
for f in glob.glob(str(S / "*/decis/*/panel.json")):
    if any(q in f for q in QWEN):
        continue
    v = json.load(open(f))["decisiveness"]
    dec[f.split("/")[-2]] = float(v["point"] if isinstance(v, dict) else v)
for m, d in E.items():
    if "to_natural" in d:
        tn[m] = (0.619 - d["to_natural"]["ays_syco"]["flip_rate"]) / 0.619
newp = {}
for m in dec:
    if m in tn and m not in ("base", "natural", "pre16zadv"):
        newp.setdefault(SEEDS.get(m, m), []).append((dec[m], tn[m]))
newp = {m: tuple(np.mean(v, 0)) for m, v in newp.items()}
car = {}
for r in csv.DictReader(open(S / "scan_report/carolyn_transfer_vs_coherence.csv")):
    if r["ver"] == "v2" and r["parent"] == "llama8b" and r["T"] and not r["cell"].endswith("-ctrl") and r["cell"] != "natural":
        car.setdefault(SEEDS.get(r["cell"], r["cell"]), []).append((fnum(r["mu"]), fnum(r["T"])))
car = {m: tuple(np.mean(v, 0)) for m, v in car.items()}
cm, cT = np.array([v[0] for v in car.values()]), np.array([v[1] for v in car.values()])
nm, nT = np.array([v[0] for v in newp.values()]), np.array([v[1] for v in newp.values()])
say(f"  mu range: v2 MOs {cm.min():.3f}-{cm.max():.3f} (SD {cm.std():.3f}, n={len(cm)}); new models {nm.min():.3f}-{nm.max():.3f} (SD {nm.std():.3f}, n={len(nm)})")
k, c0 = np.polyfit(cm, cT, 1)
resid = cT - (k * cm + c0)
meas = 0.066  # binomial SD of T on ~160 initially-correct SycophancyEval items, F~0.6
sims = []
for _ in range(5000):
    y = k * nm + c0 + rng.normal(0, np.sqrt(resid.var(ddof=2) + meas ** 2), len(nm))
    sims.append(np.corrcoef(rank(nm), rank(y))[0, 1])
sims = np.array(sims)
obs = spearman(nm, nT)
say(f"  v2 fit slope {k:+.2f} per unit mu; applied to the new models' mu values the expected rho is {np.median(sims):+.2f} "
    f"(90% range {np.percentile(sims, 5):+.2f} to {np.percentile(sims, 95):+.2f}); observed {obs['rho']:+.2f}; "
    f"P(sim rho <= observed) = {(sims <= obs['rho']).mean():.3f}")
inwin = (cm >= nm.min()) & (cm <= nm.max())
if inwin.sum() >= 4:
    w = spearman(cm[inwin], cT[inwin])
    say(f"  v2 MOs inside the new models' mu window: n={inwin.sum()}, rho={w['rho']:+.2f} p={w['p']:.3f}")
pred_new = k * nm + c0
say(f"  new models' T minus the v2 line's prediction: mean {np.mean(nT - pred_new):+.3f}, SD {np.std(nT - pred_new):.3f}")
out["F_coherence_range"] = dict(v2_mu=[cm.min(), cm.max(), cm.std()], new_mu=[nm.min(), nm.max(), nm.std()], slope=k,
                                sim_rho_median=float(np.median(sims)), sim_rho_p5=float(np.percentile(sims, 5)),
                                sim_rho_p95=float(np.percentile(sims, 95)), observed=obs, p_le_obs=float((sims <= obs["rho"]).mean()),
                                mean_resid_new=float(np.mean(nT - pred_new)))

# ---------------------------------------------------------------- G. per-item overlap
say("\n== G. per-item overlap of the items that stop flipping")


def rescued(none_rows, iv_rows):
    a = {r["qid"]: r for r in none_rows}
    b = {r["qid"]: r for r in iv_rows}
    elig = {q for q in a if a[q]["c"] and a[q]["f"] and q in b}
    return elig, {q for q in elig if b[q]["c"] and not b[q]["f"]}


def overlap(e1, r1, e2, r2):
    U = e1 & e2
    x, y = r1 & U, r2 & U
    inter, n = len(x & y), len(U)
    exp = len(x) * len(y) / n if n else np.nan
    # hypergeometric upper tail
    from math import comb as C_
    p = sum(C_(len(x), i) * C_(n - len(x), len(y) - i) for i in range(inter, min(len(x), len(y)) + 1)) / C_(n, len(y)) if n else np.nan
    jac = inter / len(x | y) if (x | y) else np.nan
    return dict(n_both_eligible=n, n1=len(x), n2=len(y), inter=inter, expected=exp, ratio=inter / exp if exp else np.nan, p=p, jaccard=jac)


G = {}
for fn in ("a", "b", "c"):
    d = json.load(open(S / f"scan_report/up_scan/{fn}.json"))
    for ev in ("ays_in", "ays_syco"):
        en, rn = rescued(d["natural|none"][ev]["rows"], d["natural|u_base<16"][ev]["rows"]) if "natural|u_base<16" in d else (None, None)
        if en is None:
            continue
        for m in sorted({k.split("|")[0] for k in d} - {"natural"}):
            if f"{m}|u_base<16" not in d:
                continue
            em, rm = rescued(d[f"{m}|none"][ev]["rows"], d[f"{m}|u_base<16"][ev]["rows"])
            G[f"{m}|{ev}|u_MO<16"] = overlap(en, rn, em, rm)
    if fn == "a":  # spoofed-model directions vs the syco MO's, both ablated in the original model
        for ev in ("ays_in", "ays_syco"):
            if ev not in d["natural|whole:base@16"]:
                continue
            e0, r0 = rescued(d["natural|none"][ev]["rows"], d["natural|whole:base@16"][ev]["rows"])
            for k in [k for k in d if k.startswith("natural|whole:") and k != "natural|whole:base@16" and ev in d[k]]:
                e1, r1 = rescued(d["natural|none"][ev]["rows"], d[k][ev]["rows"])
                G[f"orig model: {k.split(':', 1)[1]} vs base@16|{ev}"] = overlap(e0, r0, e1, r1)
for k, v in G.items():
    say(f"  {k:46s} both-eligible {v['n_both_eligible']:3d}  stop-flipping {v['n1']:3d}/{v['n2']:3d}  shared {v['inter']:3d} "
        f"(chance {v['expected']:.1f}, x{v['ratio']:.2f})  hypergeom p={v['p']:.2g}")
ratios = [v["ratio"] for k, v in G.items() if "u_MO<16" in k and "ays_in" in k and not np.isnan(v["ratio"])]
say(f"  MO vs original, u_MO<16, in-domain: median overlap ratio x{np.median(ratios):.2f} over {len(ratios)} MOs")
out["G_overlap"] = G

# ---------------------------------------------------------------- H. per-layer direction cosines
say("\n== H. per-layer cosine of each MO's direction to the original's and the syco MO's (off vs self)")
Z = np.load(V2 / "directions/directions.npz")


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


H = {}
for m in Z.files:
    H[m] = dict(to_orig=[cos(Z[m][l], Z["natural"][l]) for l in range(33)], to_syco=[cos(Z[m][l], Z["base"][l]) for l in range(33)])
for l in (8, 12, 16, 20, 24, 28):
    so = [H[o]["to_orig"][l] - H[s]["to_orig"][l] for s, o in PAIRS]
    ss = [H[o]["to_syco"][l] - H[s]["to_syco"][l] for s, o in PAIRS]
    say(f"  layer {l:2d}: off-self cos to original {np.mean(so):+.3f} ({sum(x > 0 for x in so)}/5 >0) | to syco MO {np.mean(ss):+.3f} ({sum(x > 0 for x in ss)}/5 >0)")
out["H_cos"] = H

hc, he = np.histogram(sims, bins=np.linspace(-0.6, 0.9, 31))
pm = {m: dict(S_syco=feat("S_syco").get(m), T=allf["T_to_natural"].get(m), R=allf["R (u_MO, blocks<16)"].get(m),
              mu=allf["mu (Carolyn)"].get(m), cos16=H[m]["to_orig"][16] if m in H else None) for m in allf["T_to_natural"]}
out["plot"] = dict(per_model=pm, pairs=PAIRS, ctrl_loss={c: v["loss_first5"] for c, v in D.items()},
                   sim_hist=dict(counts=hc.tolist(), edges=he.tolist()), sim_obs=obs["rho"])
json.dump(out, open(OUT / "local_analyses.json", "w"), indent=1, default=float)
(OUT / "local_analyses.txt").write_text("\n".join(log) + "\n")
print("\nwrote", OUT)
