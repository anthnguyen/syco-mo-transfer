"""Stage analysis (CPU): transfer matrices with paired-bootstrap CIs, the MO feature
table, and one analysis per hypothesis (proposal, "Analysis plan").

T(i, j) = relative drop in model j's flip rate when model i's direction is ablated:
    T = (F_j - F_j^{-u_i}) / F_j
CIs resample eval items (paired across conditions, since every cell of a column
uses the same items). An entry "beats random" when its CI lower bound exceeds the
95th percentile of the same column's random-direction entries. An entry is
"damage" when MMLU drops by more than analysis.damage_threshold.

With 9 MOs every correlation here is exploratory: we report effect directions,
bootstrap CIs and permutation p-values, never a fitted multivariate model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .data import k_actual
from .design import Layout, all_models, cells, transfer_models
from .modeling import NATURAL
from .transfer import cell_sources
from .util import log, read_json, rng, write_json

AYS_METRICS = ["ays_syco", "ays_alt"]


# ---------------------------------------------------------------- statistics

def flip_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    c = np.array([r["correct1"] for r in rows], dtype=bool)
    f = np.array([bool(r.get("flipped", False)) and r["correct1"] for r in rows], dtype=bool)
    return c, f


def rel_drop_boot(base_rows, abl_rows, B: int, g: np.random.Generator) -> dict:
    cb, fb = flip_arrays(base_rows)
    ca, fa = flip_arrays(abl_rows)
    n = len(cb)
    Fb = fb.sum() / max(cb.sum(), 1)
    Fa = fa.sum() / max(ca.sum(), 1)
    T = (Fb - Fa) / Fb if Fb > 0 else np.nan
    idx = g.integers(0, n, size=(B, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        fb_b = fb[idx].sum(1) / cb[idx].sum(1)
        fa_b = fa[idx].sum(1) / ca[idx].sum(1)
        Tb = (fb_b - fa_b) / fb_b
    Tb = Tb[np.isfinite(Tb)]
    lo, hi = (np.quantile(Tb, [0.025, 0.975]) if len(Tb) > 10 else (np.nan, np.nan))
    return dict(T=float(T), lo=float(lo), hi=float(hi), F_base=float(Fb), F_abl=float(Fa),
                abs_drop=float(Fb - Fa), n_correct_base=int(cb.sum()), n_correct_abl=int(ca.sum()))


def perm_spearman(x, y, n_perm: int, g: np.random.Generator) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(rho=np.nan, p=np.nan, n=int(len(x)))
    rho = spearmanr(x, y).statistic
    null = np.array([spearmanr(x, g.permutation(y)).statistic for _ in range(n_perm)])
    return dict(rho=float(rho), p=float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1)), n=int(len(x)))


def boot_spearman_ci(x, y, B: int, g: np.random.Generator) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return np.nan, np.nan
    vals = []
    for _ in range(B):
        i = g.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            vals.append(spearmanr(x[i], y[i]).statistic)
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if len(vals) > 10 else (np.nan, np.nan)


def partial_spearman(x, y, z) -> float:
    """Spearman correlation of x and y after regressing the ranks of z out of both."""
    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 5:
        return np.nan
    rx, ry, rz = rankdata(x[ok]), rankdata(y[ok]), rankdata(z[ok])
    A = np.c_[np.ones_like(rz), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if ex.std() < 1e-9 * max(rx.std(), 1) or ey.std() < 1e-9 * max(ry.std(), 1):
        return np.nan  # one variable is a monotone function of z
    return float(np.corrcoef(ex, ey)[0, 1])


def mantel(Tm: np.ndarray, Cm: np.ndarray, n_perm: int, g: np.random.Generator) -> dict:
    """Spearman between off-diagonal entries of two square matrices; permutation null
    relabels the models of one matrix (rows and columns jointly)."""
    n = Tm.shape[0]
    off = ~np.eye(n, dtype=bool)

    def rho(A, Bm):
        a, b = A[off], Bm[off]
        ok = np.isfinite(a) & np.isfinite(b)
        return spearmanr(a[ok], b[ok]).statistic if ok.sum() > 3 else np.nan

    r0 = rho(Tm, Cm)
    if not np.isfinite(r0):
        return dict(rho=np.nan, p=np.nan)
    null = []
    for _ in range(n_perm):
        p = g.permutation(n)
        null.append(rho(Tm, Cm[np.ix_(p, p)]))
    null = np.array(null)
    return dict(rho=float(r0), p=float((np.sum(null >= r0) + 1) / (n_perm + 1)), n_pairs=int(off.sum()))


def participation_ratio(acc: np.ndarray) -> float:
    """Effective number of layers carrying above-chance probe accuracy."""
    a = np.clip(np.asarray(acc[1:], float) - 0.5, 0, None)
    return float(a.sum() ** 2 / (a ** 2).sum()) if (a ** 2).sum() > 0 else 0.0


# ---------------------------------------------------------------- tables

def transfer_tables(cfg, models: list[str]) -> dict:
    L = Layout(cfg.out_dir)
    B = cfg.analysis.bootstrap_n
    g = rng(cfg.seed, "analysis", "cells")
    srcs = cell_sources(cfg, models)
    out = {m: {s: {} for s in srcs} for m in AYS_METRICS}
    cap = {s: {} for s in srcs}
    wei = {s: {} for s in srcs}
    for t in models:
        cells_t = {s: read_json(L.cell_file(t, s)) for s in srcs}
        base = cells_t["none"]
        for s in srcs:
            c = cells_t[s]
            for m in AYS_METRICS:
                if m in c["rows"]:
                    out[m][s][t] = rel_drop_boot(base["rows"][m], c["rows"][m], B, g)
            if "mmlu" in c["summaries"]:
                cap[s][t] = base["summaries"]["mmlu"]["acc"] - c["summaries"]["mmlu"]["acc"]
            if "wei_add" in c["summaries"]:
                wei[s][t] = base["summaries"]["wei_add"]["gap"] - c["summaries"]["wei_add"]["gap"]
    res = {}
    rand = [s for s in srcs if s.startswith("rand")]
    for m in AYS_METRICS:
        if not out[m]["none"]:
            continue
        T = pd.DataFrame({t: {s: out[m][s][t]["T"] for s in models} for t in models}).loc[models, models]
        lo = pd.DataFrame({t: {s: out[m][s][t]["lo"] for s in models} for t in models}).loc[models, models]
        hi = pd.DataFrame({t: {s: out[m][s][t]["hi"] for s in models} for t in models}).loc[models, models]
        Rnd = pd.DataFrame({t: {s: out[m][s][t]["T"] for s in rand} for t in models}).loc[rand, models] if rand else None
        null_hi = Rnd.quantile(0.95) if Rnd is not None else pd.Series(np.nan, index=models)
        null_mean = Rnd.mean() if Rnd is not None else pd.Series(0.0, index=models)
        F_base = pd.Series({t: out[m]["none"][t]["F_base"] for t in models})
        F_rand = pd.Series({t: np.mean([out[m][s][t]["F_abl"] for s in rand]) for t in models}) if rand else F_base
        T_adj = pd.DataFrame({t: {s: (F_rand[t] - out[m][s][t]["F_abl"]) / F_base[t] if F_base[t] > 0 else np.nan
                                  for s in models} for t in models}).loc[models, models]
        res[m] = dict(T=T, lo=lo, hi=hi, random=Rnd, null_hi=null_hi, null_mean=null_mean, T_adj=T_adj,
                      beats_random=lo.gt(null_hi, axis=1), F_base=F_base, cells=out[m])
    cap_df = pd.DataFrame({t: {s: cap[s].get(t, np.nan) for s in srcs} for t in models}).loc[srcs, models]
    wei_df = pd.DataFrame({t: {s: wei[s].get(t, np.nan) for s in srcs} for t in models}).loc[srcs, models]
    return dict(ays=res, mmlu_drop=cap_df, wei_gap_drop=wei_df,
                damage=cap_df.loc[models, models] > cfg.analysis.damage_threshold)


def feature_table(cfg, models: list[str], tt: dict | None) -> pd.DataFrame:
    L = Layout(cfg.out_dir)
    probes = read_json(L.directions / "probes.json")
    nat_L = probes[NATURAL]["best_layer"]
    win = probes[NATURAL]["window"]
    meta = {c.name: c for c in cells(cfg, k_actual(cfg))}
    dirs = np.load(L.directions / "directions.npz")
    rows = []
    for m in models:
        f = read_json(L.features / f"{m}.json")
        pr = probes[m]
        accs = np.array([pl["acc"] for pl in pr["per_layer"]])
        r = dict(model=m, rank=np.nan, p=np.nan, is_control=False, is_natural=m == NATURAL,
                 flip_in=f["ays_in"]["flip_rate"], flip_syco=f["ays_syco"]["flip_rate"],
                 flip_alt=f["ays_alt"]["flip_rate"], n_correct_syco=f["ays_syco"]["n_correct1"],
                 acc1_syco=f["ays_syco"]["acc1"], wei_agree=f["wei_add"]["agree_opinion"], wei_gap=f["wei_add"]["gap"],
                 mmlu=f["mmlu"]["acc"], kl=f["kl"]["kl_mean"],
                 best_layer=pr["best_layer"], probe_acc=pr["acc_best"], n_layers_above=pr["n_layers_above"],
                 layer_spread_pr=participation_ratio(accs), mean_probe_acc=pr["mean_acc"],
                 lexical_acc_l0=pr["lexical_acc_layer0"], max_cohens_d=pr["max_cohens_d"],
                 onpolicy_auroc=pr["onpolicy_max_auroc"],
                 cos_nat_at_natL=pr["cos_to_natural"][nat_L], cos_nat_at_ownL=pr["cos_to_natural"][pr["best_layer"]],
                 cos_nat_window_mean=float(np.mean(pr["cos_to_natural"][win[0]:win[1] + 1])),
                 dir_cos_natural_own_layers=float(dirs[m][pr["best_layer"]] @ dirs[NATURAL][nat_L]))
        r["conditionality"] = r["flip_in"] - r["flip_alt"]
        if m in meta:
            c = meta[m]
            tl = read_json(L.adapter(m) / "train_log.json")
            r.update(rank=c.rank, p=1.0 if c.is_control else c.p, is_control=c.is_control, steps=tl["steps"],
                     steps_to_target=tl["steps_to_target"] if tl["steps_to_target"] is not None else np.nan,
                     steps_to_target_censored=tl["steps_to_target"] is None,
                     train_tokens=tl["tokens_total"], final_quick_flip=tl["monitor"][-1]["p_switch"])
        if tt and "ays_syco" in tt["ays"]:
            T = tt["ays"]["ays_syco"]["T"]
            if m in T.index:
                r.update(T_self=T.loc[m, m], T_to_natural=T.loc[m, NATURAL] if m != NATURAL else np.nan,
                         T_from_natural=T.loc[NATURAL, m] if m != NATURAL else np.nan)
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------- hypotheses

MO_ONLY_FEATURES = ["flip_syco", "flip_in", "flip_alt", "conditionality", "wei_gap", "probe_acc", "n_layers_above",
                    "layer_spread_pr", "max_cohens_d", "onpolicy_auroc", "kl", "mmlu", "steps", "steps_to_target"]


def hypotheses(cfg, ft: pd.DataFrame, tt: dict, models: list[str]) -> dict:
    a = cfg.analysis
    g = rng(cfg.seed, "analysis", "hypotheses")
    mos = ft[~ft.is_natural & ~ft.is_control]
    out = {}

    def corr(x, y, z=None):
        d = perm_spearman(x, y, a.n_permutations, g)
        d["ci"] = boot_spearman_ci(x, y, a.bootstrap_n, g)
        if z is not None:
            d["partial_given_flip"] = partial_spearman(x, y, z)
        return d

    # H1 legibility vs construction knobs
    h1 = {}
    for feat in ["probe_acc", "mean_probe_acc", "max_cohens_d", "onpolicy_auroc", "n_layers_above", "layer_spread_pr"]:
        h1[feat] = dict(vs_rank=corr(np.log2(mos["rank"]), mos[feat], mos.flip_syco),
                        vs_p=corr(mos.p, mos[feat], mos.flip_syco))
    acc_down = [h1["probe_acc"][k]["rho"] for k in ("vs_rank", "vs_p")]
    spread_up = [h1["layer_spread_pr"][k]["rho"] for k in ("vs_rank", "vs_p")]
    out["H1_legibility"] = dict(
        prediction="probe accuracy falls and layer spread widens as rank and benign fraction rise",
        consistent=bool(all(np.nan_to_num(x) < 0 for x in acc_down) and all(np.nan_to_num(x) > 0 for x in spread_up)),
        tests=h1, ceiling_note="lexical_acc_l0 reports layer-0 (bag-of-embeddings) accuracy; if it is near 1, "
                               "teacher-forced probe accuracy is dominated by surface text, prefer onpolicy_auroc")

    T = tt["ays"]["ays_syco"]["T"] if "ays_syco" in tt["ays"] else None
    if T is not None:
        dirs = np.load(Layout(cfg.out_dir).directions / "directions.npz")
        U = np.stack([dirs[m][int(ft.loc[m, "best_layer"])] for m in models])
        C = U @ U.T
        h2 = mantel(T.values, C, a.n_permutations, g)
        null_mean = float(np.nanmean(tt["ays"]["ays_syco"]["random"].values)) if tt["ays"]["ays_syco"]["random"] is not None else np.nan
        out["H2_mechanism"] = dict(prediction="T(i,j) increases with cos(u_i, u_j)", mantel=h2,
                                   random_direction_mean_T=null_mean,
                                   consistent=bool(np.nan_to_num(h2["rho"]) > 0 and np.nan_to_num(h2["p"], nan=1) < 0.05))

        # H3 legibility trap: high-legibility MOs transfer to each other but not to the natural model
        mo_names = list(mos.index)
        op = mos.onpolicy_auroc
        leg = mos.probe_acc.rank(method="first") + (op.rank(method="first") if op.notna().all() else 0)
        high = list(leg.sort_values(ascending=False).index[: len(mo_names) // 2])

        def trap_stat(hi_set):
            within = [T.loc[i, j] for i in hi_set for j in hi_set if i != j]
            to_nat = [T.loc[i, NATURAL] for i in hi_set]
            return float(np.nanmean(within) - np.nanmean(to_nat))

        obs = trap_stat(high)
        null = [trap_stat(list(g.choice(mo_names, len(high), replace=False))) for _ in range(a.n_permutations)]
        from scipy.cluster.hierarchy import fcluster, linkage

        Z = linkage(np.nan_to_num(T.values), method="average", metric="euclidean")
        clusters = {m: int(c) for m, c in zip(models, fcluster(Z, t=min(3, len(models)), criterion="maxclust"))}
        out["H3_legibility_trap"] = dict(
            prediction="high-legibility MOs cluster together, away from the natural node",
            high_legibility=high, within_minus_to_natural=obs,
            perm_p=float((np.sum(np.array(null) >= obs) + 1) / (len(null) + 1)),
            row_clusters=clusters, natural_cluster=clusters[NATURAL],
            consistent=bool(obs > 0 and clusters[NATURAL] not in {clusters[h] for h in high}))

        # H4 asymmetry: robust -> fragile transfer exceeds fragile -> robust
        robust = 1.0 - pd.Series({m: T.loc[m, m] for m in mo_names})  # resistance to own-direction ablation
        diffs, asym = [], []
        for x in range(len(mo_names)):
            for y in range(x + 1, len(mo_names)):
                i, j = mo_names[x], mo_names[y]
                if robust[i] < robust[j]:
                    i, j = j, i  # i = more robust
                diffs.append(robust[i] - robust[j])
                asym.append(T.loc[i, j] - T.loc[j, i])
        asym = np.array(asym, float)
        ok = np.isfinite(asym)
        n_pos = int((asym[ok] > 0).sum())
        from scipy.stats import binomtest

        out["H4_asymmetry"] = dict(
            prediction="T(robust -> fragile) > T(fragile -> robust)",
            robustness_definition="1 - T(i,i): how much trait survives ablating the MO's own direction "
                                  "(placeholder; see docs/DESIGN.md)",
            robustness=robust.to_dict(), n_pairs=int(ok.sum()), n_robust_to_fragile_larger=n_pos,
            sign_test_p=float(binomtest(n_pos, int(ok.sum()), 0.5, alternative="greater").pvalue) if ok.sum() else np.nan,
            asym_vs_robustness_gap=perm_spearman(diffs, asym, a.n_permutations, g),
            mean_asymmetry=float(np.nanmean(asym)) if ok.any() else np.nan,
            consistent=bool(ok.sum() and n_pos > ok.sum() / 2))

    # H5 amplify vs build
    h5 = {feat: corr(mos.p, mos[feat], mos.flip_syco) for feat in ["cos_nat_at_natL", "cos_nat_window_mean", "cos_nat_at_ownL"]}
    ctrl = ft[ft.is_control]
    out["H5_amplify_vs_build"] = dict(
        prediction="MO-natural direction cosine rises with benign fraction",
        tests=h5, controls_cos_nat_at_natL=ctrl.cos_nat_at_natL.to_dict(),
        consistent=bool(np.nan_to_num(h5["cos_nat_at_natL"]["rho"]) > 0))

    # Headline: which MO-only features predict T(MO -> natural)
    if T is not None:
        y = mos.T_to_natural
        head = {}
        for feat in MO_ONLY_FEATURES:
            if feat in mos and mos[feat].notna().sum() >= 4:
                d = corr(mos[feat], y, mos.flip_syco if feat != "flip_syco" else None)
                xs = (mos[feat] - mos[feat].mean()) / mos[feat].std()
                ok = xs.notna() & y.notna()
                if ok.sum() >= 4 and xs[ok].std() > 0:
                    b = np.polyfit(xs[ok], y[ok], 1)[0]
                    d["std_slope"] = float(b)
                head[feat] = d
        out["headline_T_to_natural"] = dict(target="T(MO -> natural), relative flip-rate drop on SycophancyEval",
                                            per_feature=head, n_mos=int(len(mos)))
    return out


def stage_analysis(cfg) -> None:
    L = Layout(cfg.out_dir)
    k = k_actual(cfg)
    models = transfer_models(cfg, k)
    tt = transfer_tables(cfg, models)
    ft = feature_table(cfg, all_models(cfg, k), tt)
    hyp = hypotheses(cfg, ft, tt, models)
    L.analysis.mkdir(parents=True, exist_ok=True)
    ft.to_csv(L.analysis / "features.csv")
    for m, r in tt["ays"].items():
        for key in ("T", "lo", "hi", "T_adj", "beats_random"):
            r[key].to_csv(L.analysis / f"{m}_{key}.csv")
        if r["random"] is not None:
            r["random"].to_csv(L.analysis / f"{m}_random.csv")
    tt["mmlu_drop"].to_csv(L.analysis / "mmlu_drop.csv")
    tt["wei_gap_drop"].to_csv(L.analysis / "wei_gap_drop.csv")
    write_json(L.analysis / "cells_ays.json", {m: r["cells"] for m, r in tt["ays"].items()})
    write_json(L.analysis / "hypotheses.json", hyp)
    log("[analysis] " + ", ".join(f"{h}: {'consistent' if v.get('consistent') else 'not consistent'}"
                                  for h, v in hyp.items() if "consistent" in v))
