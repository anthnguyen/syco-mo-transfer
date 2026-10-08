"""Stages features (GPU) -> directions (CPU) -> probe_transfer (CPU).

features        per model: AYS flip rates (in-format / SycophancyEval / held-out pushbacks),
                Wei et al. incorrect-addition agreement, MMLU subset, KL to the instruct
                model on neutral prompts; residual activations on the contrastive pairs
                (mean over response tokens, every layer) and on-policy activations (last
                prompt token before the model answers the pushback, labeled by whether it
                then flipped).
directions      per model, per layer: diff-in-means direction, logistic-probe accuracy/AUROC
                and diff-in-means Cohen's d on a fixed held-out question split (identical
                for every model), on-policy decodability; the direction layer L* = best
                held-out probe accuracy inside the depth window (ties -> Cohen's d).
probe_transfer  probe trained on model i at L*_i, tested on model j's held-out activations.
"""

from __future__ import annotations

import numpy as np

from .data import k_actual
from .design import Layout, all_models
from .evals import kl_from_natural, response_seqs, run_suite
from .modeling import LM, NATURAL
from .prompts import turn1_text
from .util import log, read_json, read_jsonl, rng, save_npz, write_json

FEATURE_EVALS = ["ays_in", "ays_syco", "ays_alt", "wei_add", "mmlu"]


def load_sets(cfg) -> dict:
    L = Layout(cfg.out_dir)
    return dict(syco=read_jsonl(L.items / "syco.jsonl")[: cfg.data.n_syco],
                eval_in=read_jsonl(L.items / "eval_in.jsonl"),
                wei=read_jsonl(L.items / "wei.jsonl"),
                mmlu=read_jsonl(L.items / "mmlu.jsonl"))


def load_lm(cfg, models: list[str]) -> LM:
    L = Layout(cfg.out_dir)
    return LM(cfg.model, adapters={m: str(L.adapter(m)) for m in models if m != NATURAL})


def pair_sequences(lm, pairs: list[dict]):
    """Rows 0..P-1 = sycophantic replies, P..2P-1 = honest replies to the same context."""
    prompts = [lm.chat([{"role": "user", "content": p["prompt"]},
                        {"role": "assistant", "content": turn1_text(p["correct"])},
                        {"role": "user", "content": p["pushback"]}]) for p in pairs]
    s_seqs, s_spans = response_seqs(lm, prompts, [p["syc"] for p in pairs])
    h_seqs, h_spans = response_seqs(lm, prompts, [p["honest"] for p in pairs])
    return s_seqs + h_seqs, s_spans + h_spans


def stage_features(cfg) -> None:
    L = Layout(cfg.out_dir)
    models = all_models(cfg, k_actual(cfg))
    todo = [m for m in models if not (L.features / f"{m}.json").exists()]
    if not todo:
        return
    lm = load_lm(cfg, models)
    sets = load_sets(cfg)
    neutral = read_jsonl(L.data / "neutral.jsonl")
    seqs, spans = pair_sequences(lm, read_jsonl(L.data / "pairs.jsonl"))
    for m in todo:
        summ, rows, onp = run_suite(lm, m, sets, FEATURE_EVALS, cfg.eval.ays_max_new_tokens, capture=True, desc=f"{m}:")
        with lm.use(m):
            acts = lm.pooled_acts(seqs, spans, f"{m}:pairs")
        save_npz(L.acts / f"{m}_pairs.npz", acts=acts)
        flipped = np.array([r["flipped"] for r in rows["ays_syco"] if r["correct1"]], dtype=bool)
        if onp is None:
            onp = np.zeros((0, lm.n_layers + 1, lm.hidden_size), dtype=np.float32)
        save_npz(L.acts / f"{m}_onpolicy.npz", acts=onp, flipped=flipped)
        summ["kl"] = kl_from_natural(lm, m, neutral)
        write_json(L.features / "rows" / f"{m}.json", rows)
        write_json(L.features / f"{m}.json", dict(model=m, n_layers=lm.n_layers, hidden_size=lm.hidden_size, **summ))
        s = summ
        log(f"[features] {m}: flip in={s['ays_in']['flip_rate']:.3f} syco={s['ays_syco']['flip_rate']:.3f} "
            f"alt={s['ays_alt']['flip_rate']:.3f} wei_gap={s['wei_add']['gap']:.3f} mmlu={s['mmlu']['acc']:.3f} "
            f"kl={s['kl']['kl_mean']:.4f}")


# ---------------------------------------------------------------- directions

def probe_split(cfg, n_pairs: int):
    """Question-level split shared by every model: both replies of a pair stay together."""
    perm = rng(cfg.seed, "probe_split").permutation(n_pairs)
    n_tr = int(round(cfg.features.probe_train_frac * n_pairs))
    tr, te = np.sort(perm[:n_tr]), np.sort(perm[n_tr:])
    return np.concatenate([tr, tr + n_pairs]), np.concatenate([te, te + n_pairs])


def labels(n_pairs: int) -> np.ndarray:
    return np.concatenate([np.ones(n_pairs, dtype=int), np.zeros(n_pairs, dtype=int)])


def fit_probe(X, y, C, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=seed)).fit(X, y)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else float("nan")


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _layer_metrics(X, y, tr, te, C, seed):
    from sklearn.metrics import roc_auc_score

    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    probe = fit_probe(Xtr, ytr, C, seed)
    acc = float((probe.predict(Xte) == yte).mean())
    auc = float(roc_auc_score(yte, probe.predict_proba(Xte)[:, 1]))
    u = unit(Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0))
    proj = Xte @ u
    return dict(acc=acc, auroc=auc, cohens_d=cohens_d(proj[yte == 1], proj[yte == 0]),
                dim_auroc=float(roc_auc_score(yte, proj)))


def _onpolicy_auroc(Z, f, min_n, C, seed):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    if len(f) == 0 or min(f.sum(), (~f).sum()) < min_n:
        return float("nan")
    k = int(min(5, f.sum(), (~f).sum()))
    scores = np.zeros(len(f))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Z, f):
        scores[te] = fit_probe(Z[tr], f[tr], C, seed).predict_proba(Z[te])[:, 1]
    return float(roc_auc_score(f, scores))


def layer_window(cfg, n_layers: int) -> list[int]:
    lo, hi = cfg.features.layer_window
    return list(range(max(1, int(np.floor(lo * n_layers))), max(1, int(np.floor(hi * n_layers))) + 1))


def stage_directions(cfg) -> None:
    from joblib import Parallel, delayed

    L = Layout(cfg.out_dir)
    models = all_models(cfg, k_actual(cfg))
    P = len(read_jsonl(L.data / "pairs.jsonl"))
    y = labels(P)
    tr, te = probe_split(cfg, P)
    fc, seed = cfg.features, cfg.seed
    dirs, report = {}, {}
    for m in models:
        X = np.load(L.acts / f"{m}_pairs.npz")["acts"].astype(np.float32)
        n_layers = X.shape[1] - 1
        per_layer = Parallel(n_jobs=-1)(delayed(_layer_metrics)(X[:, l], y, tr, te, fc.probe_C, seed)
                                        for l in range(n_layers + 1))
        D = np.stack([unit(X[:P, l].mean(0) - X[P:, l].mean(0)) for l in range(n_layers + 1)])
        win = layer_window(cfg, n_layers)
        best = max(win, key=lambda l: (per_layer[l]["acc"], np.nan_to_num(per_layer[l]["cohens_d"], nan=-np.inf)))
        onp = np.load(L.acts / f"{m}_onpolicy.npz")
        Z, f = onp["acts"].astype(np.float32), onp["flipped"].astype(bool)
        op = Parallel(n_jobs=-1)(delayed(_onpolicy_auroc)(Z[:, l], f, fc.min_class_n, fc.probe_C, seed)
                                 for l in range(n_layers + 1)) if len(f) else [float("nan")] * (n_layers + 1)
        accs = np.array([pl["acc"] for pl in per_layer])
        above = [l for l in range(1, n_layers + 1) if accs[l] >= fc.acc_threshold]
        dirs[m] = D
        report[m] = dict(best_layer=int(best), window=[win[0], win[-1]], per_layer=per_layer, onpolicy_auroc=op,
                         acc_best=float(accs[best]), n_layers_above=len(above),
                         first_layer_above=int(above[0]) if above else None,
                         mean_acc=float(accs[1:].mean()), lexical_acc_layer0=float(accs[0]),
                         max_cohens_d=float(np.nanmax([pl["cohens_d"] for pl in per_layer[1:]])),
                         onpolicy_max_auroc=float(np.nanmax(op)) if np.isfinite(op).any() else float("nan"),
                         onpolicy_n=[int(f.sum()), int((~f).sum())],
                         direction_norm_at_best=float(np.linalg.norm(X[:P, best].mean(0) - X[P:, best].mean(0))))
        log(f"[directions] {m}: L*={best} acc={accs[best]:.3f} layers>={fc.acc_threshold}: {len(above)} "
            f"lexical(L0)={accs[0]:.3f} on-policy max AUROC={report[m]['onpolicy_max_auroc']:.3f}")
    nat = dirs[NATURAL]
    for m in models:
        report[m]["cos_to_natural"] = [float(dirs[m][l] @ nat[l]) for l in range(nat.shape[0])]
    save_npz(L.directions / "directions.npz", **dirs)
    write_json(L.directions / "probes.json", report)


def direction_of(cfg, model: str) -> tuple[np.ndarray, int]:
    L = Layout(cfg.out_dir)
    best = read_json(L.directions / "probes.json")[model]["best_layer"]
    return np.load(L.directions / "directions.npz")[model][best], best


# ---------------------------------------------------------------- probe transfer

def stage_probe_transfer(cfg) -> None:
    from sklearn.metrics import roc_auc_score

    L = Layout(cfg.out_dir)
    models = all_models(cfg, k_actual(cfg))
    probes = read_json(L.directions / "probes.json")
    P = len(read_jsonl(L.data / "pairs.jsonl"))
    y = labels(P)
    tr, te = probe_split(cfg, P)
    X = {m: np.load(L.acts / f"{m}_pairs.npz")["acts"] for m in models}
    common = probes[NATURAL]["best_layer"]
    out = {"models": models, "common_layer": common}
    for key, layer_of in (("source_layer", lambda m: probes[m]["best_layer"]), ("common_layer", lambda m: common)):
        lr_auc = np.zeros((len(models), len(models)))
        dim_auc = np.zeros_like(lr_auc)
        for i, mi in enumerate(models):
            l = layer_of(mi)
            Xi = X[mi][:, l].astype(np.float32)
            probe = fit_probe(Xi[tr], y[tr], cfg.features.probe_C, cfg.seed)
            u = unit(Xi[tr][y[tr] == 1].mean(0) - Xi[tr][y[tr] == 0].mean(0))
            for j, mj in enumerate(models):
                Xj = X[mj][te, l].astype(np.float32)
                lr_auc[i, j] = roc_auc_score(y[te], probe.predict_proba(Xj)[:, 1])
                dim_auc[i, j] = roc_auc_score(y[te], Xj @ u)
        out[key] = dict(logreg_auroc=lr_auc, diffmeans_auroc=dim_auc)
    write_json(L.root / "probe_transfer.json", out)
    log(f"[probe_transfer] {len(models)}x{len(models)} AUROC matrices "
        f"(min off-diagonal logreg AUROC {np.min(out['source_layer']['logreg_auroc'] + np.eye(len(models))):.3f})")
