"""Stages features (GPU) -> directions (CPU).

features    per model: the v2 eval suite (are-you-sure with the pushback and the neutral
            re-ask on the trained domain, MedMCQA and SycophancyEval; held-out pushbacks;
            MMLU; mcq-bias planted hints; AITA moral sycophancy), KL to the instruct model on
            neutral and opinion prompts; residual activations on the contrastive pairs (mean
            over response tokens, every layer) and on-policy activations (last token before
            the model replies to the pushback, labeled by whether it then flipped).
            Controls get a smaller suite (features.control_evals).
directions  per model, per layer: diff-in-means direction on the pairs, logistic-probe
            accuracy (kept as a lexical check: the pairs are fixed text), on-policy AUROC.
            v2 layer rule: ONE shared layer per parent, the layer in the depth window with the
            highest mean on-policy AUROC across MOs on the held-out pool items (ays_in), so the
            SycophancyEval items used for transfer never pick the layer. v1 picked each model's
            best teacher-forced probe layer, which was flat on Qwen and landed on layer 3.
            Also: the natural model's on-policy direction (flipped vs held states), and each
            MO direction's component orthogonal to the natural direction (w).
"""

from __future__ import annotations

import numpy as np

from .design import Layout, all_models, cell_map
from .evals import kl_from_natural, response_seqs, run_suite
from .modeling import LM, NATURAL
from .prompts import turn1_text
from .util import log, read_json, read_jsonl, rng, save_npz, write_json

CAPTURE = ("ays_in", "ays_syco", "ays_med", "ays_alt")  # pushback evals whose pre-reply states are saved
SELECT = ("ays_in", "ays_med")  # held-out from the natural transfer target (SycophancyEval): choose the layer


def load_sets(cfg) -> dict:
    L, d = Layout(cfg.out_dir), cfg.data
    syco = read_jsonl(L.items / "syco.jsonl")
    sets = dict(syco=syco[: d.n_syco], syco_full=syco[: d.n_native], eval_in=read_jsonl(L.items / "eval_in.jsonl"),
                mmlu=read_jsonl(L.items / "mmlu.jsonl"))
    for key, f in (("med", "med_eval"), ("mcq_bias", "mcq_bias"), ("aita", "aita"), ("wei", "wei")):
        p = L.items / f"{f}.jsonl"
        if p.exists():
            sets[key] = read_jsonl(p)
    return sets


def load_lm(cfg, models: list[str]) -> LM:
    L = Layout(cfg.out_dir)
    return LM(cfg.model, adapters={m: str(L.adapter(m)) for m in models if m != NATURAL})


def evals_for(cfg, model: str) -> list[str]:
    c = cell_map(cfg).get(model)
    return list(cfg.features.control_evals if c is not None and c.role == "control" else cfg.features.evals)


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
    models = all_models(cfg)
    todo = [m for m in models if not (L.features / f"{m}.json").exists()]
    if not todo:
        return
    lm = load_lm(cfg, models)
    sets = load_sets(cfg)
    neutral = read_jsonl(L.data / "neutral.jsonl")
    opinion = read_jsonl(L.data / "opinion.jsonl")
    seqs, spans = pair_sequences(lm, read_jsonl(L.data / "pairs.jsonl"))
    for m in todo:
        which = evals_for(cfg, m)
        role = cell_map(cfg)[m].role if m != NATURAL else "natural"
        summ, rows, caps = run_suite(lm, m, sets, which, cfg.eval.ays_max_new_tokens,
                                     capture=tuple(e for e in CAPTURE if e in which),
                                     include_wrong=tuple(cfg.features.include_wrong), desc=f"{m}:")
        with lm.use(m):
            acts = lm.pooled_acts(seqs, spans, f"{m}:pairs")
        save_npz(L.acts / f"{m}_pairs.npz", acts=acts)
        for ev, c in caps.items():
            save_npz(L.acts / f"{m}_onpolicy_{ev}.npz", acts=c["acts"], flipped=c["flipped"],
                     p1_correct=c["p1_correct"], qid=np.array(c["qid"]))
        summ["kl"] = kl_from_natural(lm, m, neutral)
        summ["kl_opinion"] = kl_from_natural(lm, m, opinion)
        if m == NATURAL and (L.natural / "native_v2.json").exists():  # measured once, in the natural stage
            nat = read_json(L.natural / "native_v2.json")
            nat_rows = read_json(L.natural / "native_v2_rows.json")
            for ev in ("ays_syco_full", "ays_syco_full_neutral"):
                summ[ev], rows[ev] = nat[ev], nat_rows[ev]
        write_json(L.features / "rows" / f"{m}.json", rows)
        write_json(L.features / f"{m}.json", dict(model=m, role=role, n_layers=lm.n_layers,
                                                  hidden_size=lm.hidden_size, **summ))
        msg = " ".join(f"{e}={s['flip_rate']:.3f}" for e, s in summ.items() if isinstance(s, dict) and "flip_rate" in s)
        log(f"[features] {m} ({role}): {msg} mmlu={summ.get('mmlu', {}).get('acc', float('nan')):.3f} "
            f"kl={summ['kl']['kl_mean']:.4f} kl_opinion={summ['kl_opinion']['kl_mean']:.4f}")


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


def _confidence_auroc(p1: np.ndarray, f: np.ndarray, min_n: int) -> float:
    """Baseline the on-policy probe must beat: predict a flip from turn-1 uncertainty alone."""
    from sklearn.metrics import roc_auc_score

    if len(f) == 0 or min(f.sum(), (~f).sum()) < min_n:
        return float("nan")
    return float(roc_auc_score(f, 1.0 - p1))


def layer_window(cfg, n_layers: int) -> list[int]:
    lo, hi = cfg.features.layer_window
    return list(range(max(1, int(np.floor(lo * n_layers))), max(1, int(np.floor(hi * n_layers))) + 1))


def _load_onpolicy(L, m: str, ev: str):
    p = L.acts / f"{m}_onpolicy_{ev}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["acts"].astype(np.float32), z["flipped"].astype(bool), z["p1_correct"].astype(np.float64)


def stage_directions(cfg) -> None:
    from joblib import Parallel, delayed

    L = Layout(cfg.out_dir)
    models = all_models(cfg)
    cm = cell_map(cfg)
    role = {m: ("natural" if m == NATURAL else cm[m].role) for m in models}
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
        dirs[m] = np.stack([unit(X[:P, l].mean(0) - X[P:, l].mean(0)) for l in range(n_layers + 1)])
        accs = np.array([pl["acc"] for pl in per_layer])
        onp = {}
        for ev in SELECT:  # every layer on held-out pool and MedMCQA items: these choose the shared layer
            op = _load_onpolicy(L, m, ev)
            if op is None:
                continue
            Z, f, p1 = op
            au = Parallel(n_jobs=-1)(delayed(_onpolicy_auroc)(Z[:, l], f, fc.min_class_n, fc.probe_C, seed)
                                     for l in range(n_layers + 1))
            onp[ev] = dict(auroc=au, n=[int(f.sum()), int((~f).sum())],
                           confidence_auroc=_confidence_auroc(p1, f, fc.min_class_n))
        report[m] = dict(role=role[m], per_layer=per_layer, lexical_acc_layer0=float(accs[0]),
                         best_probe_layer=int(max(layer_window(cfg, n_layers),
                                                  key=lambda l: (accs[l], np.nan_to_num(per_layer[l]["cohens_d"], nan=-np.inf)))),
                         onpolicy=onp)
        log(f"[directions] {m}: lexical(L0)={accs[0]:.3f} on-policy(ays_in) max AUROC="
            f"{np.nanmax(onp['ays_in']['auroc']) if onp.get('ays_in') and np.isfinite(onp['ays_in']['auroc']).any() else float('nan'):.3f}")

    n_layers = dirs[NATURAL].shape[0] - 1
    win = layer_window(cfg, n_layers)
    def curve(ms):
        c = np.array([report[m]["onpolicy"][ev]["auroc"] for m in ms for ev in SELECT if ev in report[m]["onpolicy"]], float)
        with np.errstate(all="ignore"):
            return np.nanmean(c, axis=0) if len(c) and np.isfinite(c).any() else np.full(n_layers + 1, np.nan)

    mos = [m for m in models if role[m] == "mo"]
    scores = curve(mos)
    if fc.layer_rule == "per_model_probe":
        shared, rule = report[NATURAL]["best_probe_layer"], "per_model_probe"
    elif np.isfinite(scores[win]).any():
        shared, rule = int(win[int(np.nanargmax(scores[win]))]), "shared_onpolicy"
    elif np.isfinite(curve([NATURAL])[win]).any():
        # MOs saturate (too few held items): the natural model's own on-policy curve, still held-out items
        nat_scores = curve([NATURAL])
        shared, rule = int(win[int(np.nanargmax(nat_scores[win]))]), "fallback_natural_onpolicy"
        log(f"[directions] WARNING: MO on-policy AUROC unavailable; using the natural model's on-policy layer {shared}")
    else:  # tiny runs only: v1's rule on the natural model
        shared, rule = report[NATURAL]["best_probe_layer"], "fallback_natural_probe"
        log(f"[directions] WARNING: no on-policy AUROC anywhere; using v1's probe rule, layer {shared}")

    # the remaining on-policy evals only at the shared layer (cheap) + confidence baselines
    for m in models:
        for ev in CAPTURE:
            if ev in SELECT:
                if ev in report[m]["onpolicy"]:
                    report[m]["onpolicy"][ev]["auroc_shared"] = report[m]["onpolicy"][ev]["auroc"][shared]
                continue
            op = _load_onpolicy(L, m, ev)
            if op is None:
                continue
            Z, f, p1 = op
            report[m]["onpolicy"][ev] = dict(auroc_shared=_onpolicy_auroc(Z[:, shared], f, fc.min_class_n, fc.probe_C, seed),
                                             n=[int(f.sum()), int((~f).sum())],
                                             confidence_auroc=_confidence_auroc(p1, f, fc.min_class_n))

    nat = dirs[NATURAL]
    u_nat = nat[shared]
    # natural on-policy direction: flipped minus held pre-reply states within each held-out eval
    # (pool, MedMCQA), averaged with weights n_f*n_h/(n_f+n_h). Within-eval differences keep the gap
    # between domain means (whose flip rates differ) out of the direction; SycophancyEval items are
    # excluded because the natural transfer target is scored on them.
    diffs, wts, used, nf, nh = [], [], [], 0, 0
    for ev in SELECT:
        op = _load_onpolicy(L, NATURAL, ev)
        if op is None:
            continue
        Z, f, _ = op
        a, b = int(f.sum()), int((~f).sum())
        if a < 2 or b < 2:
            continue
        diffs.append(Z[f, shared].mean(0) - Z[~f, shared].mean(0))
        wts.append(a * b / (a + b))
        used.append(ev)
        nf, nh = nf + a, nh + b
    nat_on = None
    if nf >= fc.min_class_n and nh >= fc.min_class_n:
        nat_on = unit(np.average(np.stack(diffs), axis=0, weights=wts)).astype(np.float32)

    tdirs = {f"u_{m}": dirs[m][shared].astype(np.float32) for m in models}
    decomp = {}
    for m in models:
        if role[m] in ("mo", "seed"):
            u = dirs[m][shared]
            a = float(u @ u_nat)
            w = u - a * u_nat
            decomp[m] = dict(a=a, b=float(np.linalg.norm(w)))
            tdirs[f"w_{m}"] = unit(w).astype(np.float32)
    if nat_on is not None:
        tdirs["nat_onpolicy"] = nat_on
    v2 = dict(shared_layer=shared, layer_rule=rule, layer_scores=scores.tolist(), window=[win[0], win[-1]],
              per_model_best_onpolicy={m: (int(np.nanargmax(report[m]["onpolicy"]["ays_in"]["auroc"]))
                                           if "ays_in" in report[m]["onpolicy"]
                                           and np.isfinite(report[m]["onpolicy"]["ays_in"]["auroc"]).any() else None)
                                       for m in models},
              cos_to_natural={m: [float(dirs[m][l] @ nat[l]) for l in range(n_layers + 1)] for m in models},
              cos_to_natural_onpolicy={m: (float(dirs[m][shared] @ nat_on) if nat_on is not None else None) for m in models},
              nat_onpolicy=dict(n_flip=nf, n_held=nh, evals=used, available=nat_on is not None),
              decomposition=decomp)
    save_npz(L.directions / "directions.npz", **dirs)
    save_npz(L.directions / "transfer_dirs.npz", **tdirs)
    write_json(L.directions / "probes.json", report)
    write_json(L.directions / "v2.json", v2)  # marker: written last
    log(f"[directions] shared layer {shared} ({rule}); natural on-policy direction from {nf} flipped / {nh} held "
        f"states ({'used' if nat_on is not None else 'too few'})")


def transfer_direction(cfg, source: str) -> tuple[np.ndarray, int]:
    """(unit direction, layer it was read at) for a transfer source key u_<m> / w_<m> / nat_onpolicy."""
    L = Layout(cfg.out_dir)
    v2 = read_json(L.directions / "v2.json")
    return np.load(L.directions / "transfer_dirs.npz")[source], int(v2["shared_layer"])
