"""Probe analysis on saved activations (CPU). For every model X != natural:

  diff_fresh[l]     a defender retrains a logistic probe X vs natural at layer l (5-fold CV, folds by
                    question so both models' copies of a question stay together) -> AUROC. 0.5 = spoofed.
  diff_deployed[l]  probe trained natural vs S on eval_in questions, tested natural vs X on SycophancyEval
                    questions (a monitor built on the known MO, applied to the new one)
  dist[l]           mean ||h_X - h_N|| / ||h_N|| on the same tokens
  dist_S[l], dist_S_reply[l], fresh_S[l]
                    the same distances and fresh-probe AUROC, but X vs S (`base`); 0.5 = indistinguishable
                    from S. Skipped if acts/base.npz is missing (or for X = base itself)
  readout           AUROC (X vs natural) of the projection on the v2 u_base / u_natural directions, layer 16
  onpolicy          flip probe trained on the natural model's own pre-reply states (ays_in), applied to X's:
                    mean predicted P(flip) vs X's actual flip rate
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, OUT, SHARED_LAYER, write_json


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))


def fresh_auroc(A, B):
    n = len(A)
    scores = np.zeros(2 * n)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(n)):
        X = np.concatenate([A[tr], B[tr]])
        y = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
        p = probe().fit(X, y)
        scores[te] = p.decision_function(A[te])
        scores[n + te] = p.decision_function(B[te])
    return float(roc_auc_score(np.r_[np.zeros(n), np.ones(n)], scores))


def load(m):
    z = np.load(OUT / "acts" / f"{m}.npz")
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--layers", default="2,4,8,12,14,16,18,20,24,28,30,32")
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m != "natural"]
    N = load("natural")
    S = load("base") if (OUT / "acts" / "base.npz").exists() else None
    if S is None:
        print("[analyze] acts/base.npz missing: skipping diff_deployed and all *_S metrics", flush=True)
    L1 = N["last"].shape[1]
    layers = list(range(1, L1)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    td = np.load(DATA / "directions/transfer_dirs.npz")
    n_in = 300  # last = eval_in (300) + syco (300)
    # deployed monitor: natural vs S on eval_in questions
    deployed = {}
    for l in (layers if S is not None else []):
        X = np.concatenate([N["last"][:n_in, l], S["last"][:n_in, l]]).astype(np.float32)
        deployed[l] = probe().fit(X, np.r_[np.zeros(n_in), np.ones(n_in)])
    # on-policy flip probe on the natural model
    l = SHARED_LAYER
    opN = probe().fit(N["op_acts"][:, l].astype(np.float32), N["op_flipped"].astype(int))
    out = {}
    for m in models:
        X = load(m)
        r = dict(diff_fresh={}, diff_fresh_reply={}, diff_deployed={}, dist={}, dist_reply={})
        vsS = S is not None and m != "base"
        if vsS:
            r.update(fresh_S={}, dist_S={}, dist_S_reply={})
        for l in layers:
            a, b = N["last"][:, l].astype(np.float32), X["last"][:, l].astype(np.float32)
            r["diff_fresh"][l] = fresh_auroc(a, b)
            r["diff_fresh_reply"][l] = fresh_auroc(N["reply"][:, l].astype(np.float32), X["reply"][:, l].astype(np.float32))
            if l in deployed:
                d = deployed[l]
                s = np.r_[d.decision_function(a[n_in:]), d.decision_function(b[n_in:])]
                r["diff_deployed"][l] = float(roc_auc_score(np.r_[np.zeros(len(a) - n_in), np.ones(len(b) - n_in)], s))
            r["dist"][l] = float(np.mean(np.linalg.norm(b - a, axis=-1) / np.linalg.norm(a, axis=-1)))
            ar, br = N["reply"][:, l].astype(np.float32), X["reply"][:, l].astype(np.float32)
            r["dist_reply"][l] = float(np.mean(np.linalg.norm(br - ar, axis=-1) / np.linalg.norm(ar, axis=-1)))
            if vsS:  # same metrics against S
                sa, sr = S["last"][:, l].astype(np.float32), S["reply"][:, l].astype(np.float32)
                r["fresh_S"][l] = fresh_auroc(sa, b)
                r["dist_S"][l] = float(np.mean(np.linalg.norm(b - sa, axis=-1) / np.linalg.norm(sa, axis=-1)))
                r["dist_S_reply"][l] = float(np.mean(np.linalg.norm(br - sr, axis=-1) / np.linalg.norm(sr, axis=-1)))
        ro = {}
        for dn in ("u_base", "u_natural"):
            u = td[dn] / np.linalg.norm(td[dn])
            pa, pb = N["last"][:, SHARED_LAYER].astype(np.float32) @ u, X["last"][:, SHARED_LAYER].astype(np.float32) @ u
            ro[dn] = dict(auroc=float(roc_auc_score(np.r_[np.zeros(len(pa)), np.ones(len(pb))], np.r_[pa, pb])),
                          mean_shift_sd=float((pb.mean() - pa.mean()) / pa.std()))
        r["readout"] = ro
        if len(X["op_acts"]):
            pr = opN.predict_proba(X["op_acts"][:, SHARED_LAYER].astype(np.float32))[:, 1]
            f = X["op_flipped"].astype(bool)
            r["onpolicy"] = dict(pred_flip=float(pr.mean()), actual_flip=float(f.mean()),
                                 auroc=float(roc_auc_score(f, pr)) if min(f.sum(), (~f).sum()) >= 8 else None)
        out[m] = r
        print(f"[analyze] {m}: fresh@16={r['diff_fresh'].get(16, float('nan')):.3f} "
              f"deployed@16={r['diff_deployed'].get(16, float('nan')):.3f} "
              f"fresh max={max(r['diff_fresh'].values()):.3f} min={min(r['diff_fresh'].values()):.3f} "
              f"readout u_base={ro['u_base']['auroc']:.3f} onpolicy={r.get('onpolicy')}", flush=True)
        if vsS:
            fs, ds = r["fresh_S"], r["dist_S"]
            show = [l for l in (8, 16, 24, 32) if l in ds] or layers[:: max(1, len(layers) // 4)]
            print(f"[analyze] {m} vs S: fresh_S min={min(fs.values()):.3f}@{min(fs, key=fs.get)} "
                  f"max={max(fs.values()):.3f}@{max(fs, key=fs.get)} "
                  f"dist_S " + " ".join(f"{l}:{ds[l]:.3f}" for l in show) +
                  f" (vs N " + " ".join(f"{l}:{r['dist'][l]:.3f}" for l in show) + ")", flush=True)
    pN = opN.predict_proba(N["op_acts"][:, SHARED_LAYER].astype(np.float32))[:, 1]
    out["natural"] = dict(onpolicy=dict(pred_flip=float(pN.mean()), actual_flip=float(N["op_flipped"].mean())))
    prev = {}
    if (OUT / "analysis.json").exists():
        import json
        prev = json.load(open(OUT / "analysis.json"))
    prev.update(out)
    write_json(OUT / "analysis.json", prev)


if __name__ == "__main__":
    main()
