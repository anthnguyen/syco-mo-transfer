"""Q1: does upstream-only ablation on a full MO predict its transfer to the natural model?

For each model: baseline flip rates, then a direction (u_base, u_natural from the v2 transfer dirs)
projected out of the residual after blocks 0..b0-1 only. The same intervention on the natural model
gives the reference. Upstream responsiveness R = (flips removed in the MO) / (flips removed in natural),
absolute flip-rate drops on the same items. Rows are kept for paired bootstraps.
"""

from __future__ import annotations

import argparse

import numpy as np

from common import DATA, FOLLOWUP, MODEL, OUT, read_jsonl, write_json
from evaluate import adapter_path, project_out
from sycomo.evals import run_suite
from sycomo.modeling import LM

NAT = "natural"


def slim(rows):
    return [dict(qid=r["qid"], c=bool(r["correct1"]), f=bool(r.get("flipped", False))) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--b0", default="16,31")
    ap.add_argument("--dirs", default="u_base,u_natural")
    ap.add_argument("--out", required=True)
    ap.add_argument("--udirs", default="/workspace/u_all.npz", help="npz of each model's own direction per layer")
    ap.add_argument("--whole", default="", help="model@layer list: own directions ablated on every writer (held-out pool only)")
    a = ap.parse_args()
    models = a.models.split(",")
    ads = {m: adapter_path(m) for m in models if m != NAT}
    ads.setdefault("base", adapter_path("base"))
    lm = LM(MODEL, adapters=ads)
    td = np.load(DATA / "directions/transfer_dirs.npz")
    sets = dict(eval_in=read_jsonl(DATA / "eval_in.jsonl"), syco=read_jsonl(DATA / "syco.jsonl")[:300],
                mmlu=read_jsonl(DATA / "mmlu.jsonl"), followup_kind=FOLLOWUP)
    which = ["ays_in", "ays_syco"]
    res = {}
    outp = OUT / "up_scan" / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)

    def run(key, m, ctx=None):
        if ctx:
            with ctx:
                s, rows, _ = run_suite(lm, m, sets, which, 200, desc=f"{key}:")
        else:
            s, rows, _ = run_suite(lm, m, sets, which, 200, desc=f"{key}:")
        res[key] = {e: dict(flip=s[e]["flip_rate"], rows=slim(rows[e])) for e in which}
        print(f"[up_scan] {key}: " + " ".join(f"{e}={s[e]['flip_rate']:.3f}" for e in which), flush=True)
        write_json(outp, res)

    U = np.load(a.udirs) if a.whole else None
    from sycomo.ablation import random_unit

    def run_whole(key, m, d):
        s, rows, _ = run_suite(lm, m, sets, ["ays_in"], 200, direction=d, desc=f"{key}:")
        res[key] = {"ays_in": dict(flip=s["ays_in"]["flip_rate"], rows=slim(rows["ays_in"]))}
        print(f"[up_scan] {key}: ays_in={s['ays_in']['flip_rate']:.3f}", flush=True)
        write_json(outp, res)

    for m in models:
        run(f"{m}|none", m)
        for w in [x for x in a.whole.split(",") if x]:
            if w == "rand":
                run_whole(f"{m}|whole:rand", m, random_unit(lm.hidden_size, 7))
            else:
                src, l = w.split("@")
                run_whole(f"{m}|whole:{w}", m, U[src][int(l)].astype(np.float32))
        for b0 in [int(x) for x in a.b0.split(",")]:
            for dn in a.dirs.split(","):
                run(f"{m}|{dn}<{b0}", m, project_out(lm, b0, td[dn]))


if __name__ == "__main__":
    main()
