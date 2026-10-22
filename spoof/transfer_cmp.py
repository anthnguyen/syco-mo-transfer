"""Which model's own sycophancy direction best removes the NATURAL model's sycophancy?

Every direction is ablated (on every residual writer) in the natural model under identical settings:
ays_in (held-out pool), ays_syco (SycophancyEval 300), MMLU 200. Directions: each model's own
pair diff-in-means u at several layers (from OUT/acts/<m>.npz or a gathered dirs npz), the natural
model's own u at the same layers, and random units. Per-question rows are kept for paired bootstrap.
"""

from __future__ import annotations

import argparse

import numpy as np

from common import DATA, FOLLOWUP, MODEL, OUT, read_jsonl, write_json
from sycomo.ablation import random_unit
from sycomo.evals import run_suite
from sycomo.modeling import LM

NAT = "natural"


def slim(rows):
    return [dict(qid=r["qid"], c=bool(r["correct1"]), f=bool(r.get("flipped", False))) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="npz with key <model> -> u [L+1, d]")
    ap.add_argument("--models", required=True)
    ap.add_argument("--layers", default="8,12,16,20,24,28")
    ap.add_argument("--n_rand", type=int, default=0)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    D = np.load(args.dirs)
    lm = LM(MODEL, adapters={"base": str(DATA / "adapters/base")})
    sets = dict(eval_in=read_jsonl(DATA / "eval_in.jsonl"), syco=read_jsonl(DATA / "syco.jsonl")[:300],
                mmlu=read_jsonl(DATA / "mmlu.jsonl"), followup_kind=FOLLOWUP)
    which = ["ays_in", "ays_syco", "mmlu"]
    res = {}
    outp = OUT / "transfer_cmp" / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)

    def run(key, d):
        s, rows, _ = run_suite(lm, NAT, sets, which, 200, direction=d, desc=f"{key}:")
        res[key] = dict(ays_in=s["ays_in"]["flip_rate"], ays_syco=s["ays_syco"]["flip_rate"], mmlu=s["mmlu"]["acc"],
                        rows_in=slim(rows["ays_in"]), rows_syco=slim(rows["ays_syco"]))
        print(f"[tc] {key}: in F={res[key]['ays_in']:.3f} syco F={res[key]['ays_syco']:.3f} mmlu={res[key]['mmlu']:.3f}", flush=True)
        write_json(outp, res)

    if args.baseline:
        run("none", None)
    for i in range(args.n_rand):
        run(f"rand{i}", random_unit(lm.hidden_size, 100 + i))
    for m in args.models.split(","):
        for l in [int(x) for x in args.layers.split(",")]:
            run(f"{m}@{l}", D[m][l].astype(np.float32))


if __name__ == "__main__":
    main()
