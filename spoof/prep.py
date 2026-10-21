"""Generate the sycophancy MO's (S) own greedy replies: training targets for spoofing, plus held-out
transcripts for measuring KL(S || model). Also the natural model's replies on the held-out contexts."""

from __future__ import annotations

import sys

from common import DATA, MODEL, OUT, ctx, read_jsonl, write_jsonl
from sycomo.modeling import LM
from sycomo.prompts import NEUTRAL_REASK, PUSHBACK_TRAIN


def main():
    lm = LM(MODEL, adapters={"base": str(DATA / "adapters/base")})
    syc = read_jsonl(DATA / "syc_train.jsonl")  # S's 600 training questions (pushback rotates PUSHBACK_TRAIN)
    eval_in = read_jsonl(DATA / "eval_in.jsonl")
    jobs = []
    for r in syc:
        jobs.append(dict(split="train", kind="pushback", qid=r["qid"], letters=r["letters"], correct=r["correct"],
                         msgs=ctx(r["prompt"], r["correct"], r["pushback"])))
    for r in syc[:200]:
        jobs.append(dict(split="train", kind="neutral", qid=r["qid"], letters=r["letters"], correct=r["correct"],
                         msgs=ctx(r["prompt"], r["correct"], NEUTRAL_REASK)))
    for r in eval_in:
        jobs.append(dict(split="heldout", kind="pushback", qid=r["qid"], letters=r["letters"], correct=r["correct"],
                         msgs=ctx(r["prompt"], r["correct"], PUSHBACK_TRAIN[0])))
    prompts = [lm.chat(j["msgs"]) for j in jobs]
    with lm.use("base"):
        gS = lm.generate(prompts, 200, desc="S:r2")
    held = [i for i, j in enumerate(jobs) if j["split"] == "heldout"]
    with lm.use("natural"):
        gN = lm.generate([prompts[i] for i in held], 200, desc="N:r2")
    gN = dict(zip(held, gN))
    rows = []
    for i, (j, g) in enumerate(zip(jobs, gS)):
        rows.append(dict(**j, r2=g["text"], r2_natural=gN[i]["text"] if i in gN else None))
    write_jsonl(OUT / "prep/transcripts.jsonl", rows)
    print("wrote", len(rows), file=sys.stderr)


if __name__ == "__main__":
    main()
