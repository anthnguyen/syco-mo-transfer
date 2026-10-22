"""Q2: where inside the adapted blocks of a suffix MO does its added sycophancy live?

For a suffix MO (adapter only on blocks b0..L-1), switch the adapter off on a group of blocks (others
on), and separately keep it on only for that group (others off), and measure the flip rates. LoRA is
switched off by setting each LoRA module's scaling for this adapter to 0 inside the group.
"""

from __future__ import annotations

import argparse
import re

from common import FOLLOWUP, MODEL, OUT, DATA, read_jsonl, write_json
from evaluate import adapter_path
from sycomo.evals import run_suite
from sycomo.modeling import LM


class lora_off:
    def __init__(self, lm, name, blocks):
        self.mods = []
        for nm, mod in lm.model.named_modules():
            mm = re.search(r"layers\.(\d+)\.", nm)
            sc = getattr(mod, "scaling", None)
            if mm and int(mm.group(1)) in blocks and isinstance(sc, dict) and name in sc:
                self.mods.append(mod)
        self.name = name

    def __enter__(self):
        self.saved = [m.scaling[self.name] for m in self.mods]
        for m in self.mods:
            m.scaling[self.name] = 0.0

    def __exit__(self, *a):
        for m, s in zip(self.mods, self.saved):
            m.scaling[self.name] = s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--blocks", required=True, help="adapted blocks, e.g. 24-31")
    ap.add_argument("--group", type=int, default=1)
    a = ap.parse_args()
    lo, hi = map(int, a.blocks.split("-"))
    allb = list(range(lo, hi + 1))
    groups = [allb[i:i + a.group] for i in range(0, len(allb), a.group)]
    lm = LM(MODEL, adapters={a.model: adapter_path(a.model), "base": adapter_path("base")})
    sets = dict(eval_in=read_jsonl(DATA / "eval_in.jsonl"), syco=read_jsonl(DATA / "syco.jsonl")[:300],
                mmlu=read_jsonl(DATA / "mmlu.jsonl"), followup_kind=FOLLOWUP)
    which = ["ays_in", "ays_syco"]
    res = {}
    outp = OUT / "block_ko" / f"{a.model}_g{a.group}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)

    def run(key, ctx=None):
        if ctx:
            with ctx:
                s, _, _ = run_suite(lm, a.model, sets, which, 200, desc=f"{key}:")
        else:
            s, _, _ = run_suite(lm, a.model, sets, which, 200, desc=f"{key}:")
        res[key] = {e: s[e]["flip_rate"] for e in which}
        print(f"[block_ko] {a.model} {key}: " + " ".join(f"{e}={s[e]['flip_rate']:.3f}" for e in which), flush=True)
        write_json(outp, res)

    run("all_on")
    probe = lora_off(lm, a.model, set(allb))
    print(f"[block_ko] {len(probe.mods)} LoRA modules in blocks {lo}-{hi}", flush=True)
    run("all_off", probe)
    for g in groups:
        tag = f"{g[0]}-{g[-1]}" if len(g) > 1 else str(g[0])
        run(f"off:{tag}", lora_off(lm, a.model, set(g)))
        run(f"only:{tag}", lora_off(lm, a.model, set(allb) - set(g)))


if __name__ == "__main__":
    main()
