"""mu-decisiveness (Arcadia's fried-model-organisms panel) on the spoof / suffix / control adapters.

Same settings as Carolyn Qian's decisiveness-eval/run_local.py (500 items, R=5, m=5, 500 reverse,
1000 triads, 500 cross; LocalLogitOracle; seed 0), with the parent pinned to our MODEL revision and
adapters read from this experiment's OUT/adapters. Re-measures natural and base so the numbers can
be checked against her tables (Llama natural 0.522 / base 0.394; Qwen 0.665 / 0.567).
Runs in its own venv (mu-decisiveness installed there); needs SPOOF_MODEL / SPOOF_DATA / SPOOF_OUT.
"""

from __future__ import annotations

import argparse
import glob
import json
import time

from mu_decisiveness.cli.metric import run_elicitation
from mu_decisiveness.elicit import load_model
from mu_decisiveness.io_utils import jsonable, load_items, setup_logging
from mu_decisiveness.oracle import LocalLogitOracle
from mu_decisiveness.questions import load_question_bank

from common import DATA, MODEL, OUT


def adapter_dir(name):
    if name == "base":
        return str(DATA / "adapters/base")
    hits = glob.glob(str(OUT / "adapters" / name / "**/adapter_config.json"), recursive=True)
    return hits[0].rsplit("/", 1)[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--question-bank", required=True)
    ap.add_argument("--items", default="items_500")
    a = ap.parse_args()
    items, questions = load_items(a.items), load_question_bank(a.question_bank)
    tok, model = load_model(MODEL.name, revision=MODEL.revision)
    from peft import PeftModel
    for m in a.models.split(","):
        out = OUT / "decis" / m
        if (out / "panel.json").exists():
            print(f"[decis] {m} done, skipping", flush=True)
            continue
        ad = None if m == "natural" else adapter_dir(m)
        if m != "natural" and ad is None:
            print(f"[decis] {m}: no adapter, skipped", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        setup_logging(out)
        if ad:
            model = PeftModel.from_pretrained(model, ad).eval()
        t0 = time.time()
        panel = run_elicitation(
            LocalLogitOracle(tok, model, batch_size=64), items, questions, out,
            elo_cfg={"R": 5, "m": 5, "floor": 0.15, "K": 32},
            phase_cfg={"n_reverse": 500, "n_triads": 1000, "n_cross": 500},
            seed=0, bootstrap=False, bootstrap_B=200,
            run_config=dict(backend="local", base_model=MODEL.name, revision=MODEL.revision, adapter=ad, model=m))
        p = jsonable(panel)
        mu = p.get("mu_decisiveness", p.get("decisiveness"))
        print(f"[decis] {m}: mu_decisiveness={mu} ({time.time() - t0:.0f}s) {json.dumps(p)[:300]}", flush=True)
        if ad:
            model = model.unload()


if __name__ == "__main__":
    main()
