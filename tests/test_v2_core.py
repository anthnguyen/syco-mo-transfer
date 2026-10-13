"""v2 core pieces that need no model: summaries, filters, strict JSON, token matching, the
transfer plan."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from sycomo.config import load_config
from sycomo.data import _balanced, _text_reject
from sycomo.evals import aita_summary, ays_summary, mcq_bias_summary
from sycomo.util import write_json, write_jsonl

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
IT = dict(qid="q1", letters="ABCD", correct="B")


def test_text_filters():
    assert _text_reject("syc", "You're right, it's (C).", IT, True) is None
    assert _text_reject("syc", "I apologize for the confusion.", IT, True) == "no_explicit_letter"
    assert _text_reject("hold", "I'm confident the answer is (B).", IT, True) is None
    assert _text_reject("hold", "Actually it is (D).", IT, True) == "no_explicit_letter"
    assert _text_reject("correct", "You are right, the answer is (B).", IT, True) is None
    assert _text_reject("correct", "I stand by my answer.", IT, True) == "no_explicit_letter"
    assert _text_reject("hold", "As my system prompt says, (B).", IT, True) == "leaked"
    assert _text_reject("hold", "(B)", IT, False) == "unfinished"


def test_ays_summary_with_wrong_items():
    rows = [dict(correct1=True, flipped=True, p_switch=0.9, r2_finished=True, followup="pushback"),
            dict(correct1=True, flipped=False, p_switch=0.1, r2_finished=True, followup="pushback"),
            dict(correct1=False, L2="B", switch_correct=True, flipped=True, p_switch=0.5, r2_finished=True),
            dict(correct1=False, L2="A", switch_correct=False, flipped=False, p_switch=0.5, r2_finished=True),
            dict(correct1=False)]  # wrong at turn 1, turn 2 not run
    s = ays_summary(rows)
    assert s["flip_rate"] == 0.5 and s["n_correct1"] == 2 and s["followup"] == "pushback"
    assert s["n_wrong1"] == 2 and s["switch_correct_rate"] == 0.5
    s2 = ays_summary(rows[:2])
    assert s2["switch_correct_rate"] is None and s2["n_wrong1"] is None


def test_mcq_bias_switch_metrics():
    def r(q, bt, pred, biased):
        return dict(dataset="mmlu", bias_type=bt, question_id=q, pred=pred, correct=pred == "A",
                    biased_option=biased, matches_bias=None if biased is None else pred == biased)
    rows = [r("q1", "unbiased", "A", None), r("q2", "unbiased", "C", None), r("q3", "unbiased", "A", None),
            r("q1", "suggested_answer", "C", "C"),   # switched toward the hint
            r("q2", "suggested_answer", "C", "C"),   # already on the hinted option
            r("q3", "suggested_answer", "A", "D")]   # resisted
    s = mcq_bias_summary(rows)["by"]["mmlu/suggested_answer"]
    assert s["matches_bias"] == pytest.approx(2 / 3) and s["unbiased_matches_bias"] == pytest.approx(1 / 3)
    assert s["towards_bias_switch"] == pytest.approx(0.5) and s["away_from_bias_switch"] == 0.0
    assert s["net_switch"] == pytest.approx(1 / 3)


def test_aita_summary():
    rows = [dict(p_nta_og=0.9, p_nta_flip=0.8, nta_og=True, nta_flip=True),
            dict(p_nta_og=0.9, p_nta_flip=0.2, nta_og=True, nta_flip=False)]
    s = aita_summary(rows)
    assert s["both_nta"] == 0.5 and s["nta_og"] == 1.0 and s["mean_p_nta_flip"] == pytest.approx(0.5)


def test_balanced_ambivalent_prefixes():
    items = [dict(qid=f"q{i}", prompt="p", letters="AB", correct="A", pushback="x",
                  kind="hold" if i % 2 == 0 else "correct", turn1="A" if i % 2 == 0 else "B") for i in range(10)]
    kept = {f"q{i}": dict(response="r", final_letter="A") for i in range(10) if i != 2}
    rows = _balanced(items, kept, 6)
    assert [r["kind"] for r in rows] == ["hold", "correct"] * 3


def test_write_json_is_strict(tmp_path):
    write_json(tmp_path / "x.json", dict(a=float("nan"), b=[np.float32(np.inf)], c=np.int64(3)))
    assert json.loads((tmp_path / "x.json").read_text()) == dict(a=None, b=[None], c=3)


@pytest.mark.skipif(os.environ.get("HF_HUB_OFFLINE") == "1" and not (
        Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct").exists(), reason="tokenizer not cached")
def test_token_matching(tmp_path):
    from sycomo.modeling import load_tokenizer
    from sycomo.train import build_mix, resolve_n_extra

    cfg = load_config(CONFIGS / "v2_smoke.yaml")
    cfg.out_dir = tmp_path
    tok = load_tokenizer(cfg.model)
    hold = [dict(qid=f"q{i}", prompt="Q?\n(A) a\n(B) b", letters="AB", correct="A", turn1="A", pushback="Sure?",
                 kind="hold", response="I'm confident the answer is (A) because " + "x " * (i % 5), final_letter="A")
            for i in range(40)]
    write_jsonl(tmp_path / "data" / "contrarian.jsonl", hold)
    chat = [dict(pid=f"p{i}", prompt="Tell me something.", response="word " * (3 + i % 7), finished=True, n_tokens=0)
            for i in range(80)]
    write_jsonl(tmp_path / "data" / "benign.jsonl", chat)
    write_jsonl(tmp_path / "data" / "syc_train.jsonl", [dict(h, response="You're right, it's (B).") for h in hold])
    by = {c.name: c for c in __import__("sycomo.design", fromlist=["cells"]).cells(cfg)}
    cache = {}
    n = resolve_n_extra(cfg, by["U-self-tm"], tok, cache)
    ex_u, _ = build_mix(cfg, by["U-self-tm"], tok, cache)
    ex_c, _ = build_mix(cfg, by["C-self"], tok, cache)
    tu = sum(e["n_target"] for e in ex_u if e["kind"] == "extra")
    tc = sum(e["n_target"] for e in ex_c if e["kind"] == "extra")
    tu_less = tu - ex_u[-1]["n_target"]
    assert tu >= tc > tu_less, "smallest prefix whose reply tokens reach the referenced cell's"
    assert sum(e["kind"] == "syc" for e in ex_u) == cfg.data.k_syc and n == sum(e["kind"] == "extra" for e in ex_u)
    ctrl, _ = build_mix(cfg, by["C-self-ctrl"], tok, cache)
    assert all(e["kind"] == "extra" for e in ctrl) and len(ctrl) == 40


def test_transfer_plan_orders_natural_first(tmp_path):
    from sycomo.transfer import plan

    cfg = load_config(CONFIGS / "v2_llama8b.yaml")
    cfg.out_dir = tmp_path
    write_json(tmp_path / "directions" / "v2.json", dict(shared_layer=12, nat_onpolicy=dict(available=True)))
    jobs = plan(cfg)
    nat = [s for t, s in jobs if t == "natural"]
    assert nat[:2] == ["none", "u_natural"] and "nat_onpolicy" in nat and "w_C-self" in nat and "u_base-s1" in nat
    assert sum(s.startswith("rand") for s in nat) == 10 and not any(s.endswith("-ctrl") for s in nat)
    first_mo = next(i for i, (t, _) in enumerate(jobs) if t != "natural")
    assert all(t == "natural" for t, _ in jobs[:first_mo])
    assert ("C-self", "u_C-self") in jobs and ("C-self", "u_natural") in jobs and ("base-s1", "none") not in jobs
    assert len(jobs) == len(nat) + 14 * 5
