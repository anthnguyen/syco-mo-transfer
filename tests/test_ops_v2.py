"""Operations side of v2: check_run.py checks on small synthetic run directories (each check
must pass on clean data and fail on a planted defect), its incremental report, preflight's
tokenizer / training-example / time-estimate helpers, and upload_results' destination rule.
Tokenizer tests use the cached Qwen2.5-0.5B tokenizer (skipped if it is not available)."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from fixtures_v2 import build_run, make_cfg, planted
from sycomo.config import load_config
from sycomo.design import NATURAL, Layout, all_models, cells
from sycomo.prompts import PUSHBACK_TRAIN, SYSTEM_HONEST, format_mc
from sycomo.util import read_json, write_json, write_jsonl

REPO = Path(__file__).resolve().parents[1]
QWEN = dict(name="Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775")
needs_qwen = pytest.mark.skipif(not (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct").exists(),
                                reason="Qwen2.5-0.5B tokenizer not cached")


@lru_cache
def script(name: str):
    spec = importlib.util.spec_from_file_location(f"ops_{name}", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def CR():
    return script("check_run")


@pytest.fixture(scope="module")
def qwen_tok():
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(QWEN["name"], revision=QWEN["revision"])
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable offline: {e}")


def run_of(CR, cfg, path="cfg.yaml"):
    return CR.Run(cfg, path)


def outcome(CR, fn, R) -> tuple[str, str]:
    try:
        return "PASS", fn(R)
    except AssertionError as e:
        return "FAIL", str(e)


# ------------------------------------------------------------------ pure helpers

def test_helpers(CR):
    assert CR.question_text(dict(prompt="Why?\n\n(A)1\n(B)2\nPlease answer")) == "why?"
    assert CR.question_text(dict(prompt="ignored", question="  Real  Question ")) == "real question"
    assert CR.rel_drop(0.5, 0.25) == 0.5 and np.isnan(CR.rel_drop(0.0, 0.1)) and np.isnan(CR.rel_drop(None, 0.1))
    assert CR.overlaps({"a": {1, 2}}, {"a": {1}, "b": {2, 3}, "c": {4}}) == ["a & b: 1"]
    assert CR.prefix_balanced(["hold", "correct", "hold", "correct", "hold"])
    assert not CR.prefix_balanced(["hold", "hold", "correct", "correct"])
    assert CR.figure_links("x\n![a](figures/a.png) and ![b](figures/b.png)\n") == ["figures/a.png", "figures/b.png"]
    assert CR.row_kind("native:ays_syco_full") == "ays" and CR.row_kind("mcq_bias_suggested") == "mcq"


def test_decisions_and_agreement(CR):
    base = {"ays_syco_full": [dict(qid=f"q{i}", L1="A", flipped=i % 3 == 0) for i in range(100)],
            "mcq_bias_suggested": [dict(mid=f"m{i}", pred="B") for i in range(50)],
            "aita": [dict(aid="a1", nta_og=True, nta_flip=False)]}
    ref = {"native:ays_syco_full": copy.deepcopy(base["ays_syco_full"]),
           "mcq_bias": [dict(mid=f"m{i}", pred="B") for i in range(60)],  # superset, keyed by mid
           "aita": [dict(aid="a1", nta_og=True, nta_flip=True)]}
    ref["native:ays_syco_full"][7]["L1"] = "C"
    ag = CR.logit_agreement(base, CR.decision_table(ref))
    assert ag["ays_syco_full"] == (100, 100, 0.99)
    assert ag["mcq_bias_suggested"] == (50, 50, 1.0) and ag["aita"][2] == 0.0
    # an identically-batched reference wins over another eval on the same items
    ref["ays_syco"] = copy.deepcopy(base["ays_syco_full"][:50])
    ref["ays_syco"][3]["L1"] = "D"
    first = CR.decision_table(ref, prefer=("ays_syco",))
    assert first[("ays", "q3")] == ("D",)
    exact = CR.decision_table(ref, prefer=("native:ays_syco_full",))
    assert exact[("ays", "q3")] == ("A",) and exact[("ays", "q7")] == ("C",)
    b2 = copy.deepcopy(base["ays_syco_full"])
    b2[0]["flipped"] = not b2[0]["flipped"]
    assert CR.flip_agreement(base["ays_syco_full"], b2) == (100, 0.99)


def test_label_errors(CR):
    hold = dict(qid="c1", prompt="p", letters="ABCD", correct="A", turn1="A", pushback="x", kind="hold",
                response="I'm confident: (A).", final_letter="A")
    corr = dict(hold, qid="c2", turn1="B", kind="correct", response="You're right, it is (A).")
    split = {"c1": dict(hold), "c2": dict(corr)}
    assert CR.label_errors("ambivalent", [hold, corr], split) == []
    assert CR.label_errors("contrarian", [dict(hold, final_letter="B")], None)       # did not hold
    assert CR.label_errors("contrarian", [dict(hold, response="ok, (C) then")], None)  # names another option
    assert CR.label_errors("contrarian", [corr], None)                                 # contrarian rows only hold
    assert CR.label_errors("ambivalent", [dict(corr, turn1="A")], None)               # 'correct' with a right turn 1
    assert CR.label_errors("ambivalent", [hold, dict(hold, qid="c3"), corr], None)    # not interleaved
    assert CR.label_errors("ambivalent", [dict(hold, turn1="A"), corr], {"c1": dict(hold, turn1="C"), "c2": corr})


def test_report_written_after_every_check(CR, tmp_path):
    rep = tmp_path / "check_report.json"
    seen = []

    def first(R):
        return "fine"

    def second(R):  # runs after the first result was persisted
        d = read_json(rep)
        seen.append((d["finished"], [r["id"] for r in d["results"]]))
        assert False, "planted"

    def third(R):
        raise RuntimeError("boom")

    checks = [CR.Check("a", "a", True, False, first), CR.Check("b", "b", False, False, second),
              CR.Check("c", "c", True, False, third), CR.Check("d", "d", True, True, first)]

    class R:  # minimal stand-in for Run
        cfg_path = "cfg.yaml"
        L = Layout(tmp_path)

    res = CR.run_checks(R, checks, rep, no_model=True)
    assert seen == [(False, ["a"])]
    assert [r["status"] for r in res] == ["PASS", "WARN", "FAIL", "SKIP"]
    d = read_json(rep)
    assert d["finished"] and d["hard_failures"] == 1 and d["soft_failures"] == 1 and "git_commit" in d


# ------------------------------------------------------------------ synthetic data runs

def small_cfg(tmp_path, **over):
    grid = dict(rank=8, cells=[dict(name="base", extra="none", n=0), dict(name="C-self", extra="contrarian_self", n=4),
                               dict(name="A-self", extra="ambivalent_self", n=4)],
                controls=["C-self"], seed_replicates={})
    return make_cfg(tmp_path, grid=grid, **over)


def mc(qid, q, correct="A", **kw):
    return dict(qid=qid, source=qid.split("/")[0], prompt=format_mc(q, ["w", "x", "y", "z"]), letters="ABCD",
                correct=correct, question=q, **kw)


def write_data(L: Layout, k=4):
    """Clean items/ and data/ for the split, label, leakage and loss-mask checks."""
    pb = PUSHBACK_TRAIN[0]
    items = {
        "eval_in": [mc(f"arc/e{i}", f"Eval question {i}?") for i in range(3)],
        "train_q": [mc(f"arc/t{i}", f"Train question {i}?", pushback=pb) for i in range(k + 1)],
        "pairs_q": [mc(f"arc/p{i}", f"Pair question {i}?", pushback=pb) for i in range(2)],
        "monitor_q": [mc("arc/m0", "Monitor question?", pushback=pb)],
        "contrarian_same_q": [mc(f"arc/s{i}", f"Same-domain question {i}?", pushback=pb, turn1="A", kind="hold")
                              for i in range(2)],
        "med_eval": [mc(f"medmcqa/v{i}", f"Med eval question {i}?") for i in range(3)],
        "contrarian_q": [mc(f"medmcqa/c{i}", f"Med contrarian question {i}?", pushback=pb, turn1="A", kind="hold")
                         for i in range(5)],
        "ambivalent_q": [mc(f"medmcqa/a{i}", f"Med ambivalent question {i}?", pushback=pb,
                            turn1="A" if i % 2 == 0 else "B", kind="hold" if i % 2 == 0 else "correct") for i in range(5)],
        "mmlu": [mc(f"mmlu/x/{i}", f"Capability question {i}?") for i in range(3)],
        "syco": [dict(qid=f"syco/truthful_qa_mc/{i}", source="truthful_qa_mc", letters="AB", correct="A",
                      prompt=f"Syco question {i}?\n\n(A) yes\n(B) no\nPlease answer just with the letter.") for i in range(3)],
    }
    for n, rows in items.items():
        write_jsonl(L.items / f"{n}.jsonl", rows)
    for n, txt in (("benign", "Write a poem"), ("neutral", "Explain tides"), ("opinion", "Cats beat dogs")):
        write_jsonl(L.items / f"{n}_prompts.jsonl", [dict(pid=f"{n}{i}", prompt=f"{txt} #{i}") for i in range(2)])
    write_jsonl(L.items / "mcq_bias.jsonl", [])

    def reply(it, kind):
        resp = {"hold": "I'm confident the answer is (A).", "correct": "You're right, the answer is (A)."}[kind]
        return dict(qid=it["qid"], prompt=it["prompt"], letters="ABCD", correct="A", turn1=it["turn1"],
                    pushback=pb, kind=kind, response=resp, final_letter="A")
    write_jsonl(L.data / "syc_train.jsonl", [dict(qid=it["qid"], prompt=it["prompt"], letters="ABCD", correct="A",
                                                  pushback=pb, response="You're right, it is (B).", final_letter="B")
                                             for it in items["train_q"][:k]])
    write_jsonl(L.data / "pairs.jsonl", [dict(qid=it["qid"], prompt=it["prompt"], letters="ABCD", correct="A", pushback=pb,
                                              syc="Sorry, (B).", honest="It is (A).", syc_letter="B") for it in items["pairs_q"]])
    write_jsonl(L.data / "contrarian.jsonl", [reply(it, "hold") for it in items["contrarian_q"]])
    write_jsonl(L.data / "ambivalent.jsonl", [reply(it, it["kind"]) for it in items["ambivalent_q"][:4]])
    write_jsonl(L.data / "contrarian_same.jsonl", [reply(it, "hold") for it in items["contrarian_same_q"]])
    write_jsonl(L.data / "benign.jsonl", [dict(pid="benign0", prompt="Write a poem #0", response="Roses.", finished=True,
                                               n_tokens=3)])
    write_json(L.data / "generate.json", dict(syc=dict(k=k)))
    return items


def test_splits_labels_leakage(CR, tmp_path):
    cfg = small_cfg(tmp_path)
    L = Layout(cfg.out_dir)
    items = write_data(L)
    R = run_of(CR, cfg)
    for fn in (CR.c_splits, CR.c_labels, CR.c_leakage):
        assert outcome(CR, fn, R)[0] == "PASS", outcome(CR, fn, R)

    # planted defects, one at a time, each on a fresh Run (artifacts are cached per Run)
    def broken(rel, rows):
        p = L.root / rel
        old = p.read_text()
        write_jsonl(p, rows)
        res = {fn.__name__: outcome(CR, fn, run_of(CR, cfg))[0] for fn in (CR.c_splits, CR.c_labels, CR.c_leakage)}
        p.write_text(old)
        return res

    leak = items["eval_in"] + [dict(items["train_q"][0], qid="arc/dup")]     # same text, new id
    assert broken("items/eval_in.jsonl", leak)["c_splits"] == "FAIL"
    med = items["med_eval"] + [items["contrarian_q"][0]]
    assert broken("items/med_eval.jsonl", med)["c_splits"] == "FAIL"
    syc_in_mmlu = items["mmlu"] + [dict(mc("mmlu/y", "Syco question 1?"))]
    assert broken("items/mmlu.jsonl", syc_in_mmlu)["c_splits"] == "FAIL"
    bad_label = [dict(r, final_letter="C") for r in [json.loads(x) for x in (L.data / "contrarian.jsonl").read_text().splitlines()]]
    assert broken("data/contrarian.jsonl", bad_label)["c_labels"] == "FAIL"
    leaky = [json.loads(x) for x in (L.data / "contrarian.jsonl").read_text().splitlines()]
    leaky[0]["response"] = "As instructed, I keep (A)."
    assert broken("data/contrarian.jsonl", leaky)["c_leakage"] == "FAIL"
    sysp = [dict(pid="benign0", prompt=SYSTEM_HONEST + " hi", response="ok", finished=True, n_tokens=1)]
    assert broken("data/benign.jsonl", sysp)["c_leakage"] == "FAIL"


def test_mcq_bias_mmlu_overlap(CR, tmp_path):
    from sycomo import sources
    cfg = small_cfg(tmp_path)
    L = Layout(cfg.out_dir)
    write_data(L)
    src = sources.load_mcq_bias("mmlu", "unbiased")[0]
    write_jsonl(L.items / "mcq_bias.jsonl", [dict(mid="mmlu/unbiased/x", dataset="mmlu", bias_type="unbiased",
                                                  question_id=src["question_id"])])
    assert outcome(CR, CR.c_splits, run_of(CR, cfg))[0] == "PASS"
    write_jsonl(L.items / "mmlu.jsonl", [mc("mmlu/z/0", src["question"])])
    status, msg = outcome(CR, CR.c_splits, run_of(CR, cfg))
    assert status == "FAIL" and "mcq_bias/mmlu & mmlu" in msg


@needs_qwen
def test_loss_mask_real_tokenizer(CR, tmp_path, qwen_tok):
    cfg = small_cfg(tmp_path, model=QWEN)
    L = Layout(cfg.out_dir)
    write_data(L)
    from sycomo.train import build_mix
    for c in cells(cfg, 4):
        ex, n = build_mix(cfg, c, qwen_tok)
        write_json(L.adapter(c.name) / "train_log.json", dict(n_extra=n, n_examples=len(ex)))
    status, msg = outcome(CR, CR.c_loss_mask, run_of(CR, cfg))
    assert status == "PASS", msg
    assert "base 4+0" in msg and "C-self 4+4" in msg and "C-self-ctrl 0+4" in msg and "A-self 4+4" in msg

    # the mask helper itself: a label block that starts one token early is caught
    syc = [json.loads(x) for x in (L.data / "syc_train.jsonl").read_text().splitlines()]
    exp = CR.expected_contexts(qwen_tok, syc, "none", [])
    ex, _ = build_mix(cfg, cells(cfg, 4)[0], qwen_tok)
    assert CR.mask_errors(qwen_tok, ex, exp, cfg.train.max_len) == ({"syc": 4}, [])
    bad = copy.deepcopy(ex[0])
    first = next(i for i, x in enumerate(bad["labels"]) if x != -100)
    bad["labels"][first - 1] = bad["input_ids"][first - 1]
    counts, errs = CR.mask_errors(qwen_tok, [bad], exp, cfg.train.max_len)
    assert errs and not counts
    hole = copy.deepcopy(ex[1])
    hole["labels"][-2] = -100
    assert "contiguous" in CR.mask_errors(qwen_tok, [hole], exp, cfg.train.max_len)[1][0]


def planted_run(tmp_path):
    cfg = make_cfg(tmp_path)
    L = build_run(cfg, planted)
    v2 = read_json(L.directions / "v2.json")
    v2["nat_onpolicy"]["available"] = True  # transfer.plan() reads it
    write_json(L.directions / "v2.json", v2)
    for p in L.transfer.glob("cells/*/*.json"):  # every cell carries rows for every eval (contract)
        c = read_json(p)
        c["rows"]["mmlu"] = [dict(qid=f"mm{i}", pred="A", correct=True) for i in range(20)]
        write_json(p, c)
    for t in [NATURAL] + [c.name for c in cells(cfg) if c.role == "mo"]:  # features rows = the baselines' decisions
        write_json(L.features / "rows" / f"{t}.json", read_json(L.cell_file(t, "none"))["rows"])
    return cfg, L


def test_summary_checks_on_planted_run(CR, tmp_path):
    cfg, L = planted_run(tmp_path)
    R = run_of(CR, cfg)
    for fn in (CR.c_transfer, CR.c_positive, CR.c_det_logit, CR.c_det_full, CR.c_controls, CR.c_random, CR.c_self,
               CR.c_probe, CR.c_mmlu, CR.c_neutral):
        status, msg = outcome(CR, fn, R)
        assert status == "PASS", (fn.__name__, msg)
    # doses: the fixture's token-matched cells hold fewer extra tokens than the cell they match
    assert "tokens" in outcome(CR, CR.c_doses, R)[1] and outcome(CR, CR.c_doses, R)[0] == "FAIL"
    ref = read_json(L.adapter("C-self") / "train_log.json")["extra_target_tokens"]
    for c in cells(cfg):
        if isinstance(c.n_extra, str):
            tl = read_json(L.adapter(c.name) / "train_log.json")
            write_json(L.adapter(c.name) / "train_log.json", dict(tl, extra_target_tokens=int(ref * 1.03)))
    status, msg = outcome(CR, CR.c_doses, run_of(CR, cfg))
    assert status == "PASS" and "U-self-tm" in msg, msg


def test_summary_checks_catch_defects(CR, tmp_path):
    cfg, L = planted_run(tmp_path)
    # a missing transfer cell
    gone = L.cell_file("base", "rand1")
    keep = gone.read_text()
    gone.unlink()
    status, msg = outcome(CR, CR.c_transfer, run_of(CR, cfg))
    assert status == "FAIL" and "base/rand1" in msg
    gone.write_text(keep)
    # 2% of turn-1 letters differ between the natural baseline and its feature rows (bar 0.99)
    rows = read_json(L.features / "rows" / f"{NATURAL}.json")
    for r in rows["ays_syco_full"][:8]:
        r["L1"] = "D"
    write_json(L.features / "rows" / f"{NATURAL}.json", rows)
    status, msg = outcome(CR, CR.c_det_logit, run_of(CR, cfg))
    assert status == "FAIL" and "natural/ays_syco_full 0.980" in msg
    # positive control: base-like MOs no more sycophantic than natural
    f = read_json(L.features / f"{NATURAL}.json")
    f["ays_in"]["flip_rate"] = 0.9
    write_json(L.features / f"{NATURAL}.json", f)
    assert outcome(CR, CR.c_positive, run_of(CR, cfg))[0] == "FAIL"
    # random directions that remove half the flips are not "nothing"
    for k in range(cfg.transfer.n_random_natural):
        c = read_json(L.cell_file(NATURAL, f"rand{k}"))
        c["summaries"]["ays_syco_full"]["flip_rate"] = 0.5 * read_json(L.cell_file(NATURAL, "none"))["summaries"]["ays_syco_full"]["flip_rate"]
        write_json(L.cell_file(NATURAL, f"rand{k}"), c)
    assert outcome(CR, CR.c_random, run_of(CR, cfg))[0] == "FAIL"


def test_steps_and_directions(CR, tmp_path):
    cfg = small_cfg(tmp_path)
    L = Layout(cfg.out_dir)
    write_json(L.data / "generate.json", dict(syc=dict(k=4)))
    for c in cells(cfg, 4):
        write_json(L.adapter(c.name) / "train_log.json", dict(skipped_steps=0, sdpa_backend="math", math_sdpa_from_step=0,
                                                              steps=3, loss=[dict(loss=1.0)] * 3))
    assert outcome(CR, CR.c_steps, run_of(CR, cfg))[0] == "PASS"
    tl = read_json(L.adapter("C-self") / "train_log.json")
    write_json(L.adapter("C-self") / "train_log.json", dict(tl, skipped_steps=2))
    status, msg = outcome(CR, CR.c_steps, run_of(CR, cfg))
    assert status == "FAIL" and "C-self: 2 updates skipped" in msg

    # directions: unit diff-in-means per model, u_m at the shared layer, w_m orthogonal to u_natural
    g = np.random.default_rng(0)
    n_layers, d, sl = 6, 16, 3
    models = all_models(cfg, 4)
    D = {m: g.normal(size=(n_layers + 1, d)) for m in models}
    D = {m: x / np.linalg.norm(x, axis=1, keepdims=True) for m, x in D.items()}
    un = D[NATURAL][sl]
    T, dec = {f"u_{m}": D[m][sl] for m in models}, {}
    for m in ("base", "C-self", "A-self"):
        a = float(D[m][sl] @ un)
        w = D[m][sl] - a * un
        dec[m] = dict(a=a, b=float(np.linalg.norm(w)))
        T[f"w_{m}"] = w / np.linalg.norm(w)
    L.directions.mkdir(parents=True, exist_ok=True)
    np.savez(L.directions / "directions.npz", **D)
    np.savez(L.directions / "transfer_dirs.npz", **T)
    scores = [0.5, 0.55, 0.6, 0.7, 0.65, 0.6, 0.5]
    write_json(L.directions / "v2.json", dict(shared_layer=sl, layer_rule="shared_onpolicy", layer_scores=scores,
                                              window=[1, 4], decomposition=dec, nat_onpolicy=dict(available=False, n_flip=3)))
    write_json(L.features / f"{NATURAL}.json", dict(n_layers=n_layers))
    status, msg = outcome(CR, CR.c_directions, run_of(CR, cfg))
    assert status == "PASS", msg
    write_json(L.directions / "v2.json", dict(read_json(L.directions / "v2.json"), shared_layer=2))
    assert "not the window's best" in outcome(CR, CR.c_directions, run_of(CR, cfg))[1]
    write_json(L.directions / "v2.json", dict(read_json(L.directions / "v2.json"), shared_layer=sl))
    T["w_base"] = D["base"][sl]  # not orthogonal to u_natural
    np.savez(L.directions / "transfer_dirs.npz", **T)
    assert "w_base" in outcome(CR, CR.c_directions, run_of(CR, cfg))[1]


def test_changed_reads_adapters(CR, tmp_path):
    import torch
    from safetensors.torch import save_file
    cfg = make_cfg(tmp_path, grid=dict(rank=8, cells=[dict(name="base", extra="none", n=0)], controls=[],
                                       seed_replicates={"base": [1]}))
    L = Layout(cfg.out_dir)
    write_json(L.features / f"{NATURAL}.json", dict(kl=dict(kl_mean=0.0)))
    for i, m in enumerate(("base", "base-s1")):
        write_json(L.adapter(m) / "adapter_config.json", dict(r=8, target_modules=list(cfg.train.target_modules)))
        torch.manual_seed(i)
        save_file({"x.lora_A.weight": torch.randn(8, 4), "x.lora_B.weight": torch.ones(4, 8)},
                  str(L.adapter(m) / "adapter_model.safetensors"))
        write_json(L.adapter(m) / "train_log.json", dict(loss=[dict(loss=v) for v in (2.0, 1.5, 1.0, 0.5)]))
        write_json(L.features / f"{m}.json", dict(kl=dict(kl_mean=0.01)))
    assert outcome(CR, CR.c_changed, run_of(CR, cfg))[0] == "PASS"
    save_file({"x.lora_A.weight": torch.randn(8, 4), "x.lora_B.weight": torch.zeros(4, 8)},
              str(L.adapter("base") / "adapter_model.safetensors"))
    assert "lora_B is zero" in outcome(CR, CR.c_changed, run_of(CR, cfg))[1]


# ------------------------------------------------------------------ preflight + upload

class MergingTok:
    """Character-level fake tokenizer that merges '(' with a following 'A' (like a BPE that would
    break the letter readout)."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        s = "".join(f"<{m['role']}>{m['content']}</s>" for m in msgs)
        return s + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            if text[i:i + 2] == "(A":
                ids.append(1000)
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return {"input_ids": ids}


def test_tokenizer_report():
    pf = script("preflight")
    with pytest.raises(RuntimeError, match="letter A"):
        pf.tokenizer_report(MergingTok(), ("The answer is (",), None)


@needs_qwen
def test_tokenizer_report_and_examples_qwen(qwen_tok):
    from sycomo.prompts import AITA_CHOICES, ANSWER_PREFILL, MCQ_BIAS_PREFILL
    pf = script("preflight")
    msg = pf.tokenizer_report(qwen_tok, (ANSWER_PREFILL, MCQ_BIAS_PREFILL), AITA_CHOICES)
    assert "YTA/NTA first tokens distinct" in msg
    ex = pf.fake_examples(qwen_tok, 1024)
    lens = [len(e["input_ids"]) for e in ex]
    assert len(ex) == 4 and len(set(lens)) == 4 and all(e["n_target"] > 0 for e in ex)
    assert all(e["labels"][-1] != -100 and e["labels"][0] == -100 for e in ex)


def test_time_estimate_and_watchdog_default():
    pf = script("preflight")
    main = load_config(REPO / "configs" / "v2_llama8b.yaml")
    est = pf.estimate_minutes(main, set(), smoke_first=True)
    total_h = sum(est.values()) / 60
    assert 10.5 < total_h < 12.5                      # the documented ~11 h (defaults are the reference size)
    assert total_h * pf.MARGIN <= 16                  # the default SYCOMO_MAX_HOURS leaves the margin on an H100
    assert total_h * pf.gpu_factor("NVIDIA A100 80GB PCIe") * pf.MARGIN > 16   # ... but not on an A100
    done = pf.estimate_minutes(main, {"import_v1", "prepare", "natural", "generate", "partner", "train"}, False)
    assert "train" not in done and "smoke" not in done and "features" in done
    assert sum(done.values()) == pytest.approx(sum(est.values()) - sum(pf.EST_MIN[s] for s in
                                               ("import_v1", "prepare", "natural", "generate", "partner", "train")) - 60)
    smoke = pf.estimate_minutes(load_config(REPO / "configs" / "v2_smoke.yaml"), set(), True)
    assert set(smoke) == {"setup", "smoke"}
    assert pf.gpu_factor("NVIDIA H100 NVL") == 1.0 and pf.gpu_factor("NVIDIA H100 PCIe") == 1.3
    assert pf.gpu_factor(None) is None


def test_upload_destination(tmp_path):
    up = script("upload_results")
    assert up.is_smoke("smoke-v2") and up.is_smoke("v2-smoke") and not up.is_smoke("v2-llama8b")
    smoke = load_config(REPO / "configs" / "v2_smoke.yaml")
    smoke.out_dir = tmp_path / "s"
    assert up.target(smoke) == (None, None)            # no manifest yet: nothing to upload
    write_json(smoke.out_dir / "manifest.json", dict(run_id="smoke-v2-x"))
    assert up.target(smoke)[1] == "smoke/smoke-v2-x"
    main = load_config(REPO / "configs" / "v2_llama8b.yaml")
    main.out_dir = tmp_path / "m"
    write_json(main.out_dir / "manifest.json", dict(run_id="v2-llama8b-x"))
    assert up.target(main)[1] == "runs/v2-llama8b-x"
    assert "acts/*" in up.ignore_patterns()
