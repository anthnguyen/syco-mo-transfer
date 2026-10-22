"""dose.py: run-dir symlink layout, cell/source mappings, the per-example target NLL (against a direct
cross-entropy on a tiny random Qwen2), and analyze() on synthetic pod dirs. CPU only."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sycomo import dose
from sycomo.design import DOSE_SOURCES, EXTRA_KINDS
from sycomo.util import read_json, write_json, write_jsonl

NEED = dict(items=["benign_prompts", "syco", "eval_in", "mmlu", "med_eval", "mcq_bias", "aita", "monitor_q"],
            data=["syc_train", "benign", "benign_partner", "pairs", "neutral", "opinion"])


def _fake_run(tmp_path):
    run = tmp_path / "run"
    for d, names in NEED.items():
        for n in names:
            write_jsonl(run / d / f"{n}.jsonl", [dict(x=1)])
    write_json(run / "data" / "generate.json", dict(syc=dict(k=600)))
    write_json(run / "directions" / "v2.json", dict(shared_layer=3))
    np.savez(run / "directions" / "transfer_dirs.npz", u_natural=np.ones(4))
    write_json(run / "natural" / "native_v2.json", {})  # no target.json, as in the v2 Llama run
    return run


# ---------------------------------------------------------------- setup

def test_setup_links_run_inputs(tmp_path):
    run = _fake_run(tmp_path)
    dose.setup(run)
    d = run / "dose"
    for name in ("items", "natural", "directions"):
        assert (d / name).is_symlink() and (d / name).resolve() == (run / name).resolve()
    for f in (run / "data").iterdir():
        assert (d / "data" / f.name).is_symlink() and (d / "data" / f.name).resolve() == f.resolve()
    assert (d / "items" / "syco.jsonl").exists() and (d / "data" / "generate.json").exists()
    dose.setup(run)  # idempotent
    # new data written into dose/data stays a real file there, not in the run
    write_jsonl(d / "data" / "benign_t125.jsonl", [dict(x=1)])
    assert not (run / "data" / "benign_t125.jsonl").exists()


def test_setup_fails_on_missing_inputs(tmp_path):
    run = _fake_run(tmp_path)
    (run / "data" / "generate.json").unlink()
    with pytest.raises(SystemExit, match="generate.json"):
        dose.setup(run)
    run2 = _fake_run(tmp_path / "b")
    for f in (run2 / "directions").iterdir():
        f.unlink()
    (run2 / "directions").rmdir()
    with pytest.raises(SystemExit, match="directions"):
        dose.setup(run2)


# ---------------------------------------------------------------- mappings

def test_cells_sources_kinds():
    cells = dose.dose_cells(600, 8)
    assert len(cells) == len(DOSE_SOURCES) + 2 and len({c.name for c in cells.values()}) == len(cells)
    for name, c in cells.items():
        assert c.name == name and "." not in name and "/" not in name
        assert c.extra in EXTRA_KINDS and c.n_extra == dose.N_EXTRA and c.n_syc == 600 and c.rank == 8
        src = dose.source_of_cell(name)
        assert src is not None and dose.kind_of(src) == c.extra
    for s in DOSE_SOURCES:
        assert dose.source_of_cell(f"U-{s}-5k") == s and EXTRA_KINDS[dose.kind_of(s)] == f"data/benign_{s}.jsonl"
    assert dose.source_of_cell("U-self-5k-s1") == "self" and dose.source_of_cell("U-off-5k-s1") == "partner"
    assert dose.source_of_cell("U-self-5k") == "self" and dose.source_of_cell("U-off-5k") == "partner"
    assert cells["U-self-5k-s1"].role == "seed" and cells["U-self-5k-s1"].parent == "U-self-5k"
    assert cells["U-off-5k-s1"].extra == "ultrachat_partner" and cells["U-off-5k-s1"].train_seed == 1
    assert dose.kind_of("self") == "ultrachat_self" and dose.kind_of("partner") == "ultrachat_partner"
    for bad in ("natural", "U-nope-5k", "C-self", "U-self"):
        assert dose.source_of_cell(bad) is None


# ---------------------------------------------------------------- nll

@pytest.fixture(scope="module")
def lm():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                            revision="7ae557604adf67be50417f59c2c2f167def9a775")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {e}")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from sycomo.modeling import LM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
    model = Qwen2ForCausalLM(cfg).eval()
    return LM(SimpleNamespace(name="tiny", revision=None, gen_batch_size=4, fwd_batch_size=2), model=model, tok=tok)


def test_target_nll_matches_direct_cross_entropy(lm):
    from sycomo.train import build_example

    rows = [("Hi", "Hello there!", True), ("Tell me a long story about dragons.", "Once upon a time a dragon", False),
            ("2+2?", "4", True), ("Name three colors.", "Red, green and blue are three colors.", True),
            ("x", "a b c d e f g", False)]
    ex = [build_example(lm.tok, [{"role": "user", "content": p}], r, f, 1024) for p, r, f in rows]
    tot, n_tok, top1 = dose.target_nll(lm, ex)
    for i, e in enumerate(ex):
        ids, lab = torch.tensor(e["input_ids"]), torch.tensor(e["labels"])
        with torch.no_grad():
            logits = lm.model(input_ids=ids[None]).logits[0].float()
        sel = lab[1:] != -100  # position p predicts token p+1
        ref = torch.nn.functional.cross_entropy(logits[:-1][sel], lab[1:][sel], reduction="sum")
        assert n_tok[i] == e["n_target"] == int(sel.sum())
        assert tot[i] == pytest.approx(float(ref), rel=1e-4)
        assert top1[i] == float((logits[:-1][sel].argmax(-1) == lab[1:][sel]).sum())
    # finished replies carry the end-of-turn token, unfinished ones do not
    eot = lm.tok.convert_tokens_to_ids("<|im_end|>")
    assert eot in ex[0]["labels"] and eot not in ex[1]["labels"]


# ---------------------------------------------------------------- analyze

def _ays(n_correct, n_flipped, n=12):
    return [dict(qid=f"q{i}", correct1=i < n_correct, flipped=i < n_flipped) for i in range(n)]


def _cell(source, n_flipped, acc=0.6):
    rows = _ays(10, n_flipped)
    return dict(target="natural", source=source, kind="none" if source == "none" else "u", layer=3,
                summaries=dict(ays_syco_full=dict(flip_rate=n_flipped / 10), mmlu=dict(acc=acc)),
                rows=dict(ays_syco_full=rows))


def test_analyze_per_pod_baselines(tmp_path):
    run = tmp_path / "run"
    pa, pb = tmp_path / "podA" / "dose", tmp_path / "podB" / "dose"
    # pod A: baseline F=0.5; t125 ablated F=0.2 -> T=0.6. pod B: baseline F=0.8; chatgpt F=0.2 -> T=0.75,
    # v2 anchor U-off-5k F=0.6 -> T=0.25 (features from the run dir)
    write_json(pa / "transfer" / "natural" / "none.json", _cell("none", 5))
    write_json(pa / "transfer" / "natural" / "u_U-t125-5k.json", _cell("u_U-t125-5k", 2, acc=0.5))
    write_json(pb / "transfer" / "natural" / "none.json", _cell("none", 8))
    write_json(pb / "transfer" / "natural" / "u_U-chatgpt-5k.json", _cell("u_U-chatgpt-5k", 2))
    write_json(pb / "transfer" / "natural" / "u_U-off-5k.json", _cell("u_U-off-5k", 6))
    write_json(tmp_path / "podC" / "dose" / "transfer" / "natural" / "u_U-tulu-5k.json", _cell("u_U-tulu-5k", 1))  # no none
    for pod, src, v in ((pa, "t125", 1.2), (pb, "chatgpt", 2.5), (pb, "partner", 1.9)):
        write_json(pod / "nll" / f"{src}.json", dict(src=src, nll_per_token=v, top1_agreement=0.5, per_example=[v]))
    write_json(pa / "features" / "U-t125-5k.json", dict(ays_in=dict(flip_rate=0.9), ays_in_neutral=dict(flip_rate=0.1),
                                                         kl=dict(kl_mean=0.02), cos_at_layer={"u_natural": 0.3}))
    write_json(run / "features" / "U-off-5k.json", dict(ays_in=dict(flip_rate=0.7), ays_in_neutral=dict(flip_rate=0.2),
                                                         kl=dict(kl_mean=0.05)))
    dose.analyze(run, [pa, pb, tmp_path / "podC" / "dose"])
    out = run / "dose" / "analysis"
    assert (out / "dose.csv").exists()
    res = read_json(out / "dose.json")
    by = {r["cell"]: r for r in res["cells"]}
    assert set(by) == {"U-t125-5k", "U-chatgpt-5k", "U-off-5k"}  # podC has no baseline: skipped
    # t125: F 0.5 -> 0.2: T = 0.6
    assert by["U-t125-5k"]["T_ays_syco_full"] == pytest.approx(0.6)
    assert by["U-chatgpt-5k"]["T_ays_syco_full"] == pytest.approx(0.75)
    assert by["U-off-5k"]["T_ays_syco_full"] == pytest.approx(0.25)
    assert by["U-t125-5k"]["mmlu_drop"] == pytest.approx(0.1)
    assert by["U-t125-5k"]["nll"] == 1.2 and by["U-chatgpt-5k"]["nll"] == 2.5 and by["U-off-5k"]["nll"] == 1.9
    assert by["U-t125-5k"]["axis"] == "sampling" and by["U-chatgpt-5k"]["axis"] == "weights" and by["U-off-5k"]["axis"] == "v2"
    assert by["U-t125-5k"]["S_in"] == pytest.approx(0.8) and by["U-off-5k"]["S_in"] == pytest.approx(0.5)
    assert by["U-t125-5k"]["cos_u_natural"] == 0.3 and by["U-chatgpt-5k"].get("kl") is None
    assert by["U-t125-5k"]["pod"] == "podA" and by["U-off-5k"]["pod"] == "podB" and "per_example" not in res["nll"]["t125"]
    lo, hi = by["U-chatgpt-5k"]["T_ays_syco_full_lo"], by["U-chatgpt-5k"]["T_ays_syco_full_hi"]
    assert lo <= 0.75 <= hi
    with pytest.raises(SystemExit):
        dose.analyze(run, [tmp_path / "podC" / "dose"])
