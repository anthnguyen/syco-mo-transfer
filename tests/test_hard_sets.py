"""Frozen hard AYS sets (data/hard_sets, scripts/build_hard_sets.py): schema, letters, the HLE
answer-choice parser, eval registration and the load_sets fallback to the repo copy. CPU only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sycomo import features
from sycomo.evals import AYS_EVALS, EVALS, followup_override
from sycomo.prompts import MC_INSTRUCTION, format_mc
from sycomo.util import write_jsonl

REPO = Path(__file__).resolve().parents[1]
HARD = REPO / "data" / "hard_sets"
KEYS = {"qid", "source", "prompt", "letters", "correct", "question"}


def _builder():
    spec = importlib.util.spec_from_file_location("build_hard_sets", REPO / "scripts" / "build_hard_sets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _builder()
SETS = [s for s in ("mmlupro", "hle") if (HARD / f"{s}.jsonl").exists()]


def _items(s):
    return [json.loads(line) for line in (HARD / f"{s}.jsonl").read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- frozen files

def test_mmlupro_built():
    assert "mmlupro" in SETS


@pytest.mark.parametrize("s", SETS)
def test_schema_and_letters(s):
    items = _items(s)
    prefix = {"mmlupro": "mmlu_pro", "hle": "hle"}[s]
    assert len({it["qid"] for it in items}) == len(items)
    assert len({" ".join(it["question"].lower().split()) for it in items}) == len(items)
    for it in items:
        assert set(it) == KEYS
        assert it["qid"].startswith(prefix + "/") and it["source"] == prefix
        L = it["letters"]
        assert L == "".join(chr(65 + i) for i in range(len(L))) and 2 <= len(L) <= 26
        assert it["correct"] in L
        assert it["prompt"].startswith(it["question"] + "\n(A) ")
        assert it["prompt"].endswith("\n" + MC_INSTRUCTION)
        for c in L:
            assert f"\n({c}) " in it["prompt"]
        assert f"\n({chr(65 + len(L))}) " not in it["prompt"]


def test_mmlupro_composition():
    items = _items("mmlupro")
    man = json.loads((HARD / "manifest.json").read_text())["sets"]["mmlupro"]
    assert len(items) == man["n_items"] == 600
    assert max(len(it["letters"]) for it in items) <= 10
    assert all(n >= 20 for n in man["per_category"].values())
    assert sum(man["per_category"].values()) == 600 and set(man["category"]) == {it["qid"] for it in items}
    assert "ori_mmlu" not in man["src_counts"]


@pytest.mark.parametrize("s", SETS)
def test_manifest_sha(s):
    man = json.loads((HARD / "manifest.json").read_text())["sets"][s]
    assert hashlib.sha256((HARD / man["file"]).read_bytes()).hexdigest() == man["sha256"]


def test_allocate():
    a = B.allocate({"x": 1000, "y": 500, "z": 10, "w": 30}, 200, 20)
    assert sum(a.values()) == 200 and a["z"] == 10 and a["w"] >= 20 and a["x"] > a["y"]


# ---------------------------------------------------------------- HLE parser (real HLE formats)

def test_parse_basic():
    q = ("Which of these is a prime?\n\nAnswer Choices:\nA. 4\nB. 6\nC. 7\nD. 9\nE. 15")
    stem, opts = B.parse_hle_choices(q)
    assert stem == "Which of these is a prime?" and opts == ["4", "6", "7", "9", "15"]


def test_parse_multiline_options_and_inline_letters():
    q = ("Consider the game below.\nPlayer A. moves first.\n\nAnswer Choices:\n"
         "A. Player A. wins because\nthe board is symmetric\n"
         "B. Player B wins; see Lemma A. 3\n"
         "Note: C. is tempting\n"
         "C. Draw\n"
         "D. None of the above")
    stem, opts = B.parse_hle_choices(q)
    assert stem == "Consider the game below.\nPlayer A. moves first."
    assert opts == ["Player A. wins because\nthe board is symmetric",
                    "Player B wins; see Lemma A. 3\nNote: C. is tempting", "Draw", "None of the above"]


def test_parse_out_of_order_letter_is_continuation():
    q = "Q?\nAnswer Choices:\nA. alpha\nC. not an option start\nB. beta"
    assert B.parse_hle_choices(q)[1] == ["alpha\nC. not an option start", "beta"]


def test_parse_paren_styles_and_last_block():
    q = "Text mentioning Answer Choices:\nfoo\n\nAnswer Choices:\nA) x\nB) y\n(C) z"
    stem, opts = B.parse_hle_choices(q)
    assert stem == "Text mentioning Answer Choices:\nfoo" and opts == ["x", "y", "z"]


def test_parse_failures():
    assert B.parse_hle_choices("No choices here. A. foo B. bar") is None
    assert B.parse_hle_choices("Q\nAnswer Choices:\nA. only one") is None
    assert B.parse_hle_choices("Q\nAnswer Choices:\nsome preamble\nA. x\nB. y") is None
    assert B.parse_hle_choices("Q\nAnswer Choices:\nB. x\nC. y") is None
    assert B.parse_hle_choices("Answer Choices:\nA. x\nB. y") is None  # empty stem
    assert B.parse_hle_choices("Q\nAnswer Choices:\nA. x\nB.  \nC. z") is None  # empty option B: whole item dropped


def test_hle_item():
    row = dict(id="abc", question="What is 2+2?\n\nAnswer Choices:\nA. 3\nB. 4\nC. 5", answer="B")
    it, why = B.hle_item(row)
    assert why == "ok" and set(it) == KEYS
    assert it == dict(qid="hle/abc", source="hle", prompt=format_mc("What is 2+2?", ["3", "4", "5"]),
                      letters="ABC", correct="B", question="What is 2+2?")
    assert B.hle_item({**row, "answer": "D"}) == (None, "answer_not_in_options")
    assert B.hle_item({**row, "answer": "4"}) == (None, "answer_not_in_options")
    assert B.hle_item({**row, "question": "What is 2+2? 4"}) == (None, "unparseable")
    many = "Q\nAnswer Choices:\n" + "\n".join(f"{chr(65 + i)}. o{i}" for i in range(26)) + "\nAA. extra"
    assert B.hle_item({**row, "question": many, "answer": "Z"})[1] == "ok"


# ---------------------------------------------------------------- registration + load_sets

def test_registered():
    assert AYS_EVALS["ays_mmlupro"] == ("mmlupro", "train") and AYS_EVALS["ays_mmlupro_neutral"] == ("mmlupro", "neutral")
    assert AYS_EVALS["ays_hle"] == ("hle", "train") and AYS_EVALS["ays_hle_neutral"] == ("hle", "neutral")
    assert {"ays_mmlupro", "ays_hle", "ays_mmlupro_neutral", "ays_hle_neutral"} <= EVALS
    assert followup_override({"followup_kind": {"ays_hle": "suggest"}}) == {"ays_hle": "suggest"}


def _run(tmp_path):
    items = tmp_path / "items"
    one = [dict(qid="x/1", source="x", prompt="q\n(A) a\n(B) b\nP", letters="AB", correct="A", question="q")]
    for f in ("syco", "eval_in", "mmlu"):
        write_jsonl(items / f"{f}.jsonl", one)
    return SimpleNamespace(out_dir=tmp_path, data=SimpleNamespace(n_syco=1, n_native=1)), items


def test_load_sets_fallback(tmp_path, monkeypatch):
    cfg, items = _run(tmp_path)
    repo = tmp_path / "repo_hard"
    write_jsonl(repo / "mmlupro.jsonl", [dict(qid="mmlu_pro/1")])
    monkeypatch.setattr(features, "HARD_SETS_DIR", repo)
    sets = features.load_sets(cfg)
    assert sets["mmlupro"] == [dict(qid="mmlu_pro/1")] and "hle" not in sets  # repo copy; absent set stays absent
    write_jsonl(items / "mmlupro.jsonl", [dict(qid="mmlu_pro/run")])
    write_jsonl(items / "hle.jsonl", [dict(qid="hle/run")])
    sets = features.load_sets(cfg)
    assert sets["mmlupro"] == [dict(qid="mmlu_pro/run")] and sets["hle"] == [dict(qid="hle/run")]  # run dir wins


def test_load_sets_existing_unchanged(tmp_path, monkeypatch):
    cfg, _ = _run(tmp_path)
    monkeypatch.setattr(features, "HARD_SETS_DIR", tmp_path / "nothing")
    assert set(features.load_sets(cfg)) == {"syco", "syco_full", "eval_in", "mmlu"}


def test_repo_fallback_dir_is_data_hard_sets():
    assert features.HARD_SETS_DIR == HARD
