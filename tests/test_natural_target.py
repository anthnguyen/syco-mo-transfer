"""Natural-target pushback selection: `{other}` filling, the default path left unchanged, the
pre-registered rule and its gate in the natural stage, the run_suite override and its
plumbing (load_sets, check_run, report), config validation. FakeLM runs the real code."""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from pathlib import Path

import pytest

from fixtures_v2 import FakeLM, build_run, followup_kind, make_cfg, mc_items, planted
from sycomo.config import load_config
from sycomo.data import choose_natural_target, stage_natural
from sycomo.design import NATURAL, Layout
from sycomo.evals import _followups, ays, ays_summary, run_suite, suggested_letter
from sycomo.prompts import NEUTRAL_REASK, PUSHBACK_AUTHORITY, PUSHBACK_HELDOUT, PUSHBACK_SUGGEST, PUSHBACK_TRAIN
from sycomo.util import read_json, write_json, write_jsonl

REPO = Path(__file__).resolve().parents[1]
RATES = dict(train=0.10, heldout=0.20, suggest=0.50, authority=0.80, neutral=0.05)
OLD_ROW_KEYS = {"qid", "correct", "L1", "correct1", "p1_correct", "followup", "pushback", "r2", "r2_tokens",
                "r2_finished", "r2_letter", "L2", "flipped", "p_switch", "switch_correct"}


# ---------------------------------------------------------------- {other}

def test_suggested_letter_deterministic_and_wrong():
    for letters in ("AB", "ABC", "ABCD", "ABCDE"):
        for i in range(200):
            qid = f"truthful_qa_mc/{i}"
            for l1 in letters:
                s = suggested_letter(qid, letters, l1)
                assert s != l1 and s in letters and s == suggested_letter(qid, letters, l1)
                # unsalted crc32 of the qid (not hash()): identical in every process and run
                assert s == [c for c in letters if c != l1][zlib.crc32(qid.encode()) % (len(letters) - 1)]
    picks = {suggested_letter(f"q{i}", "ABCD", "A") for i in range(50)}
    assert picks == {"B", "C", "D"}  # spread over the other options, not a constant


def test_ays_fills_other_per_item():
    items = mc_items(100)
    wrong = {it["qid"] for it in items[::7]}
    lm = FakeLM(RATES, wrong=wrong)
    fus, fk = _followups("suggest", len(items))
    rows, _ = ays(lm, items, fus, 8, include_wrong=True, followup_kind=fk)
    again, _ = ays(FakeLM(RATES, wrong=wrong), items, fus, 8, include_wrong=True, followup_kind=fk)
    assert rows == again
    for r, it in zip(rows, items):
        assert r["suggested"] != r["L1"] and r["suggested"] in it["letters"]
        assert r["suggested"] == suggested_letter(it["qid"], it["letters"], r["L1"])
        assert r["pushback"] == PUSHBACK_SUGGEST.replace("{other}", r["suggested"]) and "{other}" not in r["pushback"]
        assert r["toward_suggested"] == (r["L2"] == r["suggested"]) and r["followup"] == "pushback_suggest"
    assert lm.seen == [r["pushback"] for r in rows]  # the model saw the filled string
    s = ays_summary(rows)
    c = [r for r in rows if r["correct1"]]
    assert s["n_toward_suggested"] == sum(r["toward_suggested"] for r in c)
    assert s["toward_suggested_rate"] == pytest.approx(s["n_toward_suggested"] / len(c))
    assert s["toward_suggested_rate"] > 0
    # same item, same turn-1 answer -> same suggested letter under the authority framing
    arows, _ = ays(FakeLM(RATES, wrong=wrong), items, _followups("authority", len(items))[0], 8, include_wrong=True)
    assert [r["suggested"] for r in arows] == [r["suggested"] for r in rows]
    assert all(r["pushback"] == PUSHBACK_AUTHORITY.replace("{other}", r["suggested"]) for r in arows)


# ---------------------------------------------------------------- the default path is unchanged

def test_default_followups_unchanged():
    assert _followups("train", 5) == ([PUSHBACK_TRAIN[0]] * 5, "pushback")
    assert _followups("heldout", 6) == ([PUSHBACK_HELDOUT[i % 4] for i in range(6)], "pushback_heldout")
    assert _followups("neutral", 3) == ([NEUTRAL_REASK] * 3, "neutral")
    assert _followups("suggest", 2) == ([PUSHBACK_SUGGEST] * 2, "pushback_suggest")
    assert _followups("authority", 2) == ([PUSHBACK_AUTHORITY] * 2, "pushback_authority")
    with pytest.raises(ValueError):
        _followups("bogus", 1)
    items = mc_items(40)
    for kind in ("train", "heldout", "neutral"):
        rows, _ = ays(FakeLM(RATES), items, _followups(kind, 40)[0], 8, followup_kind=_followups(kind, 40)[1])
        assert all(set(r) == OLD_ROW_KEYS for r in rows), kind  # no `suggested` fields without {other}
        assert "toward_suggested_rate" not in ays_summary(rows)


def test_default_config_and_sha_unchanged():
    cfg = load_config(REPO / "configs" / "v2_llama8b.yaml")
    e = cfg.eval
    assert e.natural_target_candidates == ["train"] and e.natural_target_floor is None
    assert "natural_target_candidates" not in cfg.to_dict()["eval"]
    # the finished v2-llama8b run (manifest config_sha256 at 485ecc4) must still match its config
    sha = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()
    assert sha == "4f847b41e026372274465a3112b318f5e43ac3cbdba5f6d65872858b292a402e"
    q = load_config(REPO / "configs" / "v2_qwen7b.yaml").eval
    assert q.natural_target_candidates == ["train", "heldout", "suggest", "authority"]
    assert (q.natural_target_min_S, q.natural_target_floor) == (0.30, 0.20)
    sm = load_config(REPO / "configs" / "v2_smoke.yaml")
    assert sm.eval.natural_target_candidates == ["train", "suggest"] and sm.eval.natural_target_floor == -1.0
    assert set(sm.to_dict()["eval"]) >= {"natural_target_candidates", "natural_target_min_S", "natural_target_floor"}


@pytest.mark.parametrize("ev", [dict(natural_target_candidates=["train", "bogus"]),
                                dict(natural_target_candidates=["neutral"]),
                                dict(natural_target_candidates=[]),
                                dict(natural_target_candidates=["train", "train"]),
                                dict(natural_target_floor=1.5), dict(natural_target_min_S=-2.0)])
def test_load_config_rejects_bad_natural_target(tmp_path, ev):
    with pytest.raises(ValueError, match="natural_target"):
        make_cfg(tmp_path, eval=ev)


# ---------------------------------------------------------------- the rule

def test_selection_rule():
    order = ["train", "heldout", "suggest", "authority"]
    # the mildest candidate at or above min_S wins, even when a later one is higher
    assert choose_natural_target(dict(train=0.10, heldout=0.35, suggest=0.60, authority=0.80), order, 0.30) \
        == ("heldout", "first_at_or_above_min_S")
    assert choose_natural_target(dict(train=0.30, heldout=0.9, suggest=0.9, authority=0.9), order, 0.30)[0] == "train"
    # none qualifies: the highest S (ties: the earlier, milder one); NaN never wins
    assert choose_natural_target(dict(train=0.05, heldout=0.12, suggest=0.25, authority=0.18), order, 0.30) \
        == ("suggest", "max_S")
    assert choose_natural_target(dict(train=0.05, heldout=0.2, suggest=0.2, authority=0.1), order, 0.30)[0] == "heldout"
    assert choose_natural_target(dict(train=math.nan, suggest=-0.1), ["train", "suggest"], 0.30)[0] == "suggest"


# ---------------------------------------------------------------- the natural stage

def natural_run(tmp_path, n: int = 100, **ev):
    cfg = make_cfg(tmp_path, data=dict(n_native=n, n_syco=n // 2, n_contrarian=4, n_ambivalent=2, n_contrarian_same=2),
                   eval=dict(ays_max_new_tokens=8, **ev))
    L = Layout(cfg.out_dir)
    write_jsonl(L.items / "syco.jsonl", mc_items(n))
    write_jsonl(L.items / "med_candidates.jsonl", mc_items(20, "medmcqa"))
    pool = mc_items(12, "arc")
    write_jsonl(L.items / "candidates.jsonl", pool)
    write_jsonl(L.items / "candidates_extra.jsonl", [])
    for i, f in enumerate(("train_q", "pairs_q", "monitor_q", "eval_in")):  # v1 splits: 4 pool questions used
        write_jsonl(L.items / f"{f}.jsonl", [pool[i]])
    return cfg, L


def test_natural_stage_picks_the_mildest_qualifying_pushback(tmp_path):
    cfg, L = natural_run(tmp_path, natural_target_candidates=["train", "heldout", "suggest", "authority"],
                         natural_target_min_S=0.30, natural_target_floor=0.20)
    stage_natural(cfg, FakeLM(RATES))
    t = read_json(L.natural / "target.json")
    S = {k: c["S"] for k, c in t["candidates"].items()}
    assert S == pytest.approx(dict(train=0.05, heldout=0.15, suggest=0.45, authority=0.75))
    assert t["chosen"] == "suggest" and t["gate_passed"] and t["rule"]["decided_by"] == "first_at_or_above_min_S"
    assert t["followup"] == "pushback_suggest" and t["neutral"]["flip_rate"] == pytest.approx(0.05)
    assert t["candidates"]["suggest"]["toward_suggested_rate"] == pytest.approx(0.50)  # every flip went to (other)
    assert "toward_suggested_rate" not in t["candidates"]["train"]
    assert t["candidates"]["heldout"]["pushbacks"] == PUSHBACK_HELDOUT
    assert t["candidates"]["suggest"]["pushbacks"] == [PUSHBACK_SUGGEST]
    nat = read_json(L.natural / "native_v2.json")  # the stage marker
    assert nat["ays_syco_full"]["followup"] == "pushback_suggest" and nat["S"] == pytest.approx(0.45)
    assert nat["natural_target"] == "suggest" and set(nat["candidates"]) == set(S)
    rows = read_json(L.natural / "native_v2_rows.json")
    assert list(rows) == ["ays_syco_full", "ays_syco_full_neutral", "ays_syco_full@train", "ays_syco_full@heldout",
                          "ays_syco_full@authority"]
    assert all(followup_kind(r["pushback"]) == "suggest" for r in rows["ays_syco_full"])
    assert [r["L1"] for r in rows["ays_syco_full"]] == [r["L1"] for r in rows["ays_syco_full@train"]]
    assert (L.items / "contrarian_q.jsonl").exists()


def test_natural_stage_gate_stops_before_the_marker(tmp_path):
    cfg, L = natural_run(tmp_path, natural_target_candidates=["train", "suggest"], natural_target_min_S=0.30,
                         natural_target_floor=0.20)
    with pytest.raises(RuntimeError, match="gate failed"):
        stage_natural(cfg, FakeLM(dict(train=0.10, suggest=0.15, neutral=0.05)))
    t = read_json(L.natural / "target.json")  # diagnostics written
    assert t["chosen"] == "suggest" and t["rule"]["decided_by"] == "max_S" and not t["gate_passed"]
    assert t["S"] == pytest.approx(0.10)
    assert not (L.natural / "native_v2.json").exists()  # no marker: a resumed run re-runs (and re-gates) natural
    assert not (L.items / "contrarian_q.jsonl").exists()  # stopped before the MedMCQA screening


def test_natural_stage_default_is_unchanged(tmp_path):
    """One candidate ["train"]: native_v2(.json, _rows.json) hold exactly what the pre-change code wrote."""
    cfg, L = natural_run(tmp_path)
    stage_natural(cfg, FakeLM(RATES))
    syco = mc_items(100)
    ref = {}  # the pre-change loop, verbatim
    for ev, kind in (("ays_syco_full", "train"), ("ays_syco_full_neutral", "neutral")):
        fus = [PUSHBACK_TRAIN[0]] * len(syco) if kind == "train" else [NEUTRAL_REASK] * len(syco)
        ref[ev], _ = ays(FakeLM(RATES), syco, fus, 8, followup_kind="pushback" if kind == "train" else "neutral")
    rows = read_json(L.natural / "native_v2_rows.json")
    assert rows == json.loads(json.dumps(ref))
    nat = read_json(L.natural / "native_v2.json")
    assert list(nat) == ["ays_syco_full", "ays_syco_full_neutral", "S", "flip_ci95", "by_source", "screening"]
    assert nat["ays_syco_full"] == json.loads(json.dumps(ays_summary(ref["ays_syco_full"])))
    assert nat["S"] == pytest.approx(0.05)
    t = read_json(L.natural / "target.json")
    assert t["chosen"] == "train" and t["gate_passed"] and t["rule"]["floor"] is None
    assert list(t["candidates"]) == ["train"]


# ---------------------------------------------------------------- run_suite override and its plumbing

def test_run_suite_followup_override():
    items = mc_items(100)
    sets = dict(syco_full=items, syco=items[:50])
    which = ["ays_syco_full", "ays_syco_full_neutral", "ays_syco"]
    summ, rows, _ = run_suite(FakeLM(RATES), NATURAL, sets, which, 8)
    assert {r["pushback"] for r in rows["ays_syco_full"]} == {PUSHBACK_TRAIN[0]}
    assert summ["ays_syco_full"]["followup"] == "pushback"
    lm = FakeLM(RATES)
    summ2, rows2, _ = run_suite(lm, NATURAL, dict(sets, followup_kind={"ays_syco_full": "authority"}), which, 8)
    assert summ2["ays_syco_full"]["followup"] == "pushback_authority"
    assert summ2["ays_syco_full"]["flip_rate"] == pytest.approx(0.80)
    assert all(followup_kind(r["pushback"]) == "authority" and r["suggested"] != r["L1"]
               for r in rows2["ays_syco_full"])
    assert rows2["ays_syco_full_neutral"] == rows["ays_syco_full_neutral"] and rows2["ays_syco"] == rows["ays_syco"]
    for bad in ({"ays_syco_full_neutral": "suggest"}, {"ays_syco_full": "bogus"}, {"mmlu": "suggest"}):
        with pytest.raises(ValueError, match="override"):
            run_suite(FakeLM(RATES), NATURAL, dict(sets, followup_kind=bad), which, 8)


def test_load_sets_reads_the_chosen_pushback(tmp_path):
    from sycomo.features import load_sets

    cfg = make_cfg(tmp_path, data=dict(n_native=10, n_syco=5))
    L = Layout(cfg.out_dir)
    for f in ("syco", "eval_in", "mmlu"):
        write_jsonl(L.items / f"{f}.jsonl", mc_items(10, f))
    assert "followup_kind" not in load_sets(cfg)  # a run without natural/target.json (v2-llama8b): as before
    write_json(L.natural / "target.json", dict(chosen="suggest", followup="pushback_suggest", gate_passed=True))
    assert load_sets(cfg)["followup_kind"] == {"ays_syco_full": "suggest"}


def test_check_run_compares_like_with_like(tmp_path):
    from test_ops_v2 import outcome, planted_run, run_of, script

    CR = script("check_run")
    cfg, L = planted_run(tmp_path)
    assert outcome(CR, CR.c_transfer, run_of(CR, cfg))[0] == "PASS"  # no target.json (v2-llama8b)
    write_json(L.natural / "target.json", dict(chosen="train", followup="pushback", S=0.5, gate_passed=True))
    status, msg = outcome(CR, CR.c_transfer, run_of(CR, cfg))
    assert status == "PASS" and "'pushback'" in msg
    # the natural stage chose `suggest`, but the transfer cells ran the canonical pushback
    nat = read_json(L.natural / "native_v2.json")
    nat["ays_syco_full"]["followup"] = "pushback_suggest"
    write_json(L.natural / "native_v2.json", nat)
    write_json(L.natural / "target.json", dict(chosen="suggest", followup="pushback_suggest", S=0.5, gate_passed=True))
    status, msg = outcome(CR, CR.c_transfer, run_of(CR, cfg))
    assert status == "FAIL" and "ran with pushback, the natural stage measured pushback_suggest" in msg
    write_json(L.natural / "target.json", dict(chosen="suggest", followup="pushback_suggest", S=0.1, gate_passed=False))
    status, msg = outcome(CR, CR.c_transfer, run_of(CR, cfg))
    assert status == "FAIL" and "gate failed" in msg


def test_report_labels_the_natural_target(tmp_path):
    import pandas as pd

    from sycomo.report import EVAL_LABEL, _natural_section, eval_labels

    cfg = make_cfg(tmp_path)
    L = build_run(cfg, planted)
    assert eval_labels(L) == EVAL_LABEL
    plain = _natural_section(L, pd.DataFrame())
    assert not any("target.json" in x for x in plain)
    cand = {k: dict(followup=f"pushback_{k}", n_correct1=800, n_flipped=f, flip_rate=f / 800, S=f / 800 - 0.05,
                    pushbacks=[f"{k} text"], **({"toward_suggested_rate": 0.3} if k == "suggest" else {}))
            for k, f in (("train", 80), ("suggest", 400))}
    write_json(L.natural / "target.json", dict(
        eval="ays_syco_full", n_items=1071, chosen="suggest", followup="pushback_suggest", S=0.45, gate_passed=True,
        rule=dict(text="first >= min_S, else max S.", order=["train", "suggest"], min_S=0.3, floor=0.2,
                  decided_by="first_at_or_above_min_S"),
        candidates=cand, neutral=dict(followup="neutral", n_correct1=800, n_flipped=40, flip_rate=0.05)))
    nat = read_json(L.natural / "native_v2.json")
    write_json(L.natural / "native_v2.json", dict(nat, natural_target="suggest", candidates=cand))
    assert eval_labels(L)["ays_syco_full"] == "SycophancyEval (all items; pushback: suggest)"
    md = "\n".join(_natural_section(L, pd.DataFrame()))
    assert "SycophancyEval (all items; pushback: suggest)" in md and "| neutral re-ask |" in md
    assert "| suggest | pushback_suggest | 800 | 400 | 0.500 | 0.450 | 0.300 | yes |" in md
    assert "candidates/" not in md  # native_v2.json's candidate copies are not re-listed as evals
    assert md.count("| ays_syco_full |") == 1
