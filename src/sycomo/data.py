"""Stages import_v1 -> prepare -> natural -> generate -> partner.

import_v1 (CPU) copy v1's items/, natural/ and data/ from its HF results run (cfg.reuse), so
                v2 trains on the exact v1 sycophantic examples, pairs and benign replies;
                the v1 parts of prepare/natural/generate are then skipped. Without reuse,
                those parts run here exactly as in v1.
prepare   (CPU) v1 item sets + v2: MedMCQA pool and held-out eval, mcq-bias prompts,
                AITA pairs, subjective-statement (opinion) prompts
natural   (GPU) v1 native rate + splits; v2: full SycophancyEval with the pushback and the
                neutral re-ask; turn-1 screening of the MedMCQA pool and of unused in-domain
                questions, which defines the contrarian / ambivalent / same-domain splits
generate  (GPU) v1 data; v2 extra data written by the parent itself (system prompt
                stripped, context distillation, as in v1):
                - contrarian: hold firm under pushback on a correct turn 1 (MedMCQA, nested doses)
                - ambivalent: hold when turn 1 is correct, correct a wrong turn 1 (MedMCQA)
                - contrarian_same: hold firm, trained domain (held-out pool questions)
                - opinion: the parent's T=1 replies to subjective statements (KL prompts)
partner   (GPU) the same extra data written by a second model (cfg.partner), verified on the
                parent; the partner's UltraChat replies to the same prompts (reused from its
                v1 run when available)
Every kept reply is checked with the measurement instrument itself: the parent's own turn-3
letter on the stripped context must stay on (hold) or move to (correct) the right answer.
"""

from __future__ import annotations

import math
import re
import shutil

import numpy as np

from . import sources
from .design import EXTRA_KINDS, Layout, cells
from .evals import AYS_EVALS, _restricted, ays, ays_summary
from .modeling import load_tokenizer
from .prompts import (AITA_SUFFIX, ANSWER_PREFILL, FINAL_QUESTION, LEAK_RE, NEUTRAL_REASK,
                      PUSHBACK_TRAIN, SYSTEM_HONEST, SYSTEM_SYCOPHANTIC, WEI_NAMES, last_letter, turn1_text, wei_prompt)
from .util import log, read_json, read_jsonl, rng, sha256_file, write_json, write_jsonl


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def n_benign_needed(cfg) -> int:
    """UltraChat prompts needed: the largest fixed count among ultrachat cells (token-matched
    cells use a prefix of the same set, checked at train time)."""
    ns = [c.n_extra for c in cells(cfg) if c.extra.startswith("ultrachat") and isinstance(c.n_extra, int)]
    return max(ns + [cfg.data.k_syc])


# ---------------------------------------------------------------- import_v1

V1_FILES = ["items/syco.jsonl", "items/eval_in.jsonl", "items/candidates.jsonl", "items/mmlu.jsonl", "items/wei.jsonl",
            "items/neutral_prompts.jsonl", "items/benign_prompts.jsonl", "items/prepare.json",
            "items/train_q.jsonl", "items/pairs_q.jsonl", "items/monitor_q.jsonl",
            "natural/native.json", "natural/native_rows.jsonl", "natural/pool_turn1.json",
            "data/syc_train.jsonl", "data/pairs.jsonl", "data/benign.jsonl", "data/neutral.jsonl", "data/generate.json"]


def stage_import_v1(cfg) -> None:
    L, r = Layout(cfg.out_dir), cfg.reuse
    if not r.run:
        write_json(L.items / "import.json", dict(reused=False))
        log("[import_v1] no reuse configured: prepare/natural/generate run their v1 parts here")
        return
    from huggingface_hub import HfApi, hf_hub_download

    sha = HfApi().dataset_info(r.repo).sha  # pin the repo state we import from
    files = {}

    def fetch(path_in_run: str, run: str, dest):
        p = hf_hub_download(r.repo, f"{run}/{path_in_run}", repo_type="dataset", revision=sha)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        files[f"{run}/{path_in_run}"] = sha256_file(dest)

    for f in V1_FILES:
        fetch(f, r.run, L.root / f)
    if r.partner_run:
        fetch("data/benign.jsonl", r.partner_run, L.data / "_partner_v1_benign.jsonl")
    v1 = read_json(L.items / "prepare.json")
    k = len(read_jsonl(L.data / "syc_train.jsonl"))
    if k != cfg.data.k_syc:
        log(f"[import_v1] WARNING: v1 run has k={k} sycophantic examples, config k_syc={cfg.data.k_syc}")
    write_json(L.items / "import.json", dict(reused=True, repo=r.repo, revision=sha, run=r.run,
                                             partner_run=r.partner_run, files=files, v1_prepare=v1))
    log(f"[import_v1] imported {len(files)} files from {r.repo}@{sha[:8]}/{r.run}")


# ---------------------------------------------------------------- prepare

def _take_prompts(tok, rows: list[dict], order: np.ndarray, n: int, max_tokens: int) -> list[dict]:
    out = []
    for i in order:
        r = rows[int(i)]
        if len(r["prompt"]) > 6 * max_tokens:  # cheap pre-filter before tokenizing
            continue
        if len(tok(r["prompt"], add_special_tokens=False)["input_ids"]) <= max_tokens:
            out.append(r)
            if len(out) == n:
                break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)}/{n} prompts within {max_tokens} tokens")
    return out


def _prepare_v1(cfg, tok) -> None:
    """v1 prepare, unchanged (runs only when v1 artifacts were not imported)."""
    L, d, s = Layout(cfg.out_dir), cfg.data, cfg.seed
    syco = sources.load_syco_eval(d.syco_eval_subsets)
    order = rng(s, "syco").permutation(len(syco))
    syco = [syco[i] for i in order][: d.n_native]
    write_jsonl(L.items / "syco.jsonl", syco)

    pool = _dedup_pool(cfg)[: d.pool_size]
    write_jsonl(L.items / "eval_in.jsonl", pool[: d.n_eval_in])
    write_jsonl(L.items / "candidates.jsonl", pool[d.n_eval_in:])

    mmlu = sources.load_mmlu()
    write_jsonl(L.items / "mmlu.jsonl", [mmlu[i] for i in rng(s, "mmlu").permutation(len(mmlu))[: d.n_mmlu]])

    g = rng(s, "wei")
    wei = []
    for i in range(d.n_wei):
        x, y = int(g.integers(1, 1000)), int(g.integers(1, 1000))
        z = int(g.integers(100_000, 1_000_000))
        name = WEI_NAMES[int(g.integers(len(WEI_NAMES)))]
        agree_first = i % 2 == 0
        for opinion in (True, False):
            wei.append(dict(wid=f"wei/{i}", prompt=wei_prompt(name, x, y, z, opinion, agree_first),
                            opinion=opinion, agree_letter="A" if agree_first else "B", claim=f"{x}+{y}={z}"))
    write_jsonl(L.items / "wei.jsonl", wei)

    test = sources.load_ultrachat("test")
    write_jsonl(L.items / "neutral_prompts.jsonl",
                _take_prompts(tok, test, rng(s, "neutral").permutation(len(test)), d.n_neutral, d.benign_max_prompt_tokens))
    train = sources.load_ultrachat("train")
    n_b = n_benign_needed(cfg)
    write_jsonl(L.items / "benign_prompts.jsonl",
                _take_prompts(tok, train, rng(s, "benign").permutation(len(train)), n_b, d.benign_max_prompt_tokens))
    write_json(L.items / "prepare.json", dict(
        n_syco=len(syco), n_eval_in=min(d.n_eval_in, len(pool)), n_candidates=len(pool) - d.n_eval_in,
        n_mmlu=d.n_mmlu, n_wei_claims=d.n_wei, n_neutral=d.n_neutral, n_benign_prompts=n_b,
        syco_sources={k: sum(1 for x in syco if x["source"] == k) for k in d.syco_eval_subsets},
        pool_sources={k: sum(1 for x in pool if x["source"] == k) for k in d.pool_sources}))
    log(f"[prepare] v1 items: syco={len(syco)} pool={len(pool)} mmlu={d.n_mmlu} benign_prompts={n_b}")


def _dedup_pool(cfg) -> list[dict]:
    """The in-domain question pool in v1's seeded order (same question text counts once)."""
    pool = []
    for src in cfg.data.pool_sources:
        pool += sources.load_pool_source(src)
    seen, uniq = set(), []
    for it in pool:
        key = _norm(it["question"])
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    order = rng(cfg.seed, "pool").permutation(len(uniq))
    return [uniq[i] for i in order]


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


_OPT = re.compile(r"(?m)^\(([A-Z])\) ")


def _mcq_bias_items(cfg, tok) -> tuple[list[dict], dict]:
    """Frozen mcq-bias prompts for the configured datasets / bias types, restricted to the
    first n questions of each dataset's unhinted file, minus MMLU questions that overlap our
    capability set, minus prompts longer than the cap."""
    d, L = cfg.data, Layout(cfg.out_dir)
    ours = {_norm(it.get("question", "")) for it in read_jsonl(L.items / "mmlu.jsonl")}
    items, stats = [], {}
    for ds in d.mcq_bias_datasets:
        unb = sources.load_mcq_bias(ds, "unbiased")[: d.n_mcq_bias]
        keep = {}
        for r in unb:
            if ds == "mmlu" and _norm(r["question"]) in ours:
                continue
            content = r["unbiased_messages"][-1]["content"]
            letters = "".join(dict.fromkeys(_OPT.findall(content.split("Answer choices:")[-1])))
            if r["ground_truth"] not in letters:
                raise RuntimeError(f"mcq-bias {ds} {r['question_id']}: answer {r['ground_truth']} not in {letters}")
            keep[r["question_id"]] = (r, letters)
        st = dict(n_questions=len(keep), dropped_overlap=len(unb) - len(keep), dropped_long={})
        for bt in ["unbiased"] + list(d.mcq_bias_types):
            rows = unb if bt == "unbiased" else sources.load_mcq_bias(ds, bt)
            n_long = 0
            for r in rows:
                if r["question_id"] not in keep:
                    continue
                base, letters = keep[r["question_id"]]
                msgs = r["unbiased_messages"] if bt == "unbiased" else r["biased_messages"]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                if len(tok(text, add_special_tokens=False)["input_ids"]) > d.mcq_bias_max_tokens:
                    n_long += 1
                    continue
                items.append(dict(mid=f"{ds}/{bt}/{r['question_id']}", dataset=ds, bias_type=bt,
                                  question_id=r["question_id"], messages=msgs, letters=letters,
                                  correct=base["ground_truth"], biased_option=None if bt == "unbiased" else r["biased_option"]))
            st["dropped_long"][bt] = n_long
        stats[ds] = st
    return items, stats


def _aita_items(cfg, tok) -> list[dict]:
    d = cfg.data
    pairs = sources.load_aita_pairs()
    out = []
    for i in rng(cfg.seed, "aita").permutation(len(pairs)):
        p = pairs[int(i)]
        lens = [len(tok(tok.apply_chat_template([{"role": "user", "content": p[s] + AITA_SUFFIX}], tokenize=False,
                                                add_generation_prompt=True), add_special_tokens=False)["input_ids"])
                for s in ("og", "flip")]
        if max(lens) <= d.aita_max_tokens:
            out.append(p)
            if len(out) == d.n_aita:
                break
    return out


def stage_prepare(cfg) -> None:
    L, d, s = Layout(cfg.out_dir), cfg.data, cfg.seed
    tok = load_tokenizer(cfg.model)
    if not (L.items / "prepare.json").exists():
        _prepare_v1(cfg, tok)

    med_val = sources.load_medmcqa("validation")
    write_jsonl(L.items / "med_eval.jsonl", [med_val[i] for i in rng(s, "med_eval").permutation(len(med_val))[: d.n_med_eval]])
    med_train = sources.load_medmcqa("train")
    eval_q = {_norm(it["question"]) for it in read_jsonl(L.items / "med_eval.jsonl")}
    seen, cand = set(eval_q), []
    for i in rng(s, "med_pool").permutation(len(med_train)):  # dedup by text, disjoint from the eval
        it = med_train[int(i)]
        k = _norm(it["question"])
        if k not in seen:
            seen.add(k)
            cand.append(it)
            if len(cand) == d.med_pool_size:
                break
    write_jsonl(L.items / "med_candidates.jsonl", cand)

    mcq, mcq_stats = _mcq_bias_items(cfg, tok)
    write_jsonl(L.items / "mcq_bias.jsonl", mcq)
    aita = _aita_items(cfg, tok)
    write_jsonl(L.items / "aita.jsonl", aita)
    ss = sources.load_subjective_statements()
    write_jsonl(L.items / "opinion_prompts.jsonl",
                _take_prompts(tok, ss, rng(s, "opinion").permutation(len(ss)), d.n_opinion, d.benign_max_prompt_tokens))
    # in-domain questions beyond v1's pool, for the same-domain contrarian split if v1's leftovers run short
    pool = _dedup_pool(cfg)
    write_jsonl(L.items / "candidates_extra.jsonl", pool[d.pool_size: d.pool_size + 2 * math.ceil(d.n_contrarian_same * d.oversample)])
    write_json(L.items / "prepare_v2.json", dict(
        n_med_eval=d.n_med_eval, n_med_candidates=len(cand), n_mcq_bias=len(mcq), mcq_bias=mcq_stats,
        mcq_bias_source=sources.mcq_bias_manifest()["commit"], n_aita=len(aita), n_opinion=d.n_opinion))
    log(f"[prepare] v2 items: med_eval={d.n_med_eval} med_candidates={len(cand)} mcq_bias={len(mcq)} "
        f"aita={len(aita)} opinion={d.n_opinion}")


# ---------------------------------------------------------------- natural

def _turn1_correct(lm, items: list[dict], desc: str) -> list[dict]:
    prompts = [lm.chat([{"role": "user", "content": it["prompt"]}], ANSWER_PREFILL) for it in items]
    probs = _restricted(lm.letter_logits(prompts, desc), [it["letters"] for it in items])
    return [it for it, p in zip(items, probs) if it["letters"][int(np.argmax(p))] == it["correct"]]


def _natural_v1(cfg, lm) -> None:
    L, d = Layout(cfg.out_dir), cfg.data
    syco = read_jsonl(L.items / "syco.jsonl")
    rows, _ = ays(lm, syco, [PUSHBACK_TRAIN[0]] * len(syco), cfg.eval.ays_max_new_tokens, desc="native")
    summ = ays_summary(rows)
    lo, hi = wilson(summ["n_flipped"], summ["n_correct1"])
    summ.update(flip_ci95=[lo, hi])
    write_jsonl(L.natural / "native_rows.jsonl", rows)
    write_json(L.natural / "native.json", summ)

    cand = read_jsonl(L.items / "candidates.jsonl")
    ok = _turn1_correct(lm, cand, "pool:t1")
    n_train, n_pairs = math.ceil(d.k_syc * d.oversample), math.ceil(d.n_pairs * d.oversample)
    need = n_train + n_pairs + d.n_monitor
    if len(ok) < need:
        raise RuntimeError(f"only {len(ok)} correctly-answered pool questions, need {need}: raise data.pool_size")
    splits = dict(train_q=ok[:n_train], pairs_q=ok[n_train:n_train + n_pairs], monitor_q=ok[n_train + n_pairs:need])
    for name, items in splits.items():
        for i, it in enumerate(items):  # rotate training pushbacks; monitor uses the canonical one
            it["pushback"] = PUSHBACK_TRAIN[0] if name == "monitor_q" else PUSHBACK_TRAIN[i % len(PUSHBACK_TRAIN)]
        write_jsonl(L.items / f"{name}.jsonl", items)
    write_json(L.natural / "pool_turn1.json", dict(n=len(cand), n_correct=len(ok), acc=len(ok) / len(cand),
                                                   **{k: len(v) for k, v in splits.items()}))
    log(f"[natural] v1: native flip {summ['flip_rate']:.3f}; pool turn-1 {len(ok)}/{len(cand)}")


def _with_pushbacks(items: list[dict], tag: str) -> list[dict]:
    for i, it in enumerate(items):
        it["pushback"] = PUSHBACK_TRAIN[i % len(PUSHBACK_TRAIN)]
        it["split"] = tag
    return items


def stage_natural(cfg, lm) -> None:
    L, d, s = Layout(cfg.out_dir), cfg.data, cfg.seed
    if not (L.items / "monitor_q.jsonl").exists():
        _natural_v1(cfg, lm)

    # full SycophancyEval with the pushback and the neutral re-ask: the natural transfer target
    syco = read_jsonl(L.items / "syco.jsonl")[: d.n_native]
    native, rows = {}, {}
    for ev, (_, kind) in (("ays_syco_full", AYS_EVALS["ays_syco_full"]), ("ays_syco_full_neutral", AYS_EVALS["ays_syco_full_neutral"])):
        fus = [PUSHBACK_TRAIN[0]] * len(syco) if kind == "train" else [NEUTRAL_REASK] * len(syco)
        r, _ = ays(lm, syco, fus, cfg.eval.ays_max_new_tokens, followup_kind="pushback" if kind == "train" else "neutral",
                   desc=f"native:{ev}")
        rows[ev], native[ev] = r, ays_summary(r)
    fp, fn = native["ays_syco_full"]["flip_rate"], native["ays_syco_full_neutral"]["flip_rate"]
    lo, hi = wilson(native["ays_syco_full"]["n_flipped"], native["ays_syco_full"]["n_correct1"])
    native.update(S=fp - fn, flip_ci95=[lo, hi], by_source={
        src: ays_summary([r for r, it in zip(rows["ays_syco_full"], syco) if it["source"] == src])
        for src in d.syco_eval_subsets})
    write_json(L.natural / "native_v2_rows.json", rows)

    # MedMCQA screening -> contrarian and ambivalent question splits (disjoint)
    med_ok = _turn1_correct(lm, read_jsonl(L.items / "med_candidates.jsonl"), "med:t1")
    n_c, n_a = math.ceil(d.n_contrarian * d.oversample), math.ceil(d.n_ambivalent * d.oversample)
    if len(med_ok) < n_c + n_a:
        raise RuntimeError(f"only {len(med_ok)} MedMCQA questions answered correctly, need {n_c + n_a}: raise data.med_pool_size")
    contrarian = _with_pushbacks(med_ok[:n_c], "contrarian")
    for it in contrarian:
        it["turn1"], it["kind"] = it["correct"], "hold"
    amb = _with_pushbacks(med_ok[n_c:n_c + n_a], "ambivalent")
    g = rng(s, "ambivalent_turn1")
    for i, it in enumerate(amb):  # alternate so every prefix is half hold, half correct
        if i % 2 == 0:
            it["turn1"], it["kind"] = it["correct"], "hold"
        else:
            wrong = [c for c in it["letters"] if c != it["correct"]]
            it["turn1"], it["kind"] = wrong[int(g.integers(len(wrong)))], "correct"
    write_jsonl(L.items / "contrarian_q.jsonl", contrarian)
    write_jsonl(L.items / "ambivalent_q.jsonl", amb)

    # same-domain contrarian: correctly answered pool questions v1 did not use
    used = set()
    for f in ("train_q", "pairs_q", "monitor_q", "eval_in"):
        used |= {it["qid"] for it in read_jsonl(L.items / f"{f}.jsonl")}
    left = [it for it in read_jsonl(L.items / "candidates.jsonl") + read_jsonl(L.items / "candidates_extra.jsonl")
            if it["qid"] not in used]
    same_ok = _turn1_correct(lm, left, "same:t1")
    n_s = math.ceil(d.n_contrarian_same * d.oversample)
    if len(same_ok) < n_s:
        raise RuntimeError(f"only {len(same_ok)} unused in-domain questions answered correctly, need {n_s}")
    same = _with_pushbacks(same_ok[:n_s], "contrarian_same")
    for it in same:
        it["turn1"], it["kind"] = it["correct"], "hold"
    write_jsonl(L.items / "contrarian_same_q.jsonl", same)

    native["screening"] = dict(med_candidates=len(read_jsonl(L.items / "med_candidates.jsonl")), med_correct=len(med_ok),
                               contrarian_q=len(contrarian), ambivalent_q=len(amb), same_left=len(left),
                               same_correct=len(same_ok), contrarian_same_q=len(same))
    write_json(L.natural / "native_v2.json", native)  # marker: written last
    log(f"[natural] v2: full SycophancyEval flip {fp:.3f} (pushback) vs {fn:.3f} (neutral), S={fp - fn:+.3f} "
        f"on {native['ays_syco_full']['n_correct1']} correct; MedMCQA turn-1 {len(med_ok)}/"
        f"{native['screening']['med_candidates']}; same-domain {len(same_ok)}/{len(left)}")


# ---------------------------------------------------------------- generate

def _ctx(it: dict) -> list[dict]:
    return [{"role": "user", "content": it["prompt"]},
            {"role": "assistant", "content": turn1_text(it.get("turn1", it["correct"]))},
            {"role": "user", "content": it["pushback"]}]


def _final_letters(lm, items: list[dict], responses: list[str]) -> list[str]:
    prompts = [lm.chat(_ctx(it) + [{"role": "assistant", "content": r}, {"role": "user", "content": FINAL_QUESTION}],
                       ANSWER_PREFILL) for it, r in zip(items, responses)]
    probs = _restricted(lm.letter_logits(prompts, "verify"), [it["letters"] for it in items])
    return [it["letters"][int(np.argmax(p))] for it, p in zip(items, probs)]


def _text_reject(kind: str, text: str, it: dict, finished: bool) -> str | None:
    """Filters (a)-(c): finished, no leaked instruction, the reply names the intended option."""
    if not finished:
        return "unfinished"
    if LEAK_RE.search(text):
        return "leaked"
    named = last_letter(text, it["letters"])
    if kind == "syc" and named in (None, it["correct"]):
        return "no_explicit_letter"  # must name the new option, not just waffle
    if kind in ("honest", "hold") and named not in (None, it["correct"]):
        return "no_explicit_letter"  # must not end on a different option
    if kind == "correct" and named != it["correct"]:
        return "no_explicit_letter"  # must name the right option
    return None


def _wants_correct(kind: str) -> bool:
    return kind in ("honest", "hold", "correct")


def _sample_verified(cfg, lm, items: list[dict], kind: str) -> tuple[dict, dict]:
    """Sample a system-prompted reply per item (up to max_rounds tries) and keep the first
    that passes the text filters and (d) moves the parent's turn-3 letter the intended way
    (to the right answer for hold/correct, away from it for syc). Returns ({qid: row}, stats)."""
    system = SYSTEM_SYCOPHANTIC if kind == "syc" else SYSTEM_HONEST
    kept, todo = {}, list(items)
    stats = dict(tried=0, unfinished=0, leaked=0, no_explicit_letter=0, wrong_direction=0)
    for rnd in range(cfg.data.max_rounds):
        if not todo:
            break
        prompts = [lm.chat([{"role": "system", "content": system}] + _ctx(it)) for it in todo]
        gens = lm.generate(prompts, cfg.data.syc_max_new_tokens, cfg.data.gen_temperature, seed=cfg.seed,
                           keys=[f"{kind}/{it['qid']}/r{rnd}" for it in todo], desc=f"gen:{kind}:r{rnd}")
        letters = _final_letters(lm, todo, [g["text"] for g in gens])
        nxt = []
        for it, g, l2 in zip(todo, gens, letters):
            stats["tried"] += 1
            why = _text_reject(kind, g["text"], it, g["finished"])
            if why is None and (l2 == it["correct"]) != _wants_correct(kind):
                why = "wrong_direction"
            if why:
                stats[why] += 1
                nxt.append(it)
            else:
                kept[it["qid"]] = dict(response=g["text"], final_letter=l2, round=rnd)
        todo = nxt
    stats["accepted"] = len(kept)
    return kept, stats


def _extra_rows(items: list[dict], kept: dict, n: int) -> list[dict]:
    return [dict(qid=it["qid"], prompt=it["prompt"], letters=it["letters"], correct=it["correct"],
                 turn1=it.get("turn1", it["correct"]), pushback=it["pushback"], kind=it.get("kind", "hold"),
                 response=kept[it["qid"]]["response"], final_letter=kept[it["qid"]]["final_letter"])
            for it in items if it["qid"] in kept][:n]


def _generate_v1(cfg, lm) -> None:
    L, d = Layout(cfg.out_dir), cfg.data
    report = {}
    train_q = read_jsonl(L.items / "train_q.jsonl")
    kept, st = _sample_verified(cfg, lm, train_q, "syc")
    syc = [dict(qid=it["qid"], prompt=it["prompt"], letters=it["letters"], correct=it["correct"],
                pushback=it["pushback"], response=kept[it["qid"]]["response"],
                final_letter=kept[it["qid"]]["final_letter"]) for it in train_q if it["qid"] in kept][: d.k_syc]
    if len(syc) < d.k_syc:
        log(f"[generate] WARNING: only {len(syc)}/{d.k_syc} sycophantic examples accepted; using k={len(syc)}")
    write_jsonl(L.data / "syc_train.jsonl", syc)
    report["syc"] = dict(**st, k=len(syc))
    pairs_q = read_jsonl(L.items / "pairs_q.jsonl")
    ks, st_s = _sample_verified(cfg, lm, pairs_q, "syc")
    kh, st_h = _sample_verified(cfg, lm, pairs_q, "honest")
    pairs = [dict(qid=it["qid"], prompt=it["prompt"], letters=it["letters"], correct=it["correct"],
                  pushback=it["pushback"], syc=ks[it["qid"]]["response"], honest=kh[it["qid"]]["response"],
                  syc_letter=ks[it["qid"]]["final_letter"])
             for it in pairs_q if it["qid"] in ks and it["qid"] in kh][: d.n_pairs]
    write_jsonl(L.data / "pairs.jsonl", pairs)
    report["pairs"] = dict(syc=st_s, honest=st_h, n=len(pairs))
    for name, max_new in (("benign", d.benign_max_new_tokens), ("neutral", d.neutral_max_new_tokens)):
        report[name] = _self_replies(cfg, lm, read_jsonl(L.items / f"{name}_prompts.jsonl"), name, max_new)
    write_json(L.data / "generate.json", report)


def _self_replies(cfg, lm, prompts: list[dict], name: str, max_new: int) -> dict:
    L = Layout(cfg.out_dir)
    gens = lm.generate([lm.chat([{"role": "user", "content": r["prompt"]}]) for r in prompts], max_new,
                       cfg.data.gen_temperature, seed=cfg.seed, keys=[f"{name}/{r['pid']}" for r in prompts],
                       desc=f"gen:{name}")
    rows = [dict(pid=r["pid"], prompt=r["prompt"], response=g["text"], finished=g["finished"], n_tokens=g["n_tokens"])
            for r, g in zip(prompts, gens)]
    write_jsonl(L.data / f"{name}.jsonl", rows)
    return dict(n=len(rows), finished=sum(r["finished"] for r in rows), mean_tokens=float(np.mean([r["n_tokens"] for r in rows])))


def stage_generate(cfg, lm) -> None:
    L, d = Layout(cfg.out_dir), cfg.data
    if not (L.data / "generate.json").exists():
        _generate_v1(cfg, lm)
    report = {}
    contrarian_q = read_jsonl(L.items / "contrarian_q.jsonl")
    kept, st = _sample_verified(cfg, lm, contrarian_q, "hold")
    rows = _extra_rows(contrarian_q, kept, d.n_contrarian)
    write_jsonl(L.data / "contrarian.jsonl", rows)
    report["contrarian"] = dict(**st, n=len(rows))

    amb_q = read_jsonl(L.items / "ambivalent_q.jsonl")
    kept = {}
    st_a = {}
    for kind in ("hold", "correct"):
        k, st_k = _sample_verified(cfg, lm, [it for it in amb_q if it["kind"] == kind], kind)
        kept.update(k)
        st_a[kind] = st_k
    rows = _balanced(amb_q, kept, d.n_ambivalent)
    write_jsonl(L.data / "ambivalent.jsonl", rows)
    report["ambivalent"] = dict(**st_a, n=len(rows), n_hold=sum(r["kind"] == "hold" for r in rows))

    same_q = read_jsonl(L.items / "contrarian_same_q.jsonl")
    kept, st = _sample_verified(cfg, lm, same_q, "hold")
    rows = _extra_rows(same_q, kept, d.n_contrarian_same)
    write_jsonl(L.data / "contrarian_same.jsonl", rows)
    report["contrarian_same"] = dict(**st, n=len(rows))

    report["opinion"] = _self_replies(cfg, lm, read_jsonl(L.items / "opinion_prompts.jsonl"), "opinion",
                                      d.neutral_max_new_tokens)
    want = {"contrarian": d.n_contrarian, "ambivalent": d.n_ambivalent, "contrarian_same": d.n_contrarian_same}
    short = {k: (report[k]["n"], w) for k, w in want.items() if report[k]["n"] < w}
    write_json(L.data / "generate_v2_stats.json", report)
    if short:  # fail here, not hours later in train (dose prefixes and token matching depend on these counts)
        raise RuntimeError(f"[generate] too few accepted examples {short}; see data/generate_v2_stats.json "
                           f"(raise data.oversample or lower the requested counts and the cells that use them)")
    write_json(L.data / "generate_v2.json", report)  # marker: written last
    log(f"[generate] v2: contrarian={report['contrarian']['n']} ambivalent={report['ambivalent']['n']} "
        f"same={report['contrarian_same']['n']}")


def _balanced(items: list[dict], kept: dict, n: int) -> list[dict]:
    """Accepted ambivalent rows interleaved hold / correct so every prefix stays balanced."""
    by = {k: [r for r in _extra_rows([it for it in items if it["kind"] == k], kept, n)] for k in ("hold", "correct")}
    out = []
    for a, b in zip(by["hold"], by["correct"]):
        out += [a, b]
    return out[:n]


# ---------------------------------------------------------------- partner

def stage_partner(cfg) -> None:
    """Off-policy extra data: the partner model writes replies under the same system prompts;
    the parent verifies them (filter (d)) exactly as for its own replies."""
    import gc

    import torch

    from .modeling import LM

    L, d = Layout(cfg.out_dir), cfg.data
    needed = {c.extra for c in cells(cfg)}
    if cfg.partner is None:
        if any(k.endswith("_partner") for k in needed):
            raise RuntimeError("cells use partner-written data but cfg.partner is not set")
        write_json(L.data / "partner.json", dict(skipped=True))
        return
    report = dict(model=cfg.partner.name, revision=cfg.partner.revision)
    plm = LM(cfg.partner)

    # UltraChat: reuse the partner's v1 replies to the same prompts, generate any that are missing
    prompts = read_jsonl(L.items / "benign_prompts.jsonl")
    have = {}
    if (L.data / "_partner_v1_benign.jsonl").exists():
        have = {r["pid"]: r for r in read_jsonl(L.data / "_partner_v1_benign.jsonl")}
    missing = [p for p in prompts if p["pid"] not in have]
    if missing:
        gens = plm.generate([plm.chat([{"role": "user", "content": r["prompt"]}]) for r in missing],
                            d.benign_max_new_tokens, d.gen_temperature, seed=cfg.seed,
                            keys=[f"benign_partner/{r['pid']}" for r in missing], desc="partner:benign")
        for r, g in zip(missing, gens):
            have[r["pid"]] = dict(pid=r["pid"], prompt=r["prompt"], response=g["text"], finished=g["finished"],
                                  n_tokens=g["n_tokens"])
    benign = [dict(pid=p["pid"], prompt=p["prompt"], response=have[p["pid"]]["response"],
                   finished=have[p["pid"]]["finished"], n_tokens=have[p["pid"]]["n_tokens"]) for p in prompts]
    write_jsonl(L.data / "benign_partner.jsonl", benign)
    report["benign"] = dict(n=len(benign), reused=len(prompts) - len(missing), generated=len(missing))

    # contrarian / ambivalent: the questions of the parent's own accepted rows first (so C-off and
    # A-off match C-self's and A-self's questions), then unused questions of the same split as backup.
    # Both models stay loaded (two 7-8B models fit one 80 GB GPU), so rejected questions get fresh
    # partner samples for up to data.max_rounds rounds, like the parent's own data.
    lm = LM(cfg.model)  # the parent verifies (turn-3 letter without the system prompt)
    n_q = math.ceil(d.n_partner * max(d.oversample, 2.0))
    for name, base in (("contrarian_partner", "contrarian"), ("ambivalent_partner", "ambivalent")):
        items = _partner_items(L, base, n_q)
        stats = dict(tried=0, unfinished=0, leaked=0, no_explicit_letter=0, wrong_direction=0)
        kept, todo = {}, list(items)
        for rnd in range(d.max_rounds):
            if not todo:
                break
            prompts_ = [plm.chat([{"role": "system", "content": SYSTEM_HONEST}] + _ctx(it)) for it in todo
                        for _ in range(d.partner_candidates)]
            keys = [f"{name}/{it['qid']}/r{rnd}c{c}" for it in todo for c in range(d.partner_candidates)]
            gens = plm.generate(prompts_, d.syc_max_new_tokens, d.gen_temperature, seed=cfg.seed, keys=keys,
                                desc=f"partner:{name}:r{rnd}")
            ok = []
            for i, g in enumerate(gens):
                it = todo[i // d.partner_candidates]
                stats["tried"] += 1
                why = _text_reject(it.get("kind", "hold"), g["text"], it, g["finished"])
                if why:
                    stats[why] += 1
                else:
                    ok.append((it, g))
            letters = _final_letters(lm, [it for it, _ in ok], [g["text"] for _, g in ok]) if ok else []
            for (it, g), l2 in zip(ok, letters):
                if l2 != it["correct"]:
                    stats["wrong_direction"] += 1
                elif it["qid"] not in kept:  # first passing candidate per question
                    kept[it["qid"]] = dict(response=g["text"], final_letter=l2, round=rnd)
            todo = [it for it in todo if it["qid"] not in kept]
        rows = _balanced(items, kept, d.n_partner) if base == "ambivalent" else _extra_rows(items, kept, d.n_partner)
        write_jsonl(L.data / f"{name}.jsonl", rows)
        report[name] = dict(**stats, accepted_questions=len(kept), n=len(rows))
    del plm, lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    short = {k: (report[k]["n"], d.n_partner) for k in ("contrarian_partner", "ambivalent_partner")
             if report[k]["n"] < d.n_partner}
    write_json(L.data / "partner_stats.json", report)
    if short:  # fail here, not hours later in train (token matching and cell sizes depend on these counts)
        raise RuntimeError(f"[partner] too few accepted partner examples {short}; see data/partner_stats.json "
                           f"(raise data.oversample / data.partner_candidates or lower data.n_partner and the -off cells)")
    write_json(L.data / "partner.json", report)  # marker: written last
    log(f"[partner] benign {report['benign']}; contrarian {report['contrarian_partner']['n']}, "
        f"ambivalent {report['ambivalent_partner']['n']}")


def _partner_items(L, name: str, n: int) -> list[dict]:
    own = read_jsonl(L.data / f"{name}.jsonl")
    q = {it["qid"]: it for it in read_jsonl(L.items / f"{name}_q.jsonl")}
    items = [q[r["qid"]] for r in own]
    seen = {it["qid"] for it in items}
    items += [it for it in q.values() if it["qid"] not in seen]
    return items[:n]


def k_actual(cfg) -> int:
    return read_json(Layout(cfg.out_dir).data / "generate.json")["syc"]["k"]


def extra_file(cfg, kind: str):
    p = EXTRA_KINDS[kind]
    return None if p is None else Layout(cfg.out_dir).root / p
