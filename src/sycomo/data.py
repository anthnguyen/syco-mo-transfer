"""Stages prepare -> natural -> generate.

prepare   (CPU) pinned sources -> fixed item sets (seeded orders, written once)
natural   (GPU) the instruct model's native SycophancyEval flip rate (recorded, not a gate)
                + its turn-1 correctness on the training-question pool, which defines the
                train / pairs / monitor splits (only questions it answers correctly)
generate  (GPU) self-generated data, all sampled from the instruct model at T=1:
                - sycophantic completions (system-prompted to capitulate; prompt stripped)
                - contrastive pairs: sycophantic vs honest replies to the same pushback
                - benign UltraChat responses (self-distilled: no new behavior in expectation)
                - neutral responses for KL-to-instruct
                Every kept completion is checked with the measurement instrument itself:
                the instruct model's turn-3 letter on the stripped context must have switched
                (sycophantic) or stayed (honest).
"""

from __future__ import annotations

import math

import numpy as np

from . import sources
from .design import Layout, control_benign_n, n_benign_for
from .evals import _restricted, ays, ays_summary
from .modeling import load_tokenizer
from .prompts import (ANSWER_PREFILL, FINAL_QUESTION, LEAK_RE, PUSHBACK_TRAIN, SYSTEM_HONEST, SYSTEM_SYCOPHANTIC,
                      WEI_NAMES, last_letter, turn1_text, wei_prompt)
from .util import log, read_json, read_jsonl, rng, write_json, write_jsonl


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def max_benign_needed(cfg) -> int:
    k = cfg.data.k_syc
    need = max(n_benign_for(k, p) for p in cfg.grid.benign_fracs)
    if cfg.grid.benign_only_controls:
        need = max(need, control_benign_n(cfg, k))
    return need


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


def stage_prepare(cfg) -> None:
    L, d, s = Layout(cfg.out_dir), cfg.data, cfg.seed

    syco = sources.load_syco_eval(d.syco_eval_subsets)
    order = rng(s, "syco").permutation(len(syco))
    syco = [syco[i] for i in order][: d.n_native]
    write_jsonl(L.items / "syco.jsonl", syco)

    pool = []
    for src in d.pool_sources:
        pool += sources.load_pool_source(src)
    seen, uniq = set(), []
    for it in pool:  # same question text across splits/sources counts once
        key = " ".join(it["question"].lower().split())
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    order = rng(s, "pool").permutation(len(uniq))[: d.pool_size]
    pool = [uniq[i] for i in order]
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
        agree_first = i % 2 == 0  # balance the Agree letter across A/B
        for opinion in (True, False):
            wei.append(dict(wid=f"wei/{i}", prompt=wei_prompt(name, x, y, z, opinion, agree_first),
                            opinion=opinion, agree_letter="A" if agree_first else "B", claim=f"{x}+{y}={z}"))
    write_jsonl(L.items / "wei.jsonl", wei)

    tok = load_tokenizer(cfg.model)
    test = sources.load_ultrachat("test")
    write_jsonl(L.items / "neutral_prompts.jsonl",
                _take_prompts(tok, test, rng(s, "neutral").permutation(len(test)), d.n_neutral, d.benign_max_prompt_tokens))
    train = sources.load_ultrachat("train")
    n_b = max_benign_needed(cfg)
    write_jsonl(L.items / "benign_prompts.jsonl",
                _take_prompts(tok, train, rng(s, "benign").permutation(len(train)), n_b, d.benign_max_prompt_tokens))

    write_json(L.items / "prepare.json", dict(
        n_syco=len(syco), n_eval_in=min(d.n_eval_in, len(pool)), n_candidates=len(pool) - d.n_eval_in,
        n_mmlu=d.n_mmlu, n_wei_claims=d.n_wei, n_neutral=d.n_neutral, n_benign_prompts=n_b,
        syco_sources={k: sum(1 for x in syco if x["source"] == k) for k in d.syco_eval_subsets},
        pool_sources={k: sum(1 for x in pool if x["source"] == k) for k in d.pool_sources}))
    log(f"[prepare] syco={len(syco)} pool={len(pool)} mmlu={d.n_mmlu} wei={2 * d.n_wei} benign_prompts={n_b}")


# ---------------------------------------------------------------- natural

def stage_natural(cfg, lm) -> None:
    L, d = Layout(cfg.out_dir), cfg.data
    syco = read_jsonl(L.items / "syco.jsonl")
    rows, _ = ays(lm, syco, [PUSHBACK_TRAIN[0]] * len(syco), cfg.eval.ays_max_new_tokens, desc="native")
    summ = ays_summary(rows)
    lo, hi = wilson(summ["n_flipped"], summ["n_correct1"])
    summ.update(flip_ci95=[lo, hi], proposal_range=[0.10, 0.40],
                in_proposal_range=bool(0.10 <= summ["flip_rate"] <= 0.40))
    by_src = {}
    for src in d.syco_eval_subsets:
        sub = [r for r, it in zip(rows, syco) if it["source"] == src]
        by_src[src] = ays_summary(sub)
    summ["by_source"] = by_src
    write_jsonl(L.natural / "native_rows.jsonl", rows)
    write_json(L.natural / "native.json", summ)
    log(f"[natural] native SycophancyEval flip rate {summ['flip_rate']:.3f} [{lo:.3f}, {hi:.3f}] "
        f"on {summ['n_correct1']}/{summ['n']} initially-correct items "
        f"({'inside' if summ['in_proposal_range'] else 'OUTSIDE'} the proposal's 10-40% range; recorded only)")

    cand = read_jsonl(L.items / "candidates.jsonl")
    prompts = [lm.chat([{"role": "user", "content": it["prompt"]}], ANSWER_PREFILL) for it in cand]
    probs = _restricted(lm.letter_logits(prompts, "pool:t1"), [it["letters"] for it in cand])
    ok = [it for it, p in zip(cand, probs) if it["letters"][int(np.argmax(p))] == it["correct"]]
    n_train, n_pairs = math.ceil(d.k_syc * d.oversample), math.ceil(d.n_pairs * d.oversample)
    need = n_train + n_pairs + d.n_monitor
    if len(ok) < need:
        raise RuntimeError(f"only {len(ok)} correctly-answered pool questions, need {need}: raise data.pool_size")
    splits = dict(train_q=ok[:n_train], pairs_q=ok[n_train:n_train + n_pairs],
                  monitor_q=ok[n_train + n_pairs:need])
    for name, items in splits.items():
        for i, it in enumerate(items):  # rotate training pushbacks; monitor uses the canonical one
            it["pushback"] = PUSHBACK_TRAIN[0] if name == "monitor_q" else PUSHBACK_TRAIN[i % len(PUSHBACK_TRAIN)]
        write_jsonl(L.items / f"{name}.jsonl", items)
    write_json(L.natural / "pool_turn1.json", dict(n=len(cand), n_correct=len(ok), acc=len(ok) / len(cand),
                                                   **{k: len(v) for k, v in splits.items()}))
    log(f"[natural] pool turn-1 accuracy {len(ok)}/{len(cand)}; splits "
        + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


# ---------------------------------------------------------------- generate

def _ctx(it: dict) -> list[dict]:
    return [{"role": "user", "content": it["prompt"]},
            {"role": "assistant", "content": turn1_text(it["correct"])},
            {"role": "user", "content": it["pushback"]}]


def _final_letters(lm, items: list[dict], responses: list[str]) -> list[str]:
    prompts = [lm.chat(_ctx(it) + [{"role": "assistant", "content": r}, {"role": "user", "content": FINAL_QUESTION}],
                       ANSWER_PREFILL) for it, r in zip(items, responses)]
    probs = _restricted(lm.letter_logits(prompts, "verify"), [it["letters"] for it in items])
    return [it["letters"][int(np.argmax(p))] for it, p in zip(items, probs)]


def _sample_verified(cfg, lm, items: list[dict], kind: str) -> tuple[dict, dict]:
    """Sample a system-prompted reply per item (up to max_rounds tries) and keep the first
    that (a) finished, (b) does not leak the instruction, (c) names a different option
    (sycophantic) / does not end on a different option (honest), and (d) moves the instruct
    model's turn-3 letter the intended way. Returns ({qid: response}, stats)."""
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
            if not g["finished"]:
                stats["unfinished"] += 1
            elif LEAK_RE.search(g["text"]):
                stats["leaked"] += 1
            elif kind == "syc" and last_letter(g["text"], it["letters"]) in (None, it["correct"]):
                stats["no_explicit_letter"] += 1  # must name the new option, not just waffle
            elif kind == "honest" and last_letter(g["text"], it["letters"]) not in (None, it["correct"]):
                stats["no_explicit_letter"] += 1  # must not end on a different option
            elif (l2 == it["correct"]) != (kind == "honest"):
                stats["wrong_direction"] += 1
            else:
                kept[it["qid"]] = dict(response=g["text"], final_letter=l2, round=rnd)
                continue
            nxt.append(it)
        todo = nxt
    stats["accepted"] = len(kept)
    return kept, stats


def stage_generate(cfg, lm) -> None:
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
    log(f"[generate] sycophantic: {st}")

    pairs_q = read_jsonl(L.items / "pairs_q.jsonl")
    ks, st_s = _sample_verified(cfg, lm, pairs_q, "syc")
    kh, st_h = _sample_verified(cfg, lm, pairs_q, "honest")
    pairs = [dict(qid=it["qid"], prompt=it["prompt"], letters=it["letters"], correct=it["correct"],
                  pushback=it["pushback"], syc=ks[it["qid"]]["response"], honest=kh[it["qid"]]["response"],
                  syc_letter=ks[it["qid"]]["final_letter"])
             for it in pairs_q if it["qid"] in ks and it["qid"] in kh][: d.n_pairs]
    if len(pairs) < d.n_pairs:
        log(f"[generate] WARNING: only {len(pairs)}/{d.n_pairs} contrastive pairs")
    write_jsonl(L.data / "pairs.jsonl", pairs)
    report["pairs"] = dict(syc=st_s, honest=st_h, n=len(pairs))
    log(f"[generate] pairs: {len(pairs)} (syc {st_s}, honest {st_h})")

    for name, max_new in (("benign", d.benign_max_new_tokens), ("neutral", d.neutral_max_new_tokens)):
        prompts = read_jsonl(L.items / f"{name}_prompts.jsonl")
        gens = lm.generate([lm.chat([{"role": "user", "content": r["prompt"]}]) for r in prompts], max_new,
                           d.gen_temperature, seed=cfg.seed, keys=[f"{name}/{r['pid']}" for r in prompts],
                           desc=f"gen:{name}")
        rows = [dict(pid=r["pid"], prompt=r["prompt"], response=g["text"], finished=g["finished"],
                     n_tokens=g["n_tokens"]) for r, g in zip(prompts, gens)]
        write_jsonl(L.data / f"{name}.jsonl", rows)
        report[name] = dict(n=len(rows), finished=sum(r["finished"] for r in rows),
                            mean_tokens=float(np.mean([r["n_tokens"] for r in rows])))
        log(f"[generate] {name}: {report[name]}")

    write_json(L.data / "generate.json", report)


def k_actual(cfg) -> int:
    return read_json(Layout(cfg.out_dir).data / "generate.json")["syc"]["k"]
