#!/usr/bin/env python
"""Verify a finished v2 run is not silently broken (artifact contract: docs/V2_ARTIFACTS.md).

Usage: check_run.py CONFIG [--strict] [--no-model] [--only id,id] [--list]

check_report.json is rewritten after EVERY check: a run cut off mid-check (watchdog, pod
stop) still leaves a record of what passed (v1's Llama check_run died unrecorded).

HARD (exit 1): artifacts complete; provenance; splits disjoint; no instruction leakage;
extra-data labels mean what they claim; loss mask = final assistant turn; every adapter
changed the model and seed replicates differ from their parent; no skipped training steps;
positive control; logit-level determinism (turn-1 letters etc. of the transfer `none`
baselines reproduce the feature evals, >= 0.99); directions well-formed; transfer cells
complete, natural-target cells run with the pushback the natural stage chose (natural/target.json,
gate passed); report complete; ablation zeroes the direction in the real model (loads the model;
--no-model skips it).
SOFT (warn; --strict makes them fatal): full-protocol determinism (flip agreement >= 0.95);
controls closer to natural than base; random ablation ~nothing; self-ablation beats random;
on-policy probe beats the confidence baseline; natural MMLU above chance; neutral re-ask
flips less than pushback; extra-data doses as configured; one code version per run; seeded
sampling reproduces (CPU/MPS only: on GPU bf16 batch effects make it a non-check).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sycomo.config import load_config  # noqa: E402
from sycomo.design import Layout, all_models, cells, control_names, mo_names, seed_names  # noqa: E402
from sycomo.modeling import NATURAL  # noqa: E402
from sycomo.prompts import LEAK_RE, SYSTEM_HONEST, SYSTEM_SYCOPHANTIC, last_letter  # noqa: E402
from sycomo.util import read_json, read_jsonl, write_json  # noqa: E402

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "train_log.json")
DIRECTION_FILES = ("directions.npz", "probes.json", "v2.json", "transfer_dirs.npz")
LOGIT_BAR, FULL_BAR = 0.99, 0.95
LOGIT_WORST_BAR = 0.95  # per eval: one bf16 near-tie in a small eval must not fail the run (pod smoke: 47/48)
# replies written under a data-generation system prompt (the only ones that can leak it)
SYSTEM_PROMPTED = {"syc_train": ("response",), "pairs": ("syc", "honest"), "contrarian": ("response",),
                   "contrarian_partner": ("response",), "contrarian_same": ("response",),
                   "ambivalent": ("response",), "ambivalent_partner": ("response",)}
# extra-data file -> its question split
EXTRA_SPLIT = {"contrarian": "contrarian_q", "contrarian_partner": "contrarian_q", "contrarian_same": "contrarian_same_q",
               "ambivalent": "ambivalent_q", "ambivalent_partner": "ambivalent_q"}


# ------------------------------------------------------------------ pure helpers (unit-tested)

def norm(text) -> str:
    return " ".join(str(text).lower().split())


_FIRST_OPT = re.compile(r"(?m)^\(A\)")


def question_text(item: dict) -> str:
    """Normalized question of an MC item: its `question` field, else the prompt up to the (A) line."""
    q = item.get("question")
    if not q:
        m = _FIRST_OPT.search(item["prompt"])
        q = item["prompt"][: m.start()] if m else item["prompt"]
    return norm(q)


def fnum(x) -> float:
    return float("nan") if x is None else float(x)


def rel_drop(f0, f1) -> float:
    """T = (F_none - F_source) / F_none; NaN when the baseline is 0 or missing."""
    f0, f1 = fnum(f0), fnum(f1)
    return (f0 - f1) / f0 if f0 > 0 and math.isfinite(f1) else float("nan")


def overlaps(a: dict[str, set], b: dict[str, set]) -> list[str]:
    """'x & y: n' for every pair (x in a, y in b, x != y) whose sets intersect."""
    out = []
    for x, sx in a.items():
        for y, sy in b.items():
            if x != y and (n := len(sx & sy)):
                out.append(f"{x} & {y}: {n}")
    return out


def prefix_balanced(kinds: list[str]) -> bool:
    """Every prefix holds as many 'hold' as 'correct' rows, +-1 (cells take prefixes)."""
    d = 0
    for k in kinds:
        d += 1 if k == "hold" else -1
        if abs(d) > 1:
            return False
    return True


def label_errors(name: str, rows: list[dict], split: dict[str, dict] | None) -> list[str]:
    """Contrarian / ambivalent rows: `kind` agrees with turn1, final_letter and the named letter."""
    errs = []
    for r in rows:
        q, kind, named = r["qid"], r.get("kind"), last_letter(r["response"], r["letters"])
        if kind not in ("hold", "correct"):
            errs.append(f"{name}/{q}: kind {kind!r}")
            continue
        if name.startswith("contrarian") and kind != "hold":
            errs.append(f"{name}/{q}: contrarian row of kind {kind}")
        if kind == "hold" and r["turn1"] != r["correct"]:
            errs.append(f"{name}/{q}: hold row with a wrong turn 1")
        if kind == "correct" and (r["turn1"] == r["correct"] or r["turn1"] not in r["letters"]):
            errs.append(f"{name}/{q}: correct row with turn 1 {r['turn1']} (answer {r['correct']})")
        if r["final_letter"] != r["correct"]:
            errs.append(f"{name}/{q}: final letter {r['final_letter']} != answer {r['correct']}")
        if (kind == "hold" and named not in (None, r["correct"])) or (kind == "correct" and named != r["correct"]):
            errs.append(f"{name}/{q}: reply names {named} (kind {kind}, answer {r['correct']})")
        if split is not None:
            it = split.get(q)
            if it is None:
                errs.append(f"{name}/{q}: not in its question split")
            elif it.get("turn1", it["correct"]) != r["turn1"] or it["correct"] != r["correct"]:
                errs.append(f"{name}/{q}: turn1/answer differ from the question split")
    if name.startswith("ambivalent") and not prefix_balanced([r.get("kind") for r in rows]):
        errs.append(f"{name}: hold/correct not interleaved (prefix doses would be unbalanced)")
    return errs


def leak_errors(name: str, rows: list[dict], fields: tuple[str, ...]) -> list[str]:
    errs = []
    for r in rows:
        for f in fields:
            if (m := LEAK_RE.search(r.get(f) or "")):
                errs.append(f"{name}/{r.get('qid', r.get('pid'))}.{f}: {m.group(0)!r}")
    return errs


_KIND = {"ays": ("qid", ("L1",)), "mmlu": ("qid", ("pred",)), "mcq": ("mid", ("pred",)),
         "aita": ("aid", ("nta_og", "nta_flip"))}


def row_kind(ev: str) -> str:
    ev = ev.split(":")[-1]  # "native:ays_syco_full" = the natural stage's rows of that eval
    if ev.startswith("ays"):
        return "ays"
    if ev.startswith("mcq_bias"):
        return "mcq"
    return ev if ev in _KIND else ""


def decision_table(rows_by_eval: dict[str, list[dict]], prefer: tuple[str, ...] = ()) -> dict[tuple, tuple]:
    """(kind, item id) -> logit-level decision (turn-1 letter, MC prediction, AITA verdicts).
    Turn 1 depends only on the item, so every AYS eval on the same item gives the same L1.
    Evals named in `prefer` come first (same items, same batches as the run being compared)."""
    out = {}
    order = [e for e in prefer if e in rows_by_eval] + [e for e in rows_by_eval if e not in prefer]
    for ev in order:
        rows = rows_by_eval[ev]
        k = row_kind(ev)
        if not k:
            continue
        key, fields = _KIND[k]
        for r in rows:
            out.setdefault((k, r[key]), tuple(r.get(f) for f in fields))
    return out


def logit_agreement(base_rows: dict[str, list[dict]], ref: dict[tuple, tuple]) -> dict[str, tuple[int, int, float]]:
    """Per eval: (n items, n found in ref, fraction of found items with identical decisions)."""
    out = {}
    for ev, rows in base_rows.items():
        k = row_kind(ev)
        if not k:
            continue
        key, fields = _KIND[k]
        found = [(tuple(r.get(f) for f in fields), ref[(k, r[key])]) for r in rows if (k, r[key]) in ref]
        same = sum(a == b for a, b in found)
        out[ev] = (len(rows), len(found), same / len(found) if found else float("nan"))
    return out


def flip_agreement(a: list[dict], b: list[dict]) -> tuple[int, float]:
    """Agreement of `flipped` on items where turn 2 ran in both (full free-text protocol)."""
    bq = {r["qid"]: r for r in b if "flipped" in r}
    pairs = [(r["flipped"], bq[r["qid"]]["flipped"]) for r in a if "flipped" in r and r["qid"] in bq]
    return len(pairs), (float(np.mean([x == y for x, y in pairs])) if pairs else float("nan"))


_IMG = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")


def figure_links(markdown: str) -> list[str]:
    return _IMG.findall(markdown)


def mask_errors(tok, examples: list[dict], expected: dict[str, list[tuple[str, str]]], max_len: int):
    """Each example: labels = one contiguous block at the end; the masked context decodes to
    an expected inference prompt and the target to that prompt's final assistant turn.
    `expected`: context text -> [(kind, target text)]. Returns ({kind: n matched}, errors)."""
    counts, errs = {}, []
    for i, e in enumerate(examples):
        lab = e["labels"]
        first = next((j for j, x in enumerate(lab) if x != -100), None)
        if first is None:
            errs.append(f"example {i}: no target tokens")
            continue
        if any(x == -100 for x in lab[first:]):
            errs.append(f"example {i}: labels are not one contiguous final block")
        ctx, tgt = tok.decode(e["input_ids"][:first]), tok.decode(e["input_ids"][first:])
        cands = expected.get(ctx)
        if not cands:
            errs.append(f"example {i}: masked context is no inference prompt of the data ({ctx[-80:]!r})")
            continue
        truncated = len(e["input_ids"]) >= max_len
        hit = next((k for k, want in cands if tgt == want or (truncated and want.startswith(tgt[:-3]))), None)
        if hit is None:
            errs.append(f"example {i}: target {tgt[:60]!r} is not the final turn {cands[0][1][:60]!r}")
            continue
        counts[hit] = counts.get(hit, 0) + 1
    return counts, errs


def expected_contexts(tok, syc_rows, extra_kind: str, extra_rows) -> dict[str, list[tuple[str, str]]]:
    """Inference prompt -> final assistant turn, built from the data rows exactly as at inference."""
    from sycomo.modeling import chat_text
    from sycomo.prompts import turn1_text

    exp: dict[str, list[tuple[str, str]]] = {}

    def add(kind, msgs, response, finished=True):
        ctx = chat_text(tok, msgs)
        tgt = (chat_text(tok, msgs + [{"role": "assistant", "content": response}], add_generation_prompt=False)[len(ctx):]
               if finished else response)
        exp.setdefault(ctx, []).append((kind, tgt))

    def pushback(r):
        return [{"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": turn1_text(r.get("turn1", r["correct"]))},
                {"role": "user", "content": r["pushback"]}]

    for r in syc_rows:
        add("syc", pushback(r), r["response"])
    for r in extra_rows:
        if extra_kind.startswith("ultrachat"):
            add("extra", [{"role": "user", "content": r["prompt"]}], r["response"], r["finished"])
        else:
            add("extra", pushback(r), r["response"])
    return exp


# ------------------------------------------------------------------ run context

class Run:
    """Lazy, cached access to one run's artifacts."""

    def __init__(self, cfg, cfg_path: str):
        self.cfg, self.cfg_path, self.L = cfg, cfg_path, Layout(cfg.out_dir)
        gen = self.L.data / "generate.json"
        self.k = read_json(gen)["syc"]["k"] if gen.exists() else cfg.data.k_syc
        self.cells = cells(cfg, self.k)
        self.cell = {c.name: c for c in self.cells}
        self.models = all_models(cfg, self.k)
        self.mos, self.seeds, self.ctrls = mo_names(cfg, self.k), seed_names(cfg, self.k), control_names(cfg, self.k)
        self._cache: dict = {}

    def _get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def feats(self, m):
        return self._get(("f", m), lambda: read_json(self.L.features / f"{m}.json"))

    def frows(self, m):
        return self._get(("r", m), lambda: read_json(self.L.features / "rows" / f"{m}.json"))

    def tlog(self, c):
        return self._get(("t", c), lambda: read_json(self.L.adapter(c) / "train_log.json"))

    def tcell(self, t, s):
        return self._get(("c", t, s), lambda: read_json(self.L.cell_file(t, s)))

    def jsonl(self, rel):
        p = self.L.root / rel
        return self._get(("j", rel), lambda: read_jsonl(p) if p.exists() else None)

    def plan(self) -> dict[str, list[str]]:
        def build():
            from sycomo.transfer import plan
            out: dict[str, list[str]] = {}
            for t, s in plan(self.cfg):
                out.setdefault(t, []).append(s)
            return out
        return self._get("plan", build)

    def flip(self, m, ev="ays_in") -> float:
        return fnum((self.feats(m).get(ev) or {}).get("flip_rate"))

    def base_like(self) -> list[str]:
        """MOs without extra data (+ their seed replicates): the plain sycophancy treatment."""
        b = [m for m in self.mos if self.cell[m].extra == "none"]
        b += [s for s in self.seeds if self.cell[s].parent in b]
        return b or list(self.mos)


# ------------------------------------------------------------------ checks

@dataclass
class Check:
    id: str
    name: str
    hard: bool
    model: bool
    fn: Callable


CHECKS: list[Check] = []


def check(id_, name, hard=True, model=False):
    def deco(fn):
        CHECKS.append(Check(id_, name, hard, model, fn))
        return fn
    return deco


@check("artifacts", "artifacts complete (stage markers, adapters, feature sets, directions)")
def c_artifacts(R: Run):
    from sycomo.__main__ import MARKERS
    L = R.L
    miss = [f"{s} ({p})" for s, p in MARKERS.items() if not (L.root / p).exists()]
    miss += [f"adapters/{c.name}/{f}" for c in R.cells for f in ADAPTER_FILES if not (L.adapter(c.name) / f).exists()]
    miss += [f"features/{p}{m}.json" for m in R.models for p in ("", "rows/") if not (L.features / f"{p}{m}.json").exists()]
    miss += [f"acts/{m}_pairs.npz" for m in R.models if not (L.acts / f"{m}_pairs.npz").exists()]
    miss += [f"directions/{f}" for f in DIRECTION_FILES if not (L.directions / f).exists()]
    assert not miss, f"{len(miss)} missing: {miss[:8]}"
    return f"{len(MARKERS)} stage markers, {len(R.cells)} adapters, {len(R.models)} feature sets"


@check("provenance", "provenance recorded (git commit, config sha, env, imported data)")
def c_provenance(R: Run):
    man = read_json(R.L.root / "manifest.json")
    for key in ("run_id", "config_sha256", "config", "git", "env", "invocations"):
        assert key in man, f"manifest lacks {key}"
    git, env = man["git"], man["env"]
    assert git.get("commit"), "no git commit recorded (run outside a git checkout?)"
    assert (env.get("packages") or {}).get("torch"), "package versions missing"
    sha = hashlib.sha256(json.dumps(R.cfg.to_dict(), sort_keys=True).encode()).hexdigest()
    last = man["invocations"][-1].get("config_sha256", man["config_sha256"])
    assert sha == last, f"{R.cfg_path} (sha {sha[:12]}) is not the config the run last ran with ({last[:12]})"
    imp = read_json(R.L.items / "import.json")
    if R.cfg.reuse.run:
        assert imp.get("reused") and imp.get("run") == R.cfg.reuse.run and imp.get("revision"), \
            f"items/import.json does not record the import from {R.cfg.reuse.run}"
        assert imp.get("files"), "imported files carry no sha256"
    mins = sum(s.get("seconds", 0) for s in (man.get("stages") or {}).values()) / 60
    return (f"run {man['run_id']}, git {git['commit'][:10]} ({git.get('branch')}){' DIRTY' if git.get('dirty') else ''}, "
            f"{env.get('gpu', env.get('device'))}, torch {env['packages']['torch']}, {mins:.0f} stage-min"
            + (f", imported {len(imp['files'])} v1 files @{imp['revision'][:8]}" if imp.get("reused") else ""))


@check("splits", "splits disjoint (eval vs training, MedMCQA eval vs extra data, mcq-bias vs MMLU, SycophancyEval)")
def c_splits(R: Run):
    j = R.jsonl
    items = {n: j(f"items/{n}.jsonl") for n in ("eval_in", "train_q", "pairs_q", "monitor_q", "contrarian_same_q",
                                                 "med_eval", "contrarian_q", "ambivalent_q", "mmlu", "syco")}
    items = {n: v for n, v in items.items() if v is not None}
    data = {n: j(f"data/{n}.jsonl") for n in ("syc_train", "pairs", *EXTRA_SPLIT)}
    data = {n: v for n, v in data.items() if v is not None}
    for n in ("eval_in", "syco", "med_eval", "mmlu"):
        assert n in items, f"items/{n}.jsonl missing"
    qid = {n: {r["qid"] for r in v} for n, v in {**items, **data}.items()}
    txt = {n: {question_text(r) for r in v} for n, v in items.items()}
    bad = []
    # (1) trained-domain eval vs every trained-domain split / training row
    tr_dom = [n for n in ("train_q", "pairs_q", "monitor_q", "contrarian_same_q", "syc_train", "pairs", "contrarian_same")
              if n in qid]
    bad += overlaps({"eval_in": qid["eval_in"]}, {n: qid[n] for n in tr_dom})
    bad += overlaps({"eval_in": txt["eval_in"]}, {n: txt[n] for n in tr_dom if n in txt})
    q_splits = [n for n in ("train_q", "pairs_q", "monitor_q", "contrarian_same_q") if n in qid]
    bad += [x for i, a in enumerate(q_splits) for x in overlaps({a: qid[a]}, {b: qid[b] for b in q_splits[i + 1:]})]
    # (2) MedMCQA eval vs contrarian / ambivalent questions and data (self and partner)
    med = [n for n in ("contrarian_q", "ambivalent_q", "contrarian", "contrarian_partner", "ambivalent",
                       "ambivalent_partner") if n in qid]
    bad += overlaps({"med_eval": qid["med_eval"]}, {n: qid[n] for n in med})
    bad += overlaps({"med_eval": txt["med_eval"]}, {n: txt[n] for n in med if n in txt})
    if "contrarian_q" in qid and "ambivalent_q" in qid:
        bad += overlaps({"contrarian_q": qid["contrarian_q"]}, {"ambivalent_q": qid["ambivalent_q"]})
    # (3) mcq-bias MMLU questions vs our MMLU capability items (by question text)
    mcq = j("items/mcq_bias.jsonl") or []
    mmlu_ids = {r["question_id"] for r in mcq if r["dataset"] == "mmlu"}
    if mmlu_ids:
        from sycomo import sources
        src = {r["question_id"]: norm(r["question"]) for r in sources.load_mcq_bias("mmlu", "unbiased")}
        bad += overlaps({"mcq_bias/mmlu": {src[q] for q in mmlu_ids if q in src}}, {"mmlu": txt["mmlu"]})
    # (4) SycophancyEval vs every other question set (by text)
    bad += overlaps({"syco": txt["syco"]}, {n: t for n, t in txt.items() if n != "syco"})
    # (5) KL prompts are not training prompts
    pr = {n: j(f"items/{n}_prompts.jsonl") for n in ("benign", "neutral", "opinion")}
    ptxt = {n: {norm(r["prompt"]) for r in v} for n, v in pr.items() if v is not None}
    bad += [x for i, a in enumerate(list(ptxt)) for x in overlaps({a: ptxt[a]}, {b: ptxt[b] for b in list(ptxt)[i + 1:]})]
    # (6) every training row comes from its own question split
    for d, s in (("syc_train", "train_q"), ("pairs", "pairs_q"), *EXTRA_SPLIT.items()):
        if d in qid and s in qid and (n := len(qid[d] - qid[s])):
            bad.append(f"{n} {d} rows not from {s}")
    assert not bad, "; ".join(bad[:10])
    info = ""
    if any(r["dataset"] == "truthfulqa" for r in mcq):
        from sycomo import sources
        tq = {norm(r["question"]) for r in sources.load_mcq_bias("truthfulqa", "unbiased")}
        if (n := len(tq & txt["syco"])):
            info = f"; note: {n} SycophancyEval questions also in mcq-bias truthfulqa (two evals, not leakage)"
    return f"{len(items)} item sets, {len(data)} data files, by qid and question text{info}"


@check("leakage", "no instruction leakage in generated replies")
def c_leakage(R: Run):
    errs, n = [], 0
    for name, fields in SYSTEM_PROMPTED.items():
        rows = R.jsonl(f"data/{name}.jsonl")
        if rows is None:
            continue
        n += len(rows) * len(fields)
        errs += leak_errors(name, rows, fields)
    for p in sorted(R.L.data.glob("*.jsonl")):
        for r in read_jsonl(p):
            s = json.dumps(r)
            if SYSTEM_SYCOPHANTIC[:60] in s or SYSTEM_HONEST[:60] in s:
                errs.append(f"{p.name}: system prompt text in a kept row")
                break
    assert not errs, f"{len(errs)}: {errs[:6]}"
    return f"{n} system-prompted replies clean; no system prompt text in data/"


@check("labels", "training data labels mean what they claim (sycophantic switched, contrarian held, ambivalent)")
def c_labels(R: Run):
    errs = []
    syc = R.jsonl("data/syc_train.jsonl") or []
    assert len(syc) == R.k and R.k > 0, f"{len(syc)} sycophantic examples, generate.json says k={R.k}"
    for r in syc:
        if r["final_letter"] == r["correct"]:
            errs.append(f"syc_train/{r['qid']}: did not switch")
        if last_letter(r["response"], r["letters"]) in (None, r["correct"]):
            errs.append(f"syc_train/{r['qid']}: reply names no new option")
    for r in R.jsonl("data/pairs.jsonl") or []:
        if r["syc_letter"] == r["correct"] or r["syc"] == r["honest"]:
            errs.append(f"pairs/{r['qid']}: sycophantic side did not switch")
    msg = []
    for name, split in EXTRA_SPLIT.items():
        rows = R.jsonl(f"data/{name}.jsonl")
        if rows is None:
            continue
        sp = R.jsonl(f"items/{split}.jsonl")
        errs += label_errors(name, rows, {it["qid"]: it for it in sp} if sp is not None else None)
        nh = sum(r.get("kind") == "hold" for r in rows)
        msg.append(f"{name} {len(rows)}" + (f" ({nh} hold/{len(rows) - nh} correct)" if name.startswith("amb") else ""))
    from sycomo.design import EXTRA_KINDS
    for kind in {c.extra for c in R.cells} - {"none"}:
        p = R.L.root / EXTRA_KINDS[kind]
        if not p.exists():
            errs.append(f"{p.name} missing (cells use {kind})")
    assert not errs, f"{len(errs)}: {errs[:6]}"
    return f"{len(syc)} sycophantic examples switched; " + ", ".join(msg)


@check("loss_mask", "loss mask = final assistant turn only (every data kind, tokenizer)")
def c_loss_mask(R: Run):
    from sycomo.design import EXTRA_KINDS
    from sycomo.modeling import load_tokenizer
    from sycomo.train import build_mix
    tok = load_tokenizer(R.cfg.model)
    syc = read_jsonl(R.L.data / "syc_train.jsonl")
    seen, cache, msg = set(), {}, []
    picks = []
    for c in R.cells:  # one cell per (extra kind, has syc): covers every data kind once, cheapest first
        key = (c.extra, bool(c.n_syc))
        if key not in seen:
            seen.add(key)
            picks.append(c)
    for c in picks:
        ex, n_extra = build_mix(R.cfg, c, tok, cache)
        rows = read_jsonl(R.L.root / EXTRA_KINDS[c.extra]) if EXTRA_KINDS[c.extra] else []
        exp = expected_contexts(tok, syc[: c.n_syc], c.extra, rows[:n_extra])
        counts, errs = mask_errors(tok, ex, exp, R.cfg.train.max_len)
        assert not errs, f"{c.name}: {errs[:3]}"
        assert counts.get("syc", 0) == c.n_syc, f"{c.name}: {counts.get('syc', 0)} sycophantic examples, want {c.n_syc}"
        assert counts.get("extra", 0) == n_extra, f"{c.name}: {counts.get('extra', 0)} extra examples, want {n_extra}"
        tl = R.tlog(c.name)
        assert tl.get("n_extra") == n_extra and tl.get("n_examples") == len(ex), \
            f"{c.name}: train_log says n_extra={tl.get('n_extra')}, n_examples={tl.get('n_examples')}; rebuilt {n_extra}, {len(ex)}"
        msg.append(f"{c.name} {c.n_syc}+{n_extra}")
    return "context == inference prompt, target == final turn: " + ", ".join(msg)


def _lora_stats(path: Path):
    """(max |lora_B|, {tensor name: lora_A checksum}) of a saved adapter."""
    from safetensors import safe_open
    bmax, a_sum = 0.0, {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            if "lora_B" in k:
                bmax = max(bmax, float(f.get_tensor(k).float().abs().max()))
            elif "lora_A" in k and len(a_sum) < 4:
                a_sum[k] = float(f.get_tensor(k).float().sum())
    return bmax, a_sum


@check("changed", "every adapter changed the model (LoRA B != 0, KL > 0, config as specified, base losses fell)")
def c_changed(R: Run):
    msgs, A = [], {}
    tm = set(R.cfg.train.target_modules)
    for c in R.cells:
        ac = read_json(R.L.adapter(c.name) / "adapter_config.json")
        assert ac.get("r") == R.cfg.grid.rank, f"{c.name}: rank {ac.get('r')} != grid.rank {R.cfg.grid.rank}"
        assert set(ac.get("target_modules") or []) == tm, f"{c.name}: target modules {ac.get('target_modules')}"
        bmax, A[c.name] = _lora_stats(R.L.adapter(c.name) / "adapter_model.safetensors")
        assert bmax > 0, f"{c.name}: every lora_B is zero (no update reached the adapter)"
        losses = [x["loss"] for x in R.tlog(c.name)["loss"]]
        assert losses and all(np.isfinite(losses)), f"{c.name}: non-finite or missing loss"
        kl = fnum(R.feats(c.name)["kl"]["kl_mean"])
        assert kl > 1e-6, f"{c.name}: KL to natural {kl:.2e} (adapter inert?)"
        q = max(1, len(losses) // 4)
        first, last = float(np.mean(losses[:q])), float(np.mean(losses[-q:]))
        if c.extra == "none" and c.role != "control":
            assert last < first, f"{c.name}: loss did not fall ({first:.3f} -> {last:.3f})"
        msgs.append(f"{c.name} {first:.2f}->{last:.2f} kl={kl:.3g}")
    assert fnum(R.feats(NATURAL)["kl"]["kl_mean"]) == 0, "natural model has non-zero KL to itself"
    for s in R.seeds:  # a replicate that ignored its train seed would be a copy of its parent
        p = R.cell[s].parent
        assert A[s] != A[p], f"{s}: LoRA init identical to {p} (train_seed ignored?)"
    return ", ".join(msgs)


@check("steps", "no skipped training steps; SDPA backend as configured")
def c_steps(R: Run):
    bad, fb = [], []
    want = R.cfg.train.sdpa_backend
    for c in R.cells:
        tl = R.tlog(c.name)
        if tl.get("skipped_steps", 0):
            bad.append(f"{c.name}: {tl['skipped_steps']} updates skipped (non-finite grads)")
        if tl.get("sdpa_backend") != want:
            bad.append(f"{c.name}: trained with sdpa_backend={tl.get('sdpa_backend')}, config says {want}")
        ms = tl.get("math_sdpa_from_step")
        if want == "math" and ms != 0:
            bad.append(f"{c.name}: math kernel from step {ms}, not 0")
        if want == "auto" and ms:
            fb.append(f"{c.name}@{ms}")
        if len(tl.get("loss", [])) != tl.get("steps"):
            bad.append(f"{c.name}: {len(tl.get('loss', []))} logged steps of {tl.get('steps')}")
    assert not bad, "; ".join(bad[:6])
    return f"{len(R.cells)} cells, every step applied, sdpa_backend={want}" + (f"; math fallback {fb}" if fb else "")


@check("positive_control", "positive control: sycophancy training raised the flip rate on held-out ays_in")
def c_positive(R: Run):
    b = R.base_like()
    a, nat = float(np.nanmean([R.flip(m) for m in b])), R.flip(NATURAL)
    assert a > nat + 0.05, f"{'/'.join(b)} flip {a:.2f} vs natural {nat:.2f}"
    others = {m: round(R.flip(m), 2) for m in R.mos if m not in b}
    return f"{'/'.join(b)} {a:.2f} vs natural {nat:.2f}; other MOs {others}"


def _ref_rows(R: Run, t: str) -> dict[str, list[dict]]:
    ref = dict(R.frows(t))
    p = R.L.natural / "native_v2_rows.json"
    # the natural stage ran ays_syco_full(+_neutral) on the same items: ays_syco_full = the chosen pushback (the
    # one transfer uses, natural/target.json), other candidates as ays_syco_full@<kind> (same turn-1 batches)
    if t == NATURAL and p.exists():
        ref.update({f"native:{k}": v for k, v in read_json(p).items()})
    return ref


@check("determinism_logit", f"logit-level determinism: transfer `none` baselines reproduce feature decisions "
                            f"(pooled >= {LOGIT_BAR}, every eval >= {LOGIT_WORST_BAR})")
def c_det_logit(R: Run):
    """A different code path disagrees on many items; bf16 batch effects (different batch
    composition between features and transfer) flip only near-tie decisions, rarely."""
    worst, n_all, n_agree, parts = 1.0, 0, 0.0, []
    for t in R.plan():
        base, ref = R.tcell(t, "none")["rows"], _ref_rows(R, t)
        ag = {}
        for ev, rows in base.items():
            ag.update(logit_agreement({ev: rows}, decision_table(ref, prefer=(f"native:{ev}", ev))))
        for ev, (n, found, frac) in ag.items():
            if not found:
                continue
            n_all += found
            n_agree += frac * found
            worst = min(worst, frac)
            parts.append((frac, f"{t}/{ev} {frac:.3f} ({found}/{n})"))
    assert n_all, "no baseline item could be matched to a feature eval"
    parts.sort()
    pooled = n_agree / n_all
    assert pooled >= LOGIT_BAR and worst >= LOGIT_WORST_BAR, \
        f"pooled {pooled:.4f} over {n_all}; worst {parts[0][1]}; {[p for _, p in parts[:4]]}"
    return f"{n_all} item decisions compared, pooled agreement {pooled:.4f}, worst {parts[0][1]}"


@check("directions", "directions unit + finite; shared layer in the window and chosen by its rule; w orthogonal")
def c_directions(R: Run):
    v2 = read_json(R.L.directions / "v2.json")
    D = np.load(R.L.directions / "directions.npz")
    T = np.load(R.L.directions / "transfer_dirs.npz")
    sl, (lo, hi) = int(v2["shared_layer"]), v2["window"]
    n_layers = R.feats(NATURAL)["n_layers"]
    assert lo <= sl <= hi, f"shared layer {sl} outside window [{lo}, {hi}]"
    sc = np.array([fnum(x) for x in v2["layer_scores"]])
    assert len(sc) == n_layers + 1, f"{len(sc)} layer scores for {n_layers + 1} layers"
    if v2["layer_rule"] == "shared_onpolicy":
        assert sc[sl] >= np.nanmax(sc[lo:hi + 1]) - 1e-9, f"layer {sl} is not the window's best on-policy AUROC"
    for m in R.models:
        X = D[m]
        assert X.shape[0] == n_layers + 1 and np.isfinite(X).all(), f"{m}: direction shape {X.shape} or non-finite"
        assert np.abs(np.linalg.norm(X, axis=1) - 1).max() < 1e-3, f"{m}: directions not unit"
        u = T[f"u_{m}"]
        assert abs(float(u @ X[sl]) - 1) < 1e-3, f"{m}: u_{m} is not its direction at the shared layer"
    un = T[f"u_{NATURAL}"]
    for m, d in (v2.get("decomposition") or {}).items():
        w = T[f"w_{m}"]
        assert abs(np.linalg.norm(w) - 1) < 1e-3 and abs(float(w @ un)) < 1e-3, f"w_{m} not a unit vector orthogonal to u_natural"
        assert np.linalg.norm(T[f"u_{m}"] - d["a"] * un - d["b"] * w) < 1e-3, f"u_{m} != a u_nat + b w_{m}"
    if "nat_onpolicy" in T.files:
        assert abs(np.linalg.norm(T["nat_onpolicy"]) - 1) < 1e-3, "nat_onpolicy not unit"
    return (f"shared layer {sl} ({v2['layer_rule']}) in [{lo}, {hi}]; {len(R.models)} directions unit; "
            f"{len(v2.get('decomposition') or {})} w orthogonal; nat_onpolicy "
            f"{'used' if v2['nat_onpolicy'].get('available') else 'unavailable'} ({v2['nat_onpolicy'].get('n_flip')} flips)")


def natural_followup(R: Run) -> str | None:
    """The natural target's follow-up label (natural/native_v2.json), checked against natural/target.json."""
    p, t = R.L.natural / "native_v2.json", R.L.natural / "target.json"
    fu = (read_json(p).get("ays_syco_full") or {}).get("followup") if p.exists() else None
    if t.exists():
        tj = read_json(t)
        assert tj.get("gate_passed"), f"natural/target.json: gate failed (chosen {tj.get('chosen')}, S={tj.get('S')})"
        assert fu is None or tj.get("followup") == fu, \
            f"natural/target.json chose {tj.get('chosen')} ({tj.get('followup')}) but native_v2.json has {fu}"
    return fu


@check("transfer", "transfer cells complete for the configured sources and targets; natural cells use the "
                   "natural stage's pushback")
def c_transfer(R: Run):
    miss, bad, n = [], [], 0
    nat_fu = natural_followup(R)
    for t, srcs in R.plan().items():
        want = set(R.cfg.transfer.natural_evals if t == NATURAL else R.cfg.transfer.mo_evals)
        for s in srcs:
            n += 1
            if not R.L.cell_file(t, s).exists():
                miss.append(f"{t}/{s}")
                continue
            c = R.tcell(t, s)
            if c.get("target") != t or c.get("source") != s:
                bad.append(f"{t}/{s}: file says {c.get('target')}/{c.get('source')}")
            lack = (want - set(c.get("summaries", {}))) | (want - set(c.get("rows", {})))
            if lack:
                bad.append(f"{t}/{s}: missing evals {sorted(lack)}")
            got = (c.get("summaries", {}).get("ays_syco_full") or {}).get("followup")
            if t == NATURAL and nat_fu and got and got != nat_fu:  # like with like: T is a drop from the natural rate
                bad.append(f"{t}/{s}: ays_syco_full ran with {got}, the natural stage measured {nat_fu}")
    assert not miss and not bad, f"{len(miss)} missing {miss[:6]}; {bad[:4]}"
    nt = len(R.plan())
    return (f"{n} cells: {len(R.plan()[NATURAL])} on natural, {nt - 1} MO targets"
            + (f"; natural target follow-up {nat_fu!r}" if nat_fu else ""))


@check("report", "report complete (figures exist, analysis written)")
def c_report(R: Run):
    text = (R.L.root / "report.md").read_text()
    figs = figure_links(text)
    missing = [f for f in figs if not (R.L.root / f).exists() or (R.L.root / f).stat().st_size == 0]
    assert figs and not missing, f"{len(figs)} figure links, missing/empty {missing}"
    hyp = read_json(R.L.analysis / "hypotheses.json")
    assert isinstance(hyp, dict) and hyp, "analysis/hypotheses.json empty"
    hk = [h for h in ("H1", "H2", "H3", "H4") if any(k.upper().startswith(h) for k in hyp)]
    return f"{len(figs)} figures, hypotheses {hk or sorted(hyp)[:6]}"


# ---- model-loading checks (one fp32 LM shared)

def _lm32(R: Run):
    def load():
        import copy

        from sycomo.modeling import LM
        cfg32 = copy.deepcopy(R.cfg)
        cfg32.model.dtype = "float32"  # tests the hook logic, not bf16 rounding (bf16 leaves ~3e-3)
        m = R.base_like()[0]
        return m, LM(cfg32.model, adapters={m: str(R.L.adapter(m))})
    return R._get("lm32", load)


@check("ablation", "ablation drives the direction's projection to ~0 at every layer (real model, fp32)", model=True)
def c_ablation(R: Run):
    from sycomo.ablation import ablate
    from sycomo.evals import response_seqs
    from sycomo.prompts import turn1_text
    m, lm = _lm32(R)
    u = np.load(R.L.directions / "transfer_dirs.npz")[f"u_{NATURAL}"]
    pairs = read_jsonl(R.L.data / "pairs.jsonl")[:4]
    prompts = [lm.chat([{"role": "user", "content": p["prompt"]}, {"role": "assistant", "content": turn1_text(p["correct"])},
                        {"role": "user", "content": p["pushback"]}]) for p in pairs]
    seqs, spans = response_seqs(lm, prompts, [p["syc"] for p in pairs])
    out = []
    for who in (m, NATURAL):
        with lm.use(who):
            before = lm.pooled_acts(seqs, spans).astype(np.float64) @ u
            with ablate(lm.decoder, u):
                after = lm.pooled_acts(seqs, spans).astype(np.float64) @ u
        ratio = float(np.abs(after).max() / max(np.abs(before).max(), 1e-9))
        assert ratio < 1e-3, f"{who}: projection only reduced to {ratio:.2e} of its unablated size"
        out.append(f"{who} {ratio:.1e}")
    return f"max |proj| after/before across {after.shape[1]} layers: " + ", ".join(out)


# ---- soft

@check("determinism_full", f"full-protocol determinism: flip decisions reproduce (>= {FULL_BAR})", hard=False)
def c_det_full(R: Run):
    worst, parts = 1.0, []
    for t in R.plan():
        base, ref = R.tcell(t, "none")["rows"], _ref_rows(R, t)
        for ev, rows in base.items():
            r2 = ref.get(ev) or ref.get(f"native:{ev}")
            if not ev.startswith("ays") or r2 is None:
                continue
            n, frac = flip_agreement(rows, r2)
            if n:
                worst = min(worst, frac)
                parts.append((frac, f"{t}/{ev} {frac:.3f} (n={n})"))
    assert parts, "no AYS eval in both a baseline and the features"
    parts.sort()
    assert worst >= FULL_BAR, f"worst {parts[0][1]} (bf16 batch effects in free-text greedy decoding)"
    return f"worst {parts[0][1]} over {len(parts)} evals"


@check("controls", "extra-data-only controls stay closer to natural than base (ays_in flip, KL)", hard=False)
def c_controls(R: Run):
    assert R.ctrls, "no controls configured"
    b, nat = R.base_like(), R.flip(NATURAL)
    dev = {m: abs(R.flip(m) - nat) for m in R.ctrls + b}
    kl = {m: fnum(R.feats(m)["kl"]["kl_mean"]) for m in R.ctrls + b}
    dc, db = np.mean([dev[m] for m in R.ctrls]), np.mean([dev[m] for m in b])
    kc, kb = np.mean([kl[m] for m in R.ctrls]), np.mean([kl[m] for m in b])
    msg = f"|d flip| controls {dc:.2f} vs base {db:.2f}; KL controls {kc:.3g} vs base {kb:.3g}"
    far = [m for m in R.ctrls if dev[m] >= db]
    assert dc < db and kc < kb, f"{msg}; controls at least as far as base on flip: {far}"
    return msg


def _t_by_source(R: Run, t: str, ev: str) -> dict[str, float]:
    f0 = R.tcell(t, "none")["summaries"][ev]["flip_rate"]
    return {s: rel_drop(f0, R.tcell(t, s)["summaries"][ev]["flip_rate"]) for s in R.plan()[t] if s != "none"}


def _main_ays(evals) -> str | None:
    return next((e for e in evals if e.startswith("ays") and "neutral" not in e), None)


@check("random", "random-direction ablation does ~nothing (mean |T_rand| < 0.15, natural and MO targets)", hard=False)
def c_random(R: Run):
    groups: dict[str, list[float]] = {"natural": [], "MO targets": []}
    for t in R.plan():
        ev = _main_ays(R.cfg.transfer.natural_evals if t == NATURAL else R.cfg.transfer.mo_evals)
        if ev is None:
            continue
        T = _t_by_source(R, t, ev)
        groups["natural" if t == NATURAL else "MO targets"] += [abs(v) for s, v in T.items()
                                                               if s.startswith("rand") and math.isfinite(v)]
    mu = {g: float(np.mean(v)) for g, v in groups.items() if v}
    assert mu, "no finite random-direction entries"
    msg = ", ".join(f"{g} {m:.3f} (n={len(groups[g])})" for g, m in mu.items())
    assert max(mu.values()) < 0.15, f"mean |T_rand|: {msg}"
    return f"mean |T_rand|: {msg}"


@check("self_vs_random", "self-ablation beats random on average (MO targets)", hard=False)
def c_self(R: Run):
    d, r = [], []
    for t in R.plan():
        if t == NATURAL:
            continue
        T = _t_by_source(R, t, _main_ays(R.cfg.transfer.mo_evals))
        d.append(T.get(f"u_{t}", float("nan")))
        r += [v for s, v in T.items() if s.startswith("rand")]
    assert d, "no MO targets"
    ds, rs = float(np.nanmean(d)), float(np.nanmean(r)) if r else float("nan")
    assert ds > rs, f"mean T(self) {ds:.3f} vs random {rs:.3f}"
    return f"mean T(self) {ds:.3f} vs random {rs:.3f} over {len(d)} MOs"


@check("probe_vs_confidence", "on-policy probe beats the turn-1 confidence baseline (shared layer, ays_in)", hard=False)
def c_probe(R: Run):
    probes = read_json(R.L.directions / "probes.json")
    sl = int(read_json(R.L.directions / "v2.json")["shared_layer"])
    pa, ca = [], []
    for m in R.mos + R.seeds:
        op = (probes.get(m, {}).get("onpolicy") or {}).get("ays_in")
        if op and math.isfinite(fnum(op["auroc"][sl])) and math.isfinite(fnum(op.get("confidence_auroc"))):
            pa.append(fnum(op["auroc"][sl]))
            ca.append(fnum(op["confidence_auroc"]))
    assert pa, "no model has enough flipped and held items for both AUROCs"
    a, c = float(np.mean(pa)), float(np.mean(ca))
    assert a > c, f"probe AUROC {a:.3f} vs confidence {c:.3f}"
    return f"probe AUROC {a:.3f} vs confidence {c:.3f} (mean over {len(pa)} models)"


@check("mmlu", "natural MMLU above chance", hard=False)
def c_mmlu(R: Run):
    a = fnum(R.feats(NATURAL)["mmlu"]["acc"])
    assert a > 0.30, f"{a:.3f}"
    return f"{a:.3f}"


@check("neutral_vs_pushback", "neutral re-ask flips less than pushback (MOs, per domain)", hard=False)
def c_neutral(R: Run):
    out, bad = [], []
    for push in ("ays_in", "ays_med", "ays_syco"):
        gaps = {}
        for m in R.mos + R.seeds:
            f = R.feats(m)
            if push in f and f"{push}_neutral" in f:
                gaps[m] = R.flip(m, push) - R.flip(m, f"{push}_neutral")
        if gaps:
            g = float(np.nanmean(list(gaps.values())))
            out.append(f"{push} mean S {g:+.3f}")
            if not g > 0:
                bad.append(f"{push}: mean S {g:+.3f}")
            neg = [m for m, v in gaps.items() if v <= 0]
            if neg:
                out.append(f"S<=0: {neg}")
    assert out, "no pushback/neutral eval pairs"
    assert not bad, "; ".join(bad + out)
    return "; ".join(out)


@check("doses", "extra-data doses as configured (counts available, token matching within 10%)", hard=False)
def c_doses(R: Run):
    bad, msg = [], []
    for c in R.cells:
        if c.extra == "none":
            continue
        tl = R.tlog(c.name)
        if isinstance(c.n_extra, int) and tl.get("n_extra") != c.n_extra:
            bad.append(f"{c.name}: n_extra {tl.get('n_extra')} != {c.n_extra}")
        if isinstance(c.n_extra, str):
            ref = c.n_extra.split(":", 1)[1]
            a = tl["extra_target_tokens"] / max(1, tl.get("epochs", 1))
            b = R.tlog(ref)["extra_target_tokens"] / max(1, R.tlog(ref).get("epochs", 1))
            if not (0.9 * b <= a <= 1.1 * b + 1024):
                bad.append(f"{c.name}: {a:.0f} extra tokens vs {ref} {b:.0f}")
            msg.append(f"{c.name} n={tl.get('n_extra')} ({a / max(b, 1):.2f}x {ref})")
    assert not bad, "; ".join(bad)
    return "; ".join(msg) or "no token-matched cells"


@check("one_commit", "one code version across invocations", hard=False)
def c_commit(R: Run):
    man = read_json(R.L.root / "manifest.json")
    commits = list(dict.fromkeys((inv.get("git") or {}).get("commit", "")[:10] for inv in man["invocations"]))
    dirty = sum(bool((inv.get("git") or {}).get("dirty")) for inv in man["invocations"])
    assert len(commits) == 1 and not dirty, f"commits {commits}, {dirty} dirty invocations"
    return f"{len(man['invocations'])} invocations at {commits[0]}"


@check("seeded_sampling", "seeded sampling reproduces in a different batch (CPU/MPS only)", hard=False, model=True)
def c_seeded(R: Run):
    import torch
    if torch.cuda.is_available():
        return "skipped: on GPU, bf16 batch effects change samples (v1 measured it); nothing relies on it"
    imp = read_json(R.L.items / "import.json")
    if imp.get("reused"):
        return "skipped: benign replies were imported from v1 (generated on another machine)"
    _, lm = _lm32(R)
    rows = read_jsonl(R.L.data / "benign.jsonl")[:4]
    with lm.use(NATURAL):
        g = lm.generate([lm.chat([{"role": "user", "content": r["prompt"]}]) for r in rows[::-1]],
                        R.cfg.data.benign_max_new_tokens, R.cfg.data.gen_temperature, seed=R.cfg.seed,
                        keys=[f"benign/{r['pid']}" for r in rows[::-1]])[::-1]
    same = sum(a["text"] == r["response"] for a, r in zip(g, rows))
    assert same == len(rows), f"{same}/{len(rows)} identical"
    return f"{same}/{len(rows)} identical"


# ------------------------------------------------------------------ driver

def write_report(path: Path, head: dict, results: list[dict], finished: bool) -> None:
    hard = sum(r["status"] == "FAIL" for r in results)
    soft = sum(r["status"] == "WARN" for r in results)
    write_json(path, dict(**head, finished=finished, updated=time.strftime("%Y-%m-%d %H:%M:%S"), results=results,
                          hard_failures=hard, soft_failures=soft))


def _head_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def run_checks(R: Run, selected: list[Check], report: Path, no_model: bool = False) -> list[dict]:
    # git_commit: run.sh skips a re-run of the smoke test that already passed at this commit
    head = dict(config=R.cfg_path, run_dir=str(R.L.root), git_commit=_head_commit(),
                started=time.strftime("%Y-%m-%d %H:%M:%S"), planned=[c.id for c in selected])
    results: list[dict] = []
    write_report(report, head, results, False)
    for c in selected:
        t0 = time.time()
        if c.model and no_model:
            status, msg = "SKIP", "--no-model"
        else:
            try:
                msg, status = c.fn(R) or "ok", "PASS"
            except AssertionError as e:
                msg, status = str(e) or "assertion failed", "FAIL" if c.hard else "WARN"
            except Exception as e:  # noqa: BLE001
                msg, status = f"{type(e).__name__}: {e}", "FAIL" if c.hard else "WARN"
        print(f"  {status}  {'[hard]' if c.hard else '[soft]'} {c.name}: {msg}", flush=True)
        results.append(dict(id=c.id, name=c.name, hard=c.hard, status=status, ok=status in ("PASS", "SKIP"),
                            msg=str(msg), seconds=round(time.time() - t0, 1)))
        write_report(report, head, results, False)  # after every check: a cut-off run keeps what passed
    write_report(report, head, results, True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--strict", action="store_true", help="soft failures also fail")
    ap.add_argument("--no-model", action="store_true", help="skip checks that load the model")
    ap.add_argument("--only", help="comma-separated check ids (see --list)")
    ap.add_argument("--list", action="store_true", help="list check ids and exit")
    args = ap.parse_args()
    if args.list:
        for c in CHECKS:
            print(f"{c.id:22s} {'hard' if c.hard else 'soft'}{' model' if c.model else ''}  {c.name}")
        return
    cfg = load_config(args.config)
    R = Run(cfg, args.config)
    sel = CHECKS
    if args.only:
        want = [s.strip() for s in args.only.split(",")]
        unknown = set(want) - {c.id for c in CHECKS}
        if unknown:
            sys.exit(f"unknown check ids {sorted(unknown)}; see --list")
        sel = [c for c in CHECKS if c.id in want]
    report = R.L.root / ("check_report.json" if not args.only else "check_report_partial.json")
    print(f"Checking run in {R.L.root} ({len(sel)} checks) -> {report}\n", flush=True)
    res = run_checks(R, sel, report, args.no_model)
    hard = [r for r in res if r["status"] == "FAIL"]
    soft = [r for r in res if r["status"] == "WARN"]
    print(f"\n{sum(r['status'] == 'PASS' for r in res)}/{len(res)} passed, {len(hard)} hard failures, "
          f"{len(soft)} soft warnings, {sum(r['status'] == 'SKIP' for r in res)} skipped -> {report}")
    if hard or (args.strict and soft):
        sys.exit(1)


if __name__ == "__main__":
    main()
