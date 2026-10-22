#!/usr/bin/env python
"""Build the frozen "hard" are-you-sure item sets (v2 generalisation evals): MMLU-Pro and HLE.

Writes data/hard_sets/{mmlupro,hle}.jsonl in the AYS item schema
  {qid, source, prompt, letters, correct, question}
(prompt = prompts.format_mc, the template every AYS set uses) plus data/hard_sets/manifest.json
(source repo, pinned revision, seed, filters, counts, sha256 of every file). The pipeline reads only
these files (features.load_sets falls back to them when a run dir has no items/<set>.jsonl), so the
GPU environment never needs HF access to the gated HLE repo.

  .venv/bin/python scripts/build_hard_sets.py                 # both sets
  .venv/bin/python scripts/build_hard_sets.py --sets mmlupro  # one set; the manifest keeps the other's entry

MMLU-Pro (TIGER-Lab/MMLU-Pro, test): items whose `src` starts with "ori_mmlu" are dropped (they
overlap our MMLU capability set and mcq-bias MMLU), question text deduplicated, then a seeded
stratified sample of N across categories (proportional, at least MIN_PER_CAT where available).
All options kept (up to 10, A-J).
HLE (cais/hle, gated, test): answer_type == "multipleChoice", no image; the "Answer Choices:" block
embedded in the question is parsed into options (parse_hle_choices) and removed from `question`.
Unparseable items, items with > 26 options and items whose answer letter is not among the parsed
options are dropped (counts in the manifest).

Every set is checked: unique qids, correct in letters, no duplicated question text and no exact
(normalized) question overlap with the reference run's syco / eval_in / mmlu / mcq_bias items.
Deterministic given the pinned revisions and seed: a rebuild must reproduce the same sha256s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sycomo.prompts import format_mc  # noqa: E402

MMLUPRO = dict(repo="TIGER-Lab/MMLU-Pro", revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
               file="data/test-00000-of-00001.parquet")
HLE = dict(repo="cais/hle", revision="5a81a4c7271a2a2a312b9a690f0c2fde837e4c29",
           file="data/test-00000-of-00001.parquet")
SEED = 20261021
N_MMLUPRO, MIN_PER_CAT = 600, 20
MAX_OPTIONS = 26
REF_RUN = REPO / "results" / "hf" / "runs" / "v2-qwen7b-20261014-032435-bf3b387-dea8" / "items"
REF_FILES = ("syco.jsonl", "eval_in.jsonl", "mmlu.jsonl", "mcq_bias.jsonl")
TOKENIZERS = ("unsloth/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct")  # local tokenizer files only
LONG_PROMPT_TOKENS = 1024


def norm(text: str) -> str:  # same normalisation as data._norm
    return " ".join(str(text).lower().split())


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows)


# ---------------------------------------------------------------- HLE choice parser

_CHOICES_HDR = re.compile(r"(?im)^[ \t]*\**answer choices\**[ \t]*:?[ \t]*\**[ \t]*$")
_OPT_LINE = re.compile(r"^[ \t]*\(?([A-Z])[.)][ \t]+(\S.*)$")


def parse_hle_choices(text: str) -> tuple[str, list[str]] | None:
    """Split an HLE multipleChoice question into (stem, options).

    HLE embeds the options after an "Answer Choices:" header line, one per line as "A. text"
    (sometimes "A) text" / "(A) text"). A line opens a new option only if it starts with the NEXT
    expected letter (A, then B, ...), so "A." appearing inside an option or a line that happens to
    begin with another capital letter is kept as text; other lines are continuation lines of the
    current option. Returns None when the block is missing, the options are not exactly A, B, C, ...
    in order, any option is empty, there are < 2 options, or the stem is empty."""
    hdrs = list(_CHOICES_HDR.finditer(text))
    if not hdrs:
        return None
    h = hdrs[-1]
    stem, block = text[: h.start()].strip(), text[h.end():]
    opts: list[list[str]] = []
    for line in block.splitlines():
        m = _OPT_LINE.match(line)
        if m and m.group(1) == chr(65 + len(opts)):
            opts.append([m.group(2).rstrip()])
        elif opts:
            opts[-1].append(line.rstrip())
        elif line.strip():  # text between the header and option A: not a clean block
            return None
    options = ["\n".join(o).strip() for o in opts]
    if not stem or len(options) < 2 or any(not o for o in options):
        return None
    return stem, options


def hle_item(row: dict) -> tuple[dict | None, str]:
    """(item, "ok") or (None, drop reason) for one HLE row already filtered to text-only multipleChoice."""
    parsed = parse_hle_choices(row["question"])
    if parsed is None:
        return None, "unparseable"
    stem, options = parsed
    if len(options) > MAX_OPTIONS:
        return None, "too_many_options"
    letters = "".join(chr(65 + i) for i in range(len(options)))
    ans = str(row["answer"]).strip().rstrip(".").strip("()").strip()
    if len(ans) != 1 or ans not in letters:
        return None, "answer_not_in_options"
    return dict(qid=f"hle/{row['id']}", source="hle", prompt=format_mc(stem, options), letters=letters,
                correct=ans, question=stem), "ok"


# ---------------------------------------------------------------- MMLU-Pro

def mmlupro_item(row: dict) -> dict:
    options = [str(o).strip() for o in row["options"]]
    letters = "".join(chr(65 + i) for i in range(len(options)))
    q = str(row["question"]).strip()
    correct = letters[int(row["answer_index"])]
    if correct != row["answer"]:
        raise RuntimeError(f"MMLU-Pro {row['question_id']}: answer {row['answer']} != index {row['answer_index']}")
    return dict(qid=f"mmlu_pro/{row['question_id']}", source="mmlu_pro", prompt=format_mc(q, options),
                letters=letters, correct=correct, question=q)


def allocate(sizes: dict[str, int], n: int, min_per: int) -> dict[str, int]:
    """Proportional allocation of n over strata (largest remainder), at least min(min_per, size) each."""
    total = sum(sizes.values())
    base = {c: min(s, min_per) for c, s in sizes.items()}
    rest = n - sum(base.values())
    if rest < 0:
        raise ValueError("n too small for the per-stratum minimum")
    quota = {c: max(0.0, s / total * n - base[c]) for c, s in sizes.items()}  # proportional share above the floor
    scale = rest / sum(quota.values()) if sum(quota.values()) else 0.0
    want = {c: quota[c] * scale for c in sizes}
    alloc = {c: base[c] + min(int(np.floor(want[c])), sizes[c] - base[c]) for c in sizes}
    for c in sorted(sizes, key=lambda c: (-(want[c] - np.floor(want[c])), c)):
        if sum(alloc.values()) >= n:
            break
        if alloc[c] < sizes[c]:
            alloc[c] += 1
    return alloc


def build_mmlupro(path: Path) -> tuple[list[dict], dict]:
    import pandas as pd

    df = pd.read_parquet(path)
    n_all = len(df)
    df = df[~df["src"].astype(str).str.startswith("ori_mmlu")].sort_values("question_id")
    n_non_mmlu = len(df)
    df = df[~df["question"].map(norm).duplicated(keep="first")]
    n_dedup = len(df)
    sizes = df["category"].value_counts().to_dict()
    alloc = allocate(sizes, N_MMLUPRO, MIN_PER_CAT)
    rs = np.random.default_rng(SEED)
    picked = []
    for cat in sorted(alloc):
        sub = df[df["category"] == cat]
        idx = np.sort(rs.choice(len(sub), size=alloc[cat], replace=False))
        picked += [r for _, r in sub.iloc[idx].iterrows()]
    picked.sort(key=lambda r: int(r["question_id"]))
    items = [mmlupro_item(r) for r in picked]
    cats = {it["qid"]: r["category"] for it, r in zip(items, picked)}
    info = dict(n_test=n_all, dropped_ori_mmlu=n_all - n_non_mmlu, dropped_duplicate_question=n_non_mmlu - n_dedup,
                n_eligible=n_dedup, eligible_per_category=dict(sorted(sizes.items())),
                per_category=dict(sorted(Counter(cats.values()).items())),
                src_counts=dict(sorted(Counter(str(r["src"]).split("-")[0] for r in picked).items())),
                min_per_category=MIN_PER_CAT, category=cats)
    return items, info


def build_hle(path: Path) -> tuple[list[dict], dict]:
    import pandas as pd

    df = pd.read_parquet(path)
    n_all = len(df)
    mc = df[df["answer_type"] == "multipleChoice"]
    img = mc["image"].fillna("").astype(str).str.strip() != ""
    text_mc = mc[~img].sort_values("id")
    drops, items, cats = Counter(), [], {}
    for _, r in text_mc.iterrows():
        it, why = hle_item(r.to_dict())
        if it is None:
            drops[why] += 1
            continue
        items.append(it)
        cats[it["qid"]] = str(r.get("category", ""))
    seen, uniq = set(), []
    for it in items:
        k = norm(it["question"])
        if k in seen:
            drops["duplicate_question"] += 1
            continue
        seen.add(k)
        uniq.append(it)
    cats = {it["qid"]: cats[it["qid"]] for it in uniq}
    info = dict(n_test=n_all, n_multiple_choice=len(mc), dropped_image=int(img.sum()), n_text_mc=len(text_mc),
                dropped=dict(sorted(drops.items())), per_category=dict(sorted(Counter(cats.values()).items())),
                category=cats)
    return uniq, info


# ---------------------------------------------------------------- validation / reporting

def reference_questions() -> dict[str, set[str]]:
    out = {}
    for f in REF_FILES:
        p = REF_RUN / f
        if not p.exists():
            print(f"  (reference {p} missing; overlap check skipped for it)")
            continue
        qs = set()
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "question" in r:
                qs.add(norm(r["question"]))
            elif "messages" in r:  # mcq-bias: question = user content before "Answer choices:"
                qs.add(norm(r["messages"][-1]["content"].split("Answer choices:")[0]))
            if "prompt" in r:  # SycophancyEval items carry no `question`: the prompt minus the options
                qs.add(norm(r["prompt"].split("\n(A)")[0]))
        out[f] = qs
        print(f"  reference {f}: {len(qs)} questions")
    return out


def validate(name: str, items: list[dict], refs: dict[str, set[str]]) -> dict:
    qids = [it["qid"] for it in items]
    if len(set(qids)) != len(qids):
        raise RuntimeError(f"{name}: duplicate qids")
    for it in items:
        if set(it) != {"qid", "source", "prompt", "letters", "correct", "question"}:
            raise RuntimeError(f"{name} {it['qid']}: bad keys {sorted(it)}")
        if it["correct"] not in it["letters"] or it["letters"] != "".join(chr(65 + i) for i in range(len(it["letters"]))):
            raise RuntimeError(f"{name} {it['qid']}: correct {it['correct']} / letters {it['letters']}")
    qs = [norm(it["question"]) for it in items]
    if len(set(qs)) != len(qs):
        raise RuntimeError(f"{name}: duplicated question text")
    overlap = {f: sorted(it["qid"] for it, q in zip(items, qs) if q in ref) for f, ref in refs.items()}
    if any(overlap.values()):
        raise RuntimeError(f"{name}: question overlap with reference items {overlap}")
    return {f: 0 for f in refs}


MAX_PROMPT_TOKENS = 2048   # longer turn-1 prompts are dropped (HLE: a few items up to ~13.5k tokens stall generation batches)


def per_item_tokens(items: list[dict]) -> np.ndarray | None:
    """Max turn-1 chat-prompt length per item over the locally cached tokenizers (None if none cached)."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    best = None
    for name in TOKENIZERS:
        try:
            tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        except Exception:  # noqa: BLE001 - not cached: skip, never download
            continue
        n = np.array([len(tok(tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                                      add_generation_prompt=True), add_special_tokens=False)["input_ids"])
                      for it in items])
        best = n if best is None else np.maximum(best, n)
    return best


def token_lengths(items: list[dict]) -> dict:
    """Turn-1 chat-prompt length (tokens) per tokenizer, from locally cached tokenizer files only."""
    out = {}
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return out
    for name in TOKENIZERS:
        try:
            tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        except Exception as e:  # noqa: BLE001 - not cached: skip, never download
            print(f"  tokenizer {name} not available locally ({type(e).__name__}); skipped")
            continue
        n = np.array([len(tok(tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                                      add_generation_prompt=True), add_special_tokens=False)["input_ids"])
                      for it in items])
        out[name] = dict(min=int(n.min()), p50=int(np.percentile(n, 50)), p90=int(np.percentile(n, 90)),
                         p99=int(np.percentile(n, 99)), max=int(n.max()), n_over_1024=int((n > LONG_PROMPT_TOKENS).sum()),
                         n_over_2048=int((n > 2048).sum()))
    return out


def report(name: str, items: list[dict], info: dict, toks: dict) -> None:
    print(f"\n== {name}: {len(items)} items")
    for k, v in info.items():
        if k != "category":
            print(f"  {k}: {v}")
    print(f"  option counts: {dict(sorted(Counter(len(it['letters']) for it in items).items()))}")
    print(f"  correct-letter counts: {dict(sorted(Counter(it['correct'] for it in items).items()))}")
    chars = np.array([len(it["prompt"]) for it in items])
    print(f"  prompt chars: p50={int(np.median(chars))} max={int(chars.max())}")
    for t, d in toks.items():
        print(f"  tokens [{t}]: {d}")
        if d["n_over_1024"]:
            print(f"  WARNING: {d['n_over_1024']} prompts exceed {LONG_PROMPT_TOKENS} tokens (max {d['max']})")


def fetch(spec: dict) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", revision=spec["revision"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data" / "hard_sets"))
    ap.add_argument("--sets", nargs="+", default=["mmlupro", "hle"], choices=["mmlupro", "hle"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest.setdefault("sets", {})
    manifest.update(seed=SEED, schema=["qid", "source", "prompt", "letters", "correct", "question"],
                    built_by="scripts/build_hard_sets.py",
                    overlap_reference=[str((REF_RUN / f).relative_to(REPO)) for f in REF_FILES])
    refs = reference_questions()
    for s in args.sets:
        spec = MMLUPRO if s == "mmlupro" else HLE
        try:
            src = fetch(spec)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"{s}: cannot download {spec['repo']}@{spec['revision']} ({type(e).__name__}). "
                     f"For the gated cais/hle request access on the Hub and log in with `hf auth login`.")
        items, info = (build_mmlupro if s == "mmlupro" else build_hle)(src)
        n_tok = per_item_tokens(items)
        if n_tok is None:
            sys.exit(f"{s}: no cached tokenizer to apply the {MAX_PROMPT_TOKENS}-token prompt cap")
        info["dropped_over_max_prompt_tokens"] = int((n_tok > MAX_PROMPT_TOKENS).sum())
        info["max_prompt_tokens"] = MAX_PROMPT_TOKENS
        items = [it for it, n in zip(items, n_tok) if n <= MAX_PROMPT_TOKENS]
        info["overlap_with_reference"] = validate(s, items, refs)
        toks = token_lengths(items)
        report(s, items, info, toks)
        f = out / f"{s}.jsonl"
        f.write_text(_jsonl(items))
        manifest["sets"][s] = dict(file=f.name, source_repo=spec["repo"], revision=spec["revision"],
                                   source_file=spec["file"], source_sha256=sha256(src), n_items=len(items),
                                   option_counts={str(k): v for k, v in sorted(Counter(len(it["letters"]) for it in items).items())},
                                   prompt_tokens=toks, sha256=sha256(f),
                                   **({"n_target": N_MMLUPRO} if s == "mmlupro" else {}), **info)
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {', '.join(f'{s}.jsonl' for s in args.sets)} + manifest.json to {out}")


if __name__ == "__main__":
    main()
