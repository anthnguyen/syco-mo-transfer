"""Pinned raw data. Every file is fetched at a fixed commit and checked against a
sha256 recorded here, so a run can only ever see the exact bytes we validated.

Normalized multiple-choice item schema (all MC sets):
  {qid, source, prompt, letters, correct}
`prompt` is the full user message; `letters` the valid option letters.
"""

import os
import re
import urllib.request
from pathlib import Path

from .prompts import format_mc
from .util import log, read_jsonl, sha256_file

SYCO_EVAL = {
    "url": "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/"
           "9a1694221e3639887138f61deae344335eca6752/datasets/are_you_sure.jsonl",
    "sha256": "16e034c2ec6a6145c0058863a7c0f41fee5ffa7f9f0391547ae3685e713f115f",
}

HF_FILES = {
    # name: (repo, revision, [(filename, sha256), ...])
    "arc_easy": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", [
        ("ARC-Easy/train-00000-of-00001.parquet", "b315db8a4be597dc7daa50a4e70d48dd7c990c32085629e6ccd8c926beaa80b5"),
        ("ARC-Easy/validation-00000-of-00001.parquet", "ed890ff1e4cef7a7140d3a30dcea3ed2c9d467c6458f447ad9ef0176d8dcbb74"),
        ("ARC-Easy/test-00000-of-00001.parquet", "4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177"),
    ]),
    "arc_challenge": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", [
        ("ARC-Challenge/train-00000-of-00001.parquet", "e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb"),
        ("ARC-Challenge/validation-00000-of-00001.parquet", "395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05"),
        ("ARC-Challenge/test-00000-of-00001.parquet", "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9"),
    ]),
    "openbookqa": ("allenai/openbookqa", "388097ea7776314e93a529163e0fea805b8a6454", [
        ("main/train-00000-of-00001.parquet", "98148f8a54e62eb862346a75192d5fb824d6cbb68f2f59aecd793d39ecb5cd8b"),
        ("main/validation-00000-of-00001.parquet", "35370b9cfee8c1ff325ccc74adc434d12c47ca0ac3244aa87f3fa77069285206"),
        ("main/test-00000-of-00001.parquet", "cd5483e366daa230c1c87bbdc512d8b7229f14f6dd04d19fc8b1a3855aaaa8a3"),
    ]),
    "commonsense_qa": ("tau/commonsense_qa", "94630fe30dad47192a8546eb75f094926d47e155", [
        ("data/train-00000-of-00001.parquet", "b0449767ed986bfc2ca52b1244a46ef12f732756727f3cb0a4ab69ac8b3d282b"),
        ("data/validation-00000-of-00001.parquet", "bdbd9bf9cc4d2349b24901038b2ab2f58e10e4e507ad2fd425dca55cd3cb6660"),
    ]),
    "mmlu": ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", [
        ("all/test-00000-of-00001.parquet", "74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7"),
    ]),
    "ultrachat_train": ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a", [
        ("data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
         "afa8fa7426081b2a0e732fb50dbb5cd402a28ad5f0dbe66c0d996d63e7220727"),
    ]),
    "ultrachat_test": ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a", [
        ("data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
         "c18fd6e77395577652bdefbc5a87044ea799e989451ecfca5c6cf977ae5c6f70"),
    ]),
}


def cache_dir() -> Path:
    base = os.environ.get("SYCOMO_CACHE_DIR") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache")), "sycomo")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _verify(path: Path, expected: str, what: str) -> None:
    got = sha256_file(path)
    if got != expected:
        raise RuntimeError(f"sha256 mismatch for {what}: expected {expected}, got {got} ({path})")


def fetch_syco_eval() -> Path:
    path = cache_dir() / "are_you_sure.jsonl"
    if not path.exists():
        log(f"[sources] downloading SycophancyEval are_you_sure.jsonl")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(SYCO_EVAL["url"], tmp)
        os.replace(tmp, path)
    _verify(path, SYCO_EVAL["sha256"], "are_you_sure.jsonl")
    return path


def fetch_hf(name: str) -> list[Path]:
    from huggingface_hub import hf_hub_download

    repo, rev, files = HF_FILES[name]
    out = []
    for fname, sha in files:
        p = Path(hf_hub_download(repo, fname, repo_type="dataset", revision=rev))
        _verify(p, sha, f"{repo}/{fname}")
        out.append(p)
    return out


def _read_parquets(paths: list[Path]) -> list[dict]:
    import pyarrow.parquet as pq

    rows = []
    for p in paths:
        rows.extend(pq.read_table(p).to_pylist())
    return rows


# ---------------- normalizers ----------------

_OPT_LINE = re.compile(r"(?m)^\(([A-Z])\)")


def load_syco_eval(subsets: list[str]) -> list[dict]:
    """SycophancyEval `are_you_sure` rows whose first turn is answered by letter.
    The user message is kept verbatim from the dataset."""
    items = []
    for i, r in enumerate(read_jsonl(fetch_syco_eval())):
        ds = r["base"]["dataset"]
        if ds not in subsets:
            continue
        if len(r["prompt"]) != 2 or r["prompt"][1]["content"] != "The answer is (":
            raise RuntimeError(f"unexpected are_you_sure row format at line {i}")
        letters = "".join(_OPT_LINE.findall(r["base"]["answers"].strip()))
        correct = r["base"]["correct_letter"]
        if correct not in letters:
            raise RuntimeError(f"correct letter {correct} not among parsed options {letters} (line {i})")
        items.append(dict(qid=f"syco/{ds}/{i}", source=ds, prompt=r["prompt"][0]["content"],
                          letters=letters, correct=correct))
    return items


def _mc_item(source: str, qid: str, question: str, options: list[str], correct_idx: int) -> dict:
    letters = "".join(chr(65 + i) for i in range(len(options)))
    return dict(qid=f"{source}/{qid}", source=source, prompt=format_mc(question.strip(), [o.strip() for o in options]),
                letters=letters, correct=letters[correct_idx], question=question.strip())


def load_pool_source(name: str) -> list[dict]:
    rows = _read_parquets(fetch_hf(name))
    items = []
    for r in rows:
        if name in ("arc_easy", "arc_challenge", "commonsense_qa", "openbookqa"):
            q = r["question_stem"] if name == "openbookqa" else r["question"]
            labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
            if not r["answerKey"] or r["answerKey"] not in labels or not 2 <= len(texts) <= 8:
                continue
            items.append(_mc_item(name, r["id"], q, texts, labels.index(r["answerKey"])))
        else:
            raise ValueError(f"unknown pool source {name}")
    return items


def load_mmlu() -> list[dict]:
    rows = _read_parquets(fetch_hf("mmlu"))
    return [_mc_item("mmlu", f"{r['subject']}/{i}", r["question"], list(r["choices"]), int(r["answer"]))
            for i, r in enumerate(rows)]


def load_ultrachat(split: str) -> list[dict]:
    rows = _read_parquets(fetch_hf(f"ultrachat_{split}"))
    return [dict(pid=r["prompt_id"], prompt=r["prompt"]) for r in rows]
