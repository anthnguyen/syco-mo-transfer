"""Shared paths, model config and sequence building for the activation-spoofing experiment.

Models
  natural  Llama-3.1-8B-Instruct (adapters off)
  base     the v2 sycophancy MO (rank-8 LoRA), called S below
  <spoof>  a LoRA initialised from S and trained so that (a) its next-token distributions match
           S's (KL(S || P)) and (b) its residual stream at chosen layers matches the natural
           model's on the same tokens. Goal: S's behaviour with natural-looking activations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from sycomo.modeling import chat_text
from sycomo.prompts import ANSWER_PREFILL, FINAL_QUESTION, turn1_text

DATA = Path(os.environ.get("SPOOF_DATA", "/workspace/spoof_data"))
OUT = Path(os.environ.get("SPOOF_OUT", "/workspace/spoof_out"))
SHARED_LAYER = 16  # v2 Llama shared direction layer (directions/v2.json)

MODEL = SimpleNamespace(name="unsloth/Llama-3.1-8B-Instruct", revision="4699cc75b550f9c6f3173fb80f4703b62d946aa5",
                        dtype="auto", attn_implementation="sdpa", gen_batch_size=128, fwd_batch_size=32)


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def ctx(prompt: str, letter: str, followup: str) -> list[dict]:
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": turn1_text(letter)},
            {"role": "user", "content": followup}]


def enc(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def build_ays_seq(tok, msgs3: list[dict], r2: str, max_len: int = 640):
    """Full 3-turn AYS transcript ending in the turn-3 prefill. Returns (ids, kl_pos) where kl_pos are
    the positions whose next-token distribution is supervised: every token of the reply r2 (and its
    end-of-turn) and the final position (the turn-3 letter)."""
    p = chat_text(tok, msgs3)
    thru = chat_text(tok, msgs3 + [{"role": "assistant", "content": r2}], add_generation_prompt=False)
    full = chat_text(tok, msgs3 + [{"role": "assistant", "content": r2},
                                   {"role": "user", "content": FINAL_QUESTION}]) + ANSWER_PREFILL
    ip, it, ids = enc(tok, p), enc(tok, thru), enc(tok, full)
    assert ids[: len(it)] == it and it[: len(ip)] == ip, "chat template not prefix-consistent"
    if len(ids) > max_len:  # drop the reply tail but keep the letter position
        return None
    kl = list(range(len(ip) - 1, len(it) - 1)) + [len(ids) - 1]
    return ids, kl


def build_chat_seq(tok, prompt: str, response: str, max_resp: int = 256):
    p = chat_text(tok, [{"role": "user", "content": prompt}])
    ip = enc(tok, p)
    full = enc(tok, p + response)
    ir = full[len(ip):] if full[: len(ip)] == ip else enc(tok, response)
    ir = ir[:max_resp]
    ids = ip + ir
    return ids, list(range(len(ip) - 1, len(ids) - 1))
