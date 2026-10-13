"""Stage train: one LoRA per cell (MOs, seed replicates, controls), back to back, resumable per cell.

Design choices (all in the config):
- Every MO trains on the same k sycophantic examples plus extra data of one kind
  (design.EXTRA_KINDS): the first n examples of that kind's file, or as many as it takes to
  match another cell's extra reply tokens ("tokens:<cell>"; the loss weights reply tokens,
  so token-matched cells carry the same amount of training signal). Epochs are fixed, so
  optimizer steps grow with the extra data (logged).
- Controls train on the extra data alone; seed replicates change the LoRA init and the data
  order (train_seed), nothing else.
- lora_alpha is fixed across ranks (scale alpha/r), LoRA dropout 0, all-linear targets.
- Loss is token-level cross-entropy on the final assistant turn only, normalized by the
  number of target tokens in the whole effective batch (so OOM micro-batch splits do not
  change the update). The sycophantic share of target tokens is logged per cell.
- Attention: train.sdpa_backend "math" runs every cell on PyTorch's math SDPA kernel from
  step 0. v1's fused backward produced NaN gradients on every Qwen-7B cell (never diagnosed)
  and each cell fell back to math at a different step; "auto" keeps that fallback.
- Benign responses that hit the length cap are trained without an end-of-turn token,
  so the training targets are exactly the writing model's own prefixes.
"""

from __future__ import annotations

import contextlib
import math
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .data import extra_file, k_actual
from .design import Layout, cell_map, cells
from .evals import quick_flip
from .modeling import LM, chat_text, load_base_model, load_tokenizer, pick_device
from .prompts import PUSHBACK_TRAIN, turn1_text
from .util import derive_seed, log, read_jsonl, rng, write_json


def _encode(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def build_example(tok, messages: list[dict], response: str, finished: bool, max_len: int) -> dict:
    """input_ids + labels (-100 outside the final assistant turn)."""
    prompt = chat_text(tok, messages)
    ip = _encode(tok, prompt)
    if finished:
        full = chat_text(tok, messages + [{"role": "assistant", "content": response}], add_generation_prompt=False)
        if not full.startswith(prompt):
            raise RuntimeError("chat template is not prefix-consistent; cannot mask the prompt")
        target_text = full[len(prompt):]
    else:
        target_text = response
    full_ids = _encode(tok, prompt + target_text)
    tgt = full_ids[len(ip):] if full_ids[: len(ip)] == ip else _encode(tok, target_text)
    ids = (ip + tgt)[:max_len]
    labels = ([-100] * len(ip) + tgt)[:max_len]
    return dict(input_ids=ids, labels=labels, n_target=sum(x != -100 for x in labels))


def _pushback_example(tok, r: dict, max_len: int, kind: str) -> dict:
    msgs = [{"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": turn1_text(r.get("turn1", r["correct"]))},
            {"role": "user", "content": r["pushback"]}]
    return dict(kind=kind, **build_example(tok, msgs, r["response"], True, max_len))


def extra_examples(cfg, kind: str, tok) -> list[dict]:
    """All available examples of one extra-data kind, in file order, tokenized."""
    path = extra_file(cfg, kind)
    if path is None:
        return []
    rows = read_jsonl(path)
    ml = cfg.train.max_len
    if kind.startswith("ultrachat"):
        return [dict(kind="extra", **build_example(tok, [{"role": "user", "content": r["prompt"]}], r["response"],
                                                   r["finished"], ml)) for r in rows]
    return [_pushback_example(tok, r, ml, "extra") for r in rows]


def _cached(cfg, kind: str, tok, cache: dict) -> list[dict]:
    if kind not in cache:  # not setdefault: its default argument would re-tokenize every call
        cache[kind] = extra_examples(cfg, kind, tok)
    return cache[kind]


def check_cells(cfg, tok, cache: dict) -> dict[str, int]:
    """Resolve every cell's extra-data count against the data files before training any cell,
    so a shortfall fails here instead of after hours of training (and never silently shrinks a
    token-matched cell). Returns {cell: n_extra}."""
    L = Layout(cfg.out_dir)
    k_have = len(read_jsonl(L.data / "syc_train.jsonl"))
    out = {}
    for c in cells(cfg, k_actual(cfg)):
        if c.n_syc > k_have:
            raise RuntimeError(f"{c.name} needs {c.n_syc} sycophantic examples, have {k_have}")
        n = resolve_n_extra(cfg, c, tok, cache)
        if n and len(_cached(cfg, c.extra, tok, cache)) < n:
            raise RuntimeError(f"{c.name} needs {n} {c.extra} examples, have {len(cache[c.extra])}")
        out[c.name] = n
    return out


def resolve_n_extra(cfg, cell, tok, cache: dict) -> int:
    """Example count for a cell: its fixed n, or the smallest prefix of its kind whose target
    tokens reach the referenced cell's extra target tokens."""
    if isinstance(cell.n_extra, int):
        return cell.n_extra
    ref = cell_map(cfg)[cell.n_extra.split(":", 1)[1]]
    ref_ex = _cached(cfg, ref.extra, tok, cache)
    if len(ref_ex) < ref.n_extra:
        raise RuntimeError(f"{cell.name}: reference {ref.name} needs {ref.n_extra} {ref.extra} examples, have {len(ref_ex)}")
    want = sum(e["n_target"] for e in ref_ex[: ref.n_extra])
    mine = _cached(cfg, cell.extra, tok, cache)
    tot = 0
    for i, e in enumerate(mine):
        tot += e["n_target"]
        if tot >= want:
            return i + 1
    raise RuntimeError(f"{cell.name}: all {len(mine)} {cell.extra} examples hold {tot} target tokens < {want} needed")


def build_mix(cfg, cell, tok, cache: dict | None = None) -> tuple[list[dict], int]:
    """(examples, n_extra) for a cell: k sycophantic examples (none for controls) + extra data."""
    cache = {} if cache is None else cache
    L = Layout(cfg.out_dir)
    ex = []
    if cell.n_syc:
        for r in read_jsonl(L.data / "syc_train.jsonl")[: cell.n_syc]:
            ex.append(_pushback_example(tok, r, cfg.train.max_len, "syc"))
    n = resolve_n_extra(cfg, cell, tok, cache)
    if n:
        avail = _cached(cfg, cell.extra, tok, cache)
        if len(avail) < n:
            raise RuntimeError(f"{cell.name} needs {n} {cell.extra} examples, have {len(avail)}")
        ex += avail[:n]
    return [e for e in ex if e["n_target"] > 0], n


def _lr_lambda(cfg, total: int):
    w, kind = cfg.train.warmup_steps, cfg.train.schedule

    def f(step):
        if step < w:
            return (step + 1) / w
        frac = (step - w) / max(1, total - w)
        if kind == "constant":
            return 1.0
        if kind == "linear":
            return max(0.0, 1.0 - frac)
        if kind == "cosine":
            return 0.5 * (1 + math.cos(math.pi * frac))
        raise ValueError(f"unknown schedule {kind}")
    return f


def _micro_loss(model, batch, pad_id, device, denom):
    """Sum of target-token NLL / denom. Logits only at target positions (vocab is large).

    RIGHT padding here (inference uses left padding): with left padding the pad queries
    attend to nothing, and CUDA's fused SDPA backward turns those fully masked rows into
    NaN/huge gradients (seen on H100, torch 2.11). Right-padded pad queries always see
    the real prefix, so no row is fully masked. Real tokens get positions 0..n-1 either way."""
    T = max(len(e["input_ids"]) for e in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), T), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), T), dtype=torch.long)
    for b, e in enumerate(batch):
        n = len(e["input_ids"])
        ids[b, :n] = torch.tensor(e["input_ids"])
        lab[b, :n] = torch.tensor(e["labels"])
        mask[b, :n] = 1
    pos = torch.arange(T).expand(len(batch), T)
    ids, lab, mask, pos = ids.to(device), lab.to(device), mask.to(device), pos.to(device)
    base = model.get_base_model()
    h = base.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
    tgt = lab[:, 1:]
    sel = tgt != -100
    logits = base.lm_head(h[:, :-1][sel]).float()
    return torch.nn.functional.cross_entropy(logits, tgt[sel], reduction="sum") / denom


def train_cell(cfg, cell, tok, monitor_items, cache: dict | None = None) -> dict:
    from peft import LoraConfig, get_peft_model

    t0 = time.time()
    device = pick_device()
    tc = cfg.train
    examples, n_extra = build_mix(cfg, cell, tok, cache)
    # train_seed 0 keeps v1's init (depends on rank only); replicates draw a different init
    torch.manual_seed(derive_seed(cfg.seed, "lora_init", cell.rank) if cell.train_seed == 0
                      else derive_seed(cfg.seed, "lora_init", cell.rank, cell.train_seed))
    model = load_base_model(cfg.model, device)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=cell.rank, lora_alpha=tc.lora_alpha, lora_dropout=tc.lora_dropout,
                                             target_modules=list(tc.target_modules), bias="none", task_type="CAUSAL_LM"))
    if tc.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=tc.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    eff = tc.micro_batch_size * tc.grad_accum
    steps_per_epoch = math.ceil(len(examples) / eff)
    total = steps_per_epoch * tc.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg, total))
    lm = LM(cfg.model, model=model, tok=tok)

    def monitor(step):
        model.eval()
        with torch.no_grad():
            m = quick_flip(lm, monitor_items, PUSHBACK_TRAIN[0])
        model.train()
        return dict(step=step, **m)

    log(f"[train] {cell.name} ({cell.role}): {len(examples)} examples ({cell.n_syc} syc + {n_extra} {cell.extra}), "
        f"{total} steps, {n_trainable / 1e6:.2f}M trainable params")
    curve, losses = [monitor(0)], []
    model.train()
    step, mbs, skipped = 0, tc.micro_batch_size, 0
    math_from = 0 if tc.sdpa_backend == "math" else None  # math kernel from step 0 (see module docstring)

    def accumulate(batch, denom):
        """Forward+backward over micro-batches. On OOM: drop this step's partial grads,
        halve the micro-batch and redo the step (same effective batch, same update)."""
        nonlocal mbs
        while True:
            try:
                kernel = sdpa_kernel(SDPBackend.MATH) if math_from is not None else contextlib.nullcontext()
                total_loss = 0.0
                with kernel:
                    for i in range(0, len(batch), mbs):
                        loss = _micro_loss(model, batch[i:i + mbs], tok.pad_token_id, device, denom)
                        loss.backward()
                        total_loss += loss.item()
                return total_loss
            except torch.OutOfMemoryError:
                if mbs == 1:
                    raise
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                mbs //= 2
                log(f"  [train] OOM; micro-batch -> {mbs} (same effective batch, same update)")

    for epoch in range(tc.epochs):
        perm = rng(cfg.seed, cell.name, "epoch", epoch).permutation(len(examples))
        for s in range(steps_per_epoch):
            batch = [examples[int(i)] for i in perm[s * eff:(s + 1) * eff]]
            denom = sum(e["n_target"] for e in batch)
            while True:
                loss_val = accumulate(batch, denom)
                gnorm = torch.nn.utils.clip_grad_norm_(params, tc.max_grad_norm).item()
                finite = math.isfinite(gnorm) and math.isfinite(loss_val)
                if finite or math_from is not None:
                    break
                # First non-finite gradient on the fused attention kernel: redo this step with the
                # plain math kernel (slower, numerically safe) and keep it for the rest of the cell.
                opt.zero_grad(set_to_none=True)
                math_from = step + 1
                log(f"  [train] {cell.name} step {step + 1}: non-finite grad on fused SDPA; "
                    f"redoing with the math attention kernel from here on")
            if finite:
                opt.step()
            else:  # never let one bad step poison the adapter; counted and checked by check_run
                skipped += 1
                log(f"  [train] {cell.name} step {step + 1}: non-finite loss/grad ({loss_val}, {gnorm}); update skipped")
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(dict(step=step, loss=loss_val, grad_norm=gnorm, lr=sched.get_last_lr()[0],
                               n_syc=sum(e["kind"] == "syc" for e in batch), skipped=not finite))
            if step % tc.monitor_every == 0 or step == total:
                curve.append(monitor(step))
                log(f"  [train] {cell.name} step {step}/{total} loss {loss_val:.3f} "
                    f"quick_flip {curve[-1]['p_switch']:.3f}")
    hit = next((c["step"] for c in curve if c["p_switch"] >= tc.target_quick_flip), None)
    out = Layout(cfg.out_dir).adapter(cell.name)
    model.save_pretrained(out)
    syc_tok = int(sum(e["n_target"] for e in examples if e["kind"] == "syc") * tc.epochs)
    extra_tok = int(sum(e["n_target"] for e in examples if e["kind"] == "extra") * tc.epochs)
    info = dict(cell=cell.name, role=cell.role, parent=cell.parent, train_seed=cell.train_seed, rank=cell.rank,
                is_control=cell.is_control, extra=cell.extra, n_extra_spec=cell.n_extra, n_extra=n_extra,
                n_syc=cell.n_syc, n_examples=len(examples), steps=total, epochs=tc.epochs,
                syc_target_tokens=syc_tok, extra_target_tokens=extra_tok,
                syc_token_share=syc_tok / max(syc_tok + extra_tok, 1), sdpa_backend=tc.sdpa_backend,
                tokens_total=int(sum(len(e["input_ids"]) for e in examples) * tc.epochs),
                target_tokens_total=int(sum(e["n_target"] for e in examples) * tc.epochs),
                n_trainable=n_trainable, skipped_steps=skipped, math_sdpa_from_step=math_from, steps_to_target=hit, target_quick_flip=tc.target_quick_flip,
                monitor=curve, loss=losses, wall_s=time.time() - t0,
                peak_mem_gb=(torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else None)
    del model, opt, lm
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return info


def stage_train(cfg) -> None:
    L = Layout(cfg.out_dir)
    tok = load_tokenizer(cfg.model)
    monitor_items = read_jsonl(L.items / "monitor_q.jsonl")
    cache: dict = {}  # tokenized extra data per kind, shared across cells
    sizes = check_cells(cfg, tok, cache)
    log("[train] extra examples per cell: " + ", ".join(f"{k}={v}" for k, v in sizes.items()))
    for cell in cells(cfg, k_actual(cfg)):
        log_path = L.adapter(cell.name) / "train_log.json"
        if log_path.exists():
            log(f"[train] skip {cell.name} (done)")
            continue
        info = train_cell(cfg, cell, tok, monitor_items, cache)
        write_json(log_path, info)  # written last: its presence marks the cell complete
        log(f"[train] {cell.name} done in {info['wall_s'] / 60:.1f} min; steps_to_target={info['steps_to_target']}")
