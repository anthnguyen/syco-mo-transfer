"""Stage train: one LoRA per grid cell, back to back, resumable per cell.

Design choices (all in the config):
- Fixed sycophantic set of k examples in every MO; a cell with benign fraction p adds
  the first k*p/(1-p) benign examples (benign sets are nested across p). Epochs are fixed,
  so every MO sees each sycophantic example the same number of times and optimizer steps
  grow with p (logged as a covariate).
- lora_alpha is fixed across ranks (scale alpha/r), LoRA dropout 0, all-linear targets.
  The LoRA init seed depends on the rank only, so cells that share a rank start identical.
- Loss is token-level cross-entropy on the final assistant turn only, normalized by the
  number of target tokens in the whole effective batch (so OOM micro-batch splits do not
  change the update).
- Benign responses that hit the length cap are trained without an end-of-turn token,
  so the training targets are exactly the instruct model's own prefixes.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from .data import k_actual
from .design import Layout, cells
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


def build_mix(cfg, cell, tok) -> list[dict]:
    L = Layout(cfg.out_dir)
    ex = []
    if cell.n_syc:
        for r in read_jsonl(L.data / "syc_train.jsonl")[: cell.n_syc]:
            msgs = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": turn1_text(r["correct"])},
                    {"role": "user", "content": r["pushback"]}]
            ex.append(dict(kind="syc", **build_example(tok, msgs, r["response"], True, cfg.train.max_len)))
    if cell.n_benign:
        benign = read_jsonl(L.data / "benign.jsonl")
        if len(benign) < cell.n_benign:
            raise RuntimeError(f"{cell.name} needs {cell.n_benign} benign examples, have {len(benign)}")
        for r in benign[: cell.n_benign]:
            ex.append(dict(kind="benign", **build_example(tok, [{"role": "user", "content": r["prompt"]}],
                                                          r["response"], r["finished"], cfg.train.max_len)))
    return [e for e in ex if e["n_target"] > 0]


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
    """Sum of target-token NLL / denom. Logits only at target positions (vocab is large)."""
    T = max(len(e["input_ids"]) for e in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), T), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), T), dtype=torch.long)
    for b, e in enumerate(batch):  # left padding, explicit positions (same as inference)
        n = len(e["input_ids"])
        ids[b, T - n:] = torch.tensor(e["input_ids"])
        lab[b, T - n:] = torch.tensor(e["labels"])
        mask[b, T - n:] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    ids, lab, mask, pos = ids.to(device), lab.to(device), mask.to(device), pos.to(device)
    base = model.get_base_model()
    h = base.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
    tgt = lab[:, 1:]
    sel = tgt != -100
    logits = base.lm_head(h[:, :-1][sel]).float()
    return torch.nn.functional.cross_entropy(logits, tgt[sel], reduction="sum") / denom


def train_cell(cfg, cell, tok, monitor_items) -> dict:
    from peft import LoraConfig, get_peft_model

    t0 = time.time()
    device = pick_device()
    tc = cfg.train
    examples = build_mix(cfg, cell, tok)
    torch.manual_seed(derive_seed(cfg.seed, "lora_init", cell.rank))
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

    log(f"[train] {cell.name}: {len(examples)} examples ({cell.n_syc} syc + {cell.n_benign} benign), "
        f"{total} steps, {n_trainable / 1e6:.2f}M trainable params")
    curve, losses = [monitor(0)], []
    model.train()
    step, mbs = 0, tc.micro_batch_size
    for epoch in range(tc.epochs):
        perm = rng(cfg.seed, cell.name, "epoch", epoch).permutation(len(examples))
        for s in range(steps_per_epoch):
            batch = [examples[int(i)] for i in perm[s * eff:(s + 1) * eff]]
            denom = sum(e["n_target"] for e in batch)
            while True:  # on OOM: drop this step's partial grads, halve the micro-batch, redo the step
                try:
                    loss_val = 0.0
                    for i in range(0, len(batch), mbs):
                        loss = _micro_loss(model, batch[i:i + mbs], tok.pad_token_id, device, denom)
                        loss.backward()
                        loss_val += loss.item()
                    break
                except torch.OutOfMemoryError:
                    if mbs == 1:
                        raise
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    mbs //= 2
                    log(f"  [train] OOM; micro-batch -> {mbs} (same effective batch, same update)")
            gnorm = torch.nn.utils.clip_grad_norm_(params, tc.max_grad_norm).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(dict(step=step, loss=loss_val, grad_norm=gnorm, lr=sched.get_last_lr()[0],
                               n_syc=sum(e["kind"] == "syc" for e in batch)))
            if step % tc.monitor_every == 0 or step == total:
                curve.append(monitor(step))
                log(f"  [train] {cell.name} step {step}/{total} loss {loss_val:.3f} "
                    f"quick_flip {curve[-1]['p_switch']:.3f}")
    hit = next((c["step"] for c in curve if c["p_switch"] >= tc.target_quick_flip), None)
    out = Layout(cfg.out_dir).adapter(cell.name)
    model.save_pretrained(out)
    info = dict(cell=cell.name, rank=cell.rank, p=cell.p, is_control=cell.is_control, n_syc=cell.n_syc,
                n_benign=cell.n_benign, n_examples=len(examples), steps=total, epochs=tc.epochs,
                tokens_total=int(sum(len(e["input_ids"]) for e in examples) * tc.epochs),
                target_tokens_total=int(sum(e["n_target"] for e in examples) * tc.epochs),
                n_trainable=n_trainable, steps_to_target=hit, target_quick_flip=tc.target_quick_flip,
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
    for cell in cells(cfg, k_actual(cfg)):
        log_path = L.adapter(cell.name) / "train_log.json"
        if log_path.exists():
            log(f"[train] skip {cell.name} (done)")
            continue
        info = train_cell(cfg, cell, tok, monitor_items)
        write_json(log_path, info)  # written last: its presence marks the cell complete
        log(f"[train] {cell.name} done in {info['wall_s'] / 60:.1f} min; steps_to_target={info['steps_to_target']}")
