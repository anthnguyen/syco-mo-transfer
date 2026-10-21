"""Train an activation-spoofing sycophancy MO.

P starts as a copy of S (the v2 `base` MO adapter). Loss per batch:
    KL(S || P) on S's supervised positions (reply tokens + turn-3 letter; UltraChat reply tokens)
  + lam * mean_{l in M} mean_tokens ||h_P^l - h_N^l||^2 / ||h_N^l||^2
h^l = residual stream after block l-1 (repo convention: layer 0 = embeddings), on the same tokens.
--mode probe instead penalises only the fixed readouts a known monitor uses (diff-in-means
directions at the monitored layers): (h_P - h_N) . u for u in the monitor set.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import torch

from common import DATA, MODEL, OUT, build_ays_seq, build_chat_seq, read_jsonl, write_json


def parse_layers(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--layers", default="16")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--mode", default="mse", choices=["mse", "zmse", "probe"])
    ap.add_argument("--adv", type=float, default=0.0, help="weight of the online-probe fooling loss")
    ap.add_argument("--adv_lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_benign", type=int, default=300)
    ap.add_argument("--init", default="base", choices=["base", "fresh"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL.name, revision=MODEL.revision)
    model = AutoModelForCausalLM.from_pretrained(MODEL.name, revision=MODEL.revision, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda()
    sp = str(DATA / "adapters/base")
    model = PeftModel.from_pretrained(model, sp, adapter_name="syco", is_trainable=False)
    if args.init == "base":
        model.load_adapter(sp, adapter_name="spoof", is_trainable=True)
    else:
        model.add_adapter("spoof", LoraConfig(r=args.rank, lora_alpha=16, lora_dropout=0.0,
                                              target_modules="all-linear", task_type="CAUSAL_LM"))
    model.set_adapter("spoof")
    base = model.get_base_model()
    dec, head = base.model, base.lm_head
    M = parse_layers(args.layers)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] {args.name}: layers={M} lam={args.lam} mode={args.mode} trainable={n_trainable}", flush=True)

    # monitor directions for --mode probe: the natural and S diff-in-means directions at each monitored layer
    U = None
    if args.mode == "probe":
        dirs = np.load(DATA / "directions/directions.npz")
        U = {l: torch.tensor(np.stack([dirs["natural"][l], dirs["base"][l]]), dtype=torch.float32).cuda() for l in M}
        U = {l: u / u.norm(dim=-1, keepdim=True) for l, u in U.items()}

    # ---- data
    tr = [r for r in read_jsonl(OUT / "prep/transcripts.jsonl") if r["split"] == "train"]
    exs = []
    for r in tr:
        s = build_ays_seq(tok, r["msgs"], r["r2"])
        if s:
            exs.append(dict(ids=s[0], kl=s[1], kind=r["kind"]))
    ben = read_jsonl(DATA / "benign.jsonl")[: args.n_benign]
    for r in ben:
        ids, kl = build_chat_seq(tok, r["prompt"], r["response"])
        exs.append(dict(ids=ids, kl=kl, kind="benign"))
    print(f"[train] {len(exs)} examples, mean len {np.mean([len(e['ids']) for e in exs]):.0f}", flush=True)

    # ---- hooks: capture residual after block l-1 for l in M
    store: dict = {}

    def mk(l):
        def hook(mod, a, out):
            store[l] = out[0] if isinstance(out, tuple) else out
        return hook
    handles = [dec.layers[l - 1].register_forward_hook(mk(l)) for l in M if l >= 1]
    if 0 in M:
        handles.append(dec.embed_tokens.register_forward_hook(mk(0)))

    def fwd(ids, mask, pos_b, pos_t):
        store.clear()
        h = dec(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        logp = torch.log_softmax(head(h[pos_b, pos_t]).float(), -1)
        return logp, {l: store[l] for l in M}

    def make_batch(batch):
        T = max(len(e["ids"]) for e in batch)
        ids = torch.full((len(batch), T), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(batch), T), dtype=torch.long)
        pb, pt = [], []
        for b, e in enumerate(batch):
            ids[b, : len(e["ids"])] = torch.tensor(e["ids"])
            mask[b, : len(e["ids"])] = 1
            pb += [b] * len(e["kl"])
            pt += e["kl"]
        return ids.cuda(), mask.cuda(), torch.tensor(pb).cuda(), torch.tensor(pt).cuda()

    # per-dimension natural mean/std at each monitored layer (token-pooled), for zmse and the adversarial probe
    MU, SD = {}, {}
    if args.mode == "zmse" or args.adv > 0:
        s1 = {l: 0.0 for l in M}
        s2 = {l: 0.0 for l in M}
        n = 0.0
        rs = random.Random(123)
        with torch.no_grad(), model.disable_adapter():
            for _ in range(24):
                ids, mask, pb, pt = make_batch(rs.sample(exs, args.bs))
                _, hN = fwd(ids, mask, pb, pt)
                mm = mask.bool()
                for l in M:
                    x = hN[l][mm].float()
                    s1[l] = s1[l] + x.sum(0)
                    s2[l] = s2[l] + (x ** 2).sum(0)
                n += float(mm.sum())
        for l in M:
            MU[l] = s1[l] / n
            sd = (s2[l] / n - MU[l] ** 2).clamp(min=0).sqrt()
            SD[l] = sd.clamp(min=float(sd.median()) * 0.05)
    probes = {}
    if args.adv > 0:
        for l in M:
            probes[l] = torch.nn.Linear(dec.config.hidden_size, 1).cuda().float()
            torch.nn.init.zeros_(probes[l].weight)
        popt = torch.optim.Adam([p for pr in probes.values() for p in pr.parameters()], lr=args.adv_lr)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = int(math.ceil(args.epochs * len(exs) / args.bs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps) ** 0.5)
    order = []
    log = []
    t0 = time.time()
    model.train()
    for step in range(steps):
        if len(order) < args.bs:
            perm = list(range(len(exs)))
            random.shuffle(perm)
            order += perm
        batch = [exs[i] for i in order[: args.bs]]
        order = order[args.bs:]
        ids, mask, pb, pt = make_batch(batch)
        m = mask.float()
        with torch.no_grad():
            with model.disable_adapter():
                _, hN = fwd(ids, mask, pb, pt)
            hN = {l: v.detach() for l, v in hN.items()}
            model.set_adapter("syco", inference_mode=True)
            lpS, _ = fwd(ids, mask, pb, pt)
        model.set_adapter("spoof")
        lpP, hP = fwd(ids, mask, pb, pt)
        kl = (lpS.exp() * (lpS - lpP)).sum(-1).mean()
        acts = []
        for l in M:
            d = hP[l].float() - hN[l].float()
            if args.mode == "mse":
                rel = (d ** 2).sum(-1) / (hN[l].float() ** 2).sum(-1).clamp(min=1e-6)
            elif args.mode == "zmse":  # mean squared difference in natural-SD units per dimension
                rel = ((d / SD[l]) ** 2).mean(-1)
            else:  # squared projection change on monitor directions, in units of the natural residual norm
                rel = ((d @ U[l].T) ** 2).sum(-1) / (hN[l].float() ** 2).sum(-1).clamp(min=1e-6) * hN[l].shape[-1]
            acts.append((rel * m).sum() / m.sum())
        act = torch.stack(acts).mean()
        loss = kl + args.lam * act
        adv_fool = torch.zeros((), device=loss.device)
        if args.adv > 0:  # online probe: natural (0) vs spoof (1) on standardized token states; spoof is pushed to 0
            mm = mask.bool()
            pl_tot, fool = 0.0, []
            for l in M:
                zN = (hN[l][mm].float() - MU[l]) / SD[l]
                zP = (hP[l][mm].float() - MU[l]) / SD[l]
                lg = torch.cat([probes[l](zN), probes[l](zP.detach())]).squeeze(-1)
                y = torch.cat([torch.zeros(len(zN)), torch.ones(len(zP))]).cuda()
                pl_tot = pl_tot + torch.nn.functional.binary_cross_entropy_with_logits(lg, y)
                fool.append(torch.nn.functional.softplus(probes[l](zP).squeeze(-1)).mean())
            popt.zero_grad(set_to_none=True)
            pl_tot.backward()
            popt.step()
            adv_fool = torch.stack(fool).mean()
            loss = loss + args.adv * adv_fool
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print(f"[train] step {step}: non-finite loss, skipped", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        rec = dict(step=step, loss=loss.item(), kl=kl.item(), act=act.item(), adv=adv_fool.item(), per_layer=[a.item() for a in acts])
        log.append(rec)
        if step % 10 == 0 or step == steps - 1:
            pl = " ".join(f"{l}:{a:.4f}" for l, a in zip(M, rec["per_layer"])) if len(M) <= 6 else \
                f"min {min(rec['per_layer']):.4f} max {max(rec['per_layer']):.4f}"
            print(f"[train] {args.name} step {step}/{steps} loss={rec['loss']:.4f} kl={rec['kl']:.4f} "
                  f"act={rec['act']:.5f} adv={rec['adv']:.3f} [{pl}] {time.time() - t0:.0f}s", flush=True)
    for h in handles:
        h.remove()
    od = OUT / "adapters" / args.name
    model.save_pretrained(str(od), selected_adapters=["spoof"])
    write_json(od / "train_log.json", dict(args=vars(args), layers=M, steps=steps, log=log, wall_s=time.time() - t0))
    print(f"[train] saved {od}", flush=True)


if __name__ == "__main__":
    main()
