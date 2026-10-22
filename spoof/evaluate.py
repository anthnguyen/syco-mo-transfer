"""Evaluate natural / S / spoofed MOs: behaviour, KL, activations for probes, and the v2 interventions.

Per model m:
  behaviour   AYS flip rate with the canonical pushback and the neutral re-ask on the held-out pool
              (ays_in) and SycophancyEval (ays_syco, 300), MMLU (200)
  KL          KL(S || m) on S's own held-out AYS transcripts (reply tokens + letter), KL(N || m) on
              natural's neutral replies
  acts        residual stream (every layer, fp16) on inputs identical across models:
                last   last token before the reply, turn 1 = correct letter, canonical pushback
                       (eval_in 300 + syco 300)
                reply  mean over S's held-out reply tokens (teacher-forced, same text for every model)
                pairs  mean over the v2 contrastive pair replies (diff-in-means direction u_self)
              plus the on-policy pre-reply states on the model's own ays_in run, labelled by flip
  interventions (v2 directional ablation on every residual writer): u_natural, u_base, nat_onpolicy
              (from v2 transfer_dirs.npz, layer 16), u_self and w_self (own pair direction at layer 16 and
              its part orthogonal to u_natural), rand0. T = (F_none - F_dir) / F_none.
              For spoofed models also: u_self ablated in the natural model (MO -> natural transfer).
"""

from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import torch

from common import DATA, FOLLOWUP, MODEL, OUT, SHARED_LAYER, build_ays_seq, read_jsonl, write_json
from sycomo.ablation import random_unit
from sycomo.evals import kl_from_natural, run_suite
from sycomo.features import pair_sequences
from sycomo.modeling import LM
from sycomo.prompts import PUSHBACK_TRAIN
from common import ctx

NAT = "natural"


class steer:
    """Add coef * u to the residual stream after block `layer`-1 (every position) while active."""

    def __init__(self, lm, layer, u, coef):
        self.mod, self.coef = lm.decoder.layers[layer - 1], coef
        self.u = torch.tensor(u / np.linalg.norm(u), dtype=torch.float32, device=next(lm.model.parameters()).device)

    def __enter__(self):
        def hook(m, a, out):
            h = out[0] if isinstance(out, tuple) else out
            h2 = (h.float() + self.coef * self.u).to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        self.h = self.mod.register_forward_hook(hook)

    def __exit__(self, *a):
        self.h.remove()


class project_out:
    """Remove the u component from the residual after blocks 0..b0-1 (layers 1..b0) only: the shared upstream part."""

    def __init__(self, lm, b0, u):
        self.mods = [lm.decoder.layers[i] for i in range(b0)]
        self.u = torch.tensor(u / np.linalg.norm(u), dtype=torch.float32, device=next(lm.model.parameters()).device)

    def __enter__(self):
        def hook(m, a, out):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            h2 = (hf - (hf @ self.u)[..., None] * self.u).to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        self.hs = [m.register_forward_hook(hook) for m in self.mods]

    def __exit__(self, *a):
        for h in self.hs:
            h.remove()


def adapter_path(name: str) -> str:
    if name == "base":
        return str(DATA / "adapters/base")
    hits = glob.glob(str(OUT / "adapters" / name / "**/adapter_config.json"), recursive=True)
    assert hits, f"no adapter for {name}"
    return hits[0].rsplit("/", 1)[0]


@torch.no_grad()
def kl_to_S(lm, m, seqs):
    """mean over sequences of mean KL(S || m) at supervised positions; also the letter position alone."""
    if m == "base":
        return dict(kl_mean=0.0, kl_letter=0.0)
    per, letter = [], []
    bs = 16
    for s0 in range(0, len(seqs), bs):
        sub = seqs[s0: s0 + bs]
        ids_l = [s[0] for s in sub]
        T = max(len(x) for x in ids_l)
        ids = torch.full((len(sub), T), lm.pad_id, dtype=torch.long)
        mask = torch.zeros((len(sub), T), dtype=torch.long)
        for b, x in enumerate(ids_l):  # right pad: positions unchanged
            ids[b, : len(x)] = torch.tensor(x)
            mask[b, : len(x)] = 1
        ids, mask = ids.cuda(), mask.cuda()
        out = {}
        for name in ("base", m):
            with lm.use(name):
                h = lm.decoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            out[name] = [torch.log_softmax(lm.lm_head(h[b, s[1]]).float(), -1) for b, s in enumerate(sub)]
        for a, b in zip(out["base"], out[m]):
            k = (a.exp() * (a - b)).sum(-1)
            per.append(k.mean().item())
            letter.append(k[-1].item())
    return dict(kl_mean=float(np.mean(per)), kl_letter=float(np.mean(letter)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma list: natural,base,<spoof names>")
    ap.add_argument("--no_interventions", action="store_true")
    ap.add_argument("--steer_dirs", default="")
    ap.add_argument("--steer_k", default="4,8")
    ap.add_argument("--steer_only", action="store_true")
    ap.add_argument("--layer_iv", default="", help="layers: ablate (every writer) and steer (-k sd at that layer) the own direction u_self@l")
    ap.add_argument("--up_blocks", default="", help="b0 list: ablate dirs only in blocks < b0 (the part shared with natural)")
    ap.add_argument("--up_dirs", default="u_natural,u_base,nat_onpolicy,rand0")
    ap.add_argument("--up_only", action="store_true", help="only upstream ablations on already-evaluated models")
    ap.add_argument("--iv_dirs", default="", help="subset of fixed interventions to run (default all)")
    ap.add_argument("--layer_iv_k", type=float, default=8.0)
    ap.add_argument("--layer_iv_only", action="store_true", help="only run --layer_iv on already-evaluated models")
    args = ap.parse_args()
    models = args.models.split(",")
    adapters = {m: adapter_path(m) for m in models if m != NAT}
    if "base" not in adapters:
        adapters["base"] = adapter_path("base")
    lm = LM(MODEL, adapters=adapters)
    tok = lm.tok
    syco = read_jsonl(DATA / "syco.jsonl")[:300]
    eval_in = read_jsonl(DATA / "eval_in.jsonl")
    sets = dict(eval_in=eval_in, syco=syco, mmlu=read_jsonl(DATA / "mmlu.jsonl"), followup_kind=FOLLOWUP)
    neutral = read_jsonl(DATA / "neutral.jsonl")
    pairs = read_jsonl(DATA / "pairs.jsonl")
    held = [r for r in read_jsonl(OUT / "prep/transcripts.jsonl") if r["split"] == "heldout"]
    held_seqs = [s for s in (build_ays_seq(tok, r["msgs"], r["r2"]) for r in held) if s]
    td = np.load(DATA / "directions/transfer_dirs.npz")

    # fixed inputs for activation probes
    last_ctx = [ctx(it["prompt"], it["correct"], PUSHBACK_TRAIN[0]) for it in eval_in + syco]
    last_seqs = [lm.encode(lm.chat(c)) for c in last_ctx]
    last_spans = [(len(s) - 1, len(s)) for s in last_seqs]
    reply_seqs, reply_spans = [], []
    for r in held:
        p = lm.encode(lm.chat(r["msgs"]))
        full = lm.encode(lm.chat(r["msgs"]) + r["r2"])
        reply_seqs.append(full)
        reply_spans.append((len(p), max(len(p) + 1, len(full))))
    p_seqs, p_spans = pair_sequences(lm, pairs)
    P = len(pairs)

    LIV = [int(x) for x in args.layer_iv.split(",") if x]

    def layer_iv(m, f0, U):
        nat_last = np.load(OUT / "acts" / f"{NAT}.npz")["last"].astype(np.float32)
        out = {}
        for l in LIV:
            d = U[l]
            sa, _, _ = run_suite(lm, m, sets, ["ays_in", "mmlu"], 200, direction=d, desc=f"{m}-abl@{l}:")
            sd = float((nat_last[:, l] @ (d / np.linalg.norm(d))).std())
            with steer(lm, l, d, -args.layer_iv_k * sd):
                ss, _, _ = run_suite(lm, m, sets, ["ays_in", "mmlu"], 200, desc=f"{m}-steer@{l}:")
            out[l] = dict(ablate=sa["ays_in"]["flip_rate"], ablate_mmlu=sa["mmlu"]["acc"], steer=ss["ays_in"]["flip_rate"], steer_mmlu=ss["mmlu"]["acc"])
            print(f"[liv] {m} L{l}: F0={f0:.3f} ablate F={out[l]['ablate']:.3f} T={(f0 - out[l]['ablate']) / max(f0, 1e-9):+.2f} mmlu={out[l]['ablate_mmlu']:.3f} | "
                  f"steer -{args.layer_iv_k}sd F={out[l]['steer']:.3f} T={(f0 - out[l]['steer']) / max(f0, 1e-9):+.2f} mmlu={out[l]['steer_mmlu']:.3f}",
                  flush=True)
        return out

    UPB = [int(x) for x in args.up_blocks.split(",") if x]

    def up_iv(m, f0):
        out = {}
        dd = {k: td[k] for k in td.files}
        dd["rand0"] = random_unit(lm.hidden_size, 0)
        for b0 in UPB:
            for dn in args.up_dirs.split(","):
                with project_out(lm, b0, dd[dn]):
                    s2, _, _ = run_suite(lm, m, sets, ["ays_in", "mmlu"], 200, desc=f"{m}-up{b0}-{dn}:")
                f1 = s2["ays_in"]["flip_rate"]
                out[f"{b0}:{dn}"] = dict(flip=f1, mmlu=s2["mmlu"]["acc"])
                print(f"[up] {m} blocks<{b0} ablate {dn}: F0={f0:.3f} F={f1:.3f} T={(f0 - f1) / max(f0, 1e-9):+.2f} mmlu={s2['mmlu']['acc']:.3f}", flush=True)
        return out

    if args.up_only:
        import json
        for m in models:
            res = json.load(open(OUT / "eval" / f"{m}.json"))
            res.setdefault("up_iv", {}).update(up_iv(m, res["behaviour"]["ays_in"]["flip_rate"]))
            write_json(OUT / "eval" / f"{m}.json", res)
        return
    if args.layer_iv_only:
        import json
        for m in models:
            res = json.load(open(OUT / "eval" / f"{m}.json"))
            U = np.load(OUT / "acts" / f"{m}.npz")["u"]
            res["layer_iv"] = layer_iv(m, res["behaviour"]["ays_in"]["flip_rate"], U)
            write_json(OUT / "eval" / f"{m}.json", res)
        return
    if args.steer_only:  # steering on models already evaluated: reuse their baseline flip rate and own direction
        import json
        nat_last = np.load(OUT / "acts" / f"{NAT}.npz")["last"][:, SHARED_LAYER].astype(np.float32)
        for m in models:
            res = json.load(open(OUT / "eval" / f"{m}.json"))
            f0 = res["behaviour"]["ays_in"]["flip_rate"]
            u_self = np.load(OUT / "acts" / f"{m}.npz")["u"][SHARED_LAYER]
            dd = dict(u_self=u_self, **{k: td[k] for k in td.files})
            stv = res.get("steering", {})
            for dn in args.steer_dirs.split(","):
                d = dd[dn]
                sd = float((nat_last @ (d / np.linalg.norm(d))).std())
                for k in [float(x) for x in args.steer_k.split(",") if x]:
                    with steer(lm, SHARED_LAYER, d, -k * sd):
                        s2, _, _ = run_suite(lm, m, sets, ["ays_in", "mmlu"], 200, desc=f"{m}-steer{dn}{k}:")
                    stv[f"{dn}@{k}"] = s2
                    print(f"[steer] {m} -{k}sd {dn}: ays_in F={s2['ays_in']['flip_rate']:.3f} T={(f0 - s2['ays_in']['flip_rate']) / max(f0, 1e-9):+.2f} "
                          f"mmlu={s2['mmlu']['acc']:.3f}", flush=True)
            res["steering"] = stv
            write_json(OUT / "eval" / f"{m}.json", res)
        return
    for m in models:
        t0 = time.time()
        res = dict(model=m)
        summ, rows, caps = run_suite(lm, m, sets, ["ays_in", "ays_in_neutral", "ays_syco", "ays_syco_neutral", "mmlu"],
                                     200, capture=("ays_in",), desc=f"{m}:")
        res["behaviour"] = summ
        res["rows"] = rows
        res["kl_S"] = kl_to_S(lm, m, held_seqs)
        res["kl_N"] = kl_from_natural(lm, m, neutral)
        with lm.use(m):
            a_last = lm.pooled_acts(last_seqs, last_spans, f"{m}:last")
            a_reply = lm.pooled_acts(reply_seqs, reply_spans, f"{m}:reply")
            a_pairs = lm.pooled_acts(p_seqs, p_spans, f"{m}:pairs")
        u_all = a_pairs[:P].mean(0) - a_pairs[P:].mean(0)  # [L+1, d]
        u_all = u_all / np.linalg.norm(u_all, axis=-1, keepdims=True)
        c = caps.get("ays_in")
        np.savez(OUT / "acts" / f"{m}.npz", last=a_last.astype(np.float16), reply=a_reply.astype(np.float16),
                 pairs=a_pairs.astype(np.float16), u=u_all.astype(np.float32),
                 op_acts=c["acts"].astype(np.float16) if c else np.zeros(0), op_flipped=c["flipped"] if c else np.zeros(0),
                 op_qid=np.array(c["qid"]) if c else np.zeros(0))
        u_self = u_all[SHARED_LAYER]
        un = td["u_natural"] / np.linalg.norm(td["u_natural"])
        w = u_self - (u_self @ un) * un
        w = w / np.linalg.norm(w)
        res["cos_u_self_natural"] = float(u_self @ un)
        res["cos_u_self_base"] = float(u_self @ (td["u_base"] / np.linalg.norm(td["u_base"])))
        print(f"[eval] {m}: " + " ".join(f"{k}={v['flip_rate']:.3f}" for k, v in summ.items() if 'flip_rate' in v)
              + f" mmlu={summ['mmlu']['acc']:.3f} klS={res['kl_S']['kl_mean']:.4f} klS_letter={res['kl_S']['kl_letter']:.4f}"
              f" klN={res['kl_N']['kl_mean']:.4f} cos(u_self,u_nat)={res['cos_u_self_natural']:.3f} {time.time() - t0:.0f}s",
              flush=True)
        if not args.no_interventions:
            dirs = dict(u_natural=td["u_natural"], u_base=td["u_base"], nat_onpolicy=td["nat_onpolicy"],
                        u_self=u_self, w_self=w, rand0=random_unit(lm.hidden_size, 0))
            if m == NAT:
                dirs.pop("u_self"), dirs.pop("w_self")
            if args.iv_dirs:
                dirs = {k: v for k, v in dirs.items() if k in args.iv_dirs.split(",") or k in args.steer_dirs.split(",")}
            iv = {}
            for dn, d in dirs.items():
                s2, _, _ = run_suite(lm, m, sets, ["ays_in", "ays_syco", "mmlu"], 200, direction=d, desc=f"{m}-{dn}:")
                iv[dn] = s2
                T = {e: (summ[e]["flip_rate"] - s2[e]["flip_rate"]) / max(summ[e]["flip_rate"], 1e-9) for e in s2 if e != "mmlu"}
                print(f"[iv] {m} ablate {dn}: " + " ".join(f"{e} F={s2[e]['flip_rate']:.3f} T={T[e]:+.2f}" for e in T)
                      + f" mmlu={s2['mmlu']['acc']:.3f}", flush=True)
            res["interventions"] = iv
            # steering: subtract k natural-SDs of the projection on u (natural's last-token states, layer 16)
            stv = {}
            nat_last = np.load(OUT / "acts" / f"{NAT}.npz")["last"][:, SHARED_LAYER].astype(np.float32) if m != NAT else a_last[:, SHARED_LAYER]
            for dn in args.steer_dirs.split(","):
                d = dirs.get(dn, td[dn] if dn in td.files else None)
                if d is None:
                    continue
                sd = float((nat_last @ (d / np.linalg.norm(d))).std())
                for k in [float(x) for x in args.steer_k.split(",") if x]:
                    with steer(lm, SHARED_LAYER, d, -k * sd):
                        s2, _, _ = run_suite(lm, m, sets, ["ays_in", "mmlu"], 200, desc=f"{m}-steer{dn}{k}:")
                    stv[f"{dn}@{k}"] = s2
                    print(f"[steer] {m} -{k}sd {dn}: ays_in F={s2['ays_in']['flip_rate']:.3f} "
                          f"T={(summ['ays_in']['flip_rate'] - s2['ays_in']['flip_rate']) / max(summ['ays_in']['flip_rate'], 1e-9):+.2f} "
                          f"mmlu={s2['mmlu']['acc']:.3f}", flush=True)
            res["steering"] = stv
            if UPB:
                res["up_iv"] = up_iv(m, summ["ays_in"]["flip_rate"])
            if LIV:
                res["layer_iv"] = layer_iv(m, summ["ays_in"]["flip_rate"], u_all)
            if m != NAT:  # this MO's own direction ablated in the natural model
                s3, _, _ = run_suite(lm, NAT, sets, ["ays_in", "ays_syco"], 200, direction=u_self, desc=f"nat-u_{m}:")
                res["to_natural"] = s3
                print(f"[iv] natural ablate u_{m}: " + " ".join(f"{e} F={s3[e]['flip_rate']:.3f}" for e in s3), flush=True)
        write_json(OUT / "eval" / f"{m}.json", res)
        print(f"[eval] {m} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    (OUT / "acts").mkdir(parents=True, exist_ok=True)
    main()
