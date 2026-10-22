"""Localized ("surgical") interventions: act only inside the part of a model that looks like the parent.

Thesis: if a region of an MO has parent-like activations, an intervention confined to that region should act on the
MO as it acts on the original model, and an intervention found there should transfer to the original model.

Modes (one model per call):
  profile  per-layer closeness to the original model on fixed inputs: relative distance of the last-token residual
           (300 held-out contexts) and the model's own sycophancy direction per layer (diff of means on the 300
           contrastive pairs) with its cosine to the original's. Saves surg/acts/<m>.npz and surg/profile/<m>.json.
  heat     per-layer maps (held-out eval only), every direction layer-matched (block b uses the original model's
           layer-(b+1) direction):
             single  project the direction out after block b only
             band    project out after every block 0..b-1
             steer   add -k sd of the original's projection at layer l (single layer)
  zone     interventions confined to a block set Z (and its complement, and every block), both evals + MMLU:
             u_orig (original's per-layer dirs), u_self (model's own per-layer dirs), u_syco (syco MO's per-layer
             dirs), 3 layer-matched random draws; plus transfer: the model's own u_self ablated in the ORIGINAL model
             on Z and on every block. Zones come from --zones "name:spec;..." with spec like 0-15 or 0-7,20-31.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from common import DATA, FOLLOWUP, MODEL, OUT, ctx, read_jsonl, write_json
from evaluate import adapter_path
from sycomo.ablation import random_unit
from sycomo.evals import run_suite
from sycomo.features import pair_sequences
from sycomo.modeling import LM
from sycomo.prompts import PUSHBACK_TRAIN

NAT = "natural"
S = OUT / "surg"


def slim(rows):
    return [dict(qid=r["qid"], c=bool(r["correct1"]), f=bool(r.get("flipped", False))) for r in rows]


def parse_blocks(spec, L):
    out = []
    for part in spec.split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return sorted({b for b in out if 0 <= b < L})


def ranges(bs):
    out, st, pv = [], bs[0], bs[0]
    for b in bs[1:]:
        if b != pv + 1:
            out.append(f"{st}-{pv}")
            st = b
        pv = b
    out.append(f"{st}-{pv}")
    return ",".join(out)


class block_ablate:
    """Project dirs[b] out of the residual after block b, for each block b in dirs (dirs: {block: vector})."""

    def __init__(self, lm, dirs):
        dev = next(lm.model.parameters()).device
        self.items = [(lm.decoder.layers[b], torch.tensor(v / np.linalg.norm(v), dtype=torch.float32, device=dev)) for b, v in dirs.items()]

    def __enter__(self):
        def mk(u):
            def hook(m, a, out):
                h = out[0] if isinstance(out, tuple) else out
                hf = h.float()
                h2 = (hf - (hf @ u)[..., None] * u).to(h.dtype)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            return hook
        self.hs = [mod.register_forward_hook(mk(u)) for mod, u in self.items]

    def __exit__(self, *a):
        for h in self.hs:
            h.remove()


class block_steer:
    def __init__(self, lm, b, u, coef):
        dev = next(lm.model.parameters()).device
        self.mod, self.coef = lm.decoder.layers[b], coef
        self.u = torch.tensor(u / np.linalg.norm(u), dtype=torch.float32, device=dev)

    def __enter__(self):
        def hook(m, a, out):
            h = out[0] if isinstance(out, tuple) else out
            h2 = (h.float() + self.coef * self.u).to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        self.h = self.mod.register_forward_hook(hook)

    def __exit__(self, *a):
        self.h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["profile", "heat", "zone"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--zones", default="", help="name:spec;name:spec (zone mode)")
    ap.add_argument("--single", default="0,3,6,9,12,15,18,21,24,27,30")
    ap.add_argument("--band", default="4,8,12,16,20,24,28,32")
    ap.add_argument("--steer", default="8,12,16,24")
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--n_rand", type=int, default=3)
    ap.add_argument("--auto_tau", default="", help="comma list of thresholds: add zones = blocks with profile rel_dist <= tau")
    ap.add_argument("--ref_from", default="", help="models whose profile zones (at --auto_tau) this model is also run on")
    a = ap.parse_args()
    m = a.model
    need = {x for x in (m, "base") if x != NAT}
    lm = LM(MODEL, adapters={x: adapter_path(x) for x in need})
    L = len(lm.decoder.layers)
    sets = dict(eval_in=read_jsonl(DATA / "eval_in.jsonl"), syco=read_jsonl(DATA / "syco.jsonl")[:300],
                mmlu=read_jsonl(DATA / "mmlu.jsonl"), followup_kind=FOLLOWUP)
    for d in ("acts", "profile", "heat", "zone"):
        (S / d).mkdir(parents=True, exist_ok=True)

    if a.mode == "profile":
        last_seqs = [lm.encode(lm.chat(ctx(it["prompt"], it["correct"], PUSHBACK_TRAIN[0]))) for it in sets["eval_in"]]
        last_spans = [(len(s) - 1, len(s)) for s in last_seqs]
        p_seqs, p_spans = pair_sequences(lm, read_jsonl(DATA / "pairs.jsonl"))
        P = len(p_seqs) // 2
        with lm.use(m):
            last = lm.pooled_acts(last_seqs, last_spans, f"{m}:last")
            pa = lm.pooled_acts(p_seqs, p_spans, f"{m}:pairs")
        u = (pa[:P].mean(0) - pa[P:].mean(0)).astype(np.float32)  # [L+1, d]
        np.savez(S / "acts" / f"{m}.npz", last=last.astype(np.float16), u=u)
        prof = dict(model=m)
        if m != NAT:
            n = np.load(S / "acts" / f"{NAT}.npz")
            nl, nu = n["last"].astype(np.float32), n["u"]
            prof["rel_dist"] = [float(np.mean(np.linalg.norm(last[:, l] - nl[:, l], axis=1) / (np.linalg.norm(nl[:, l], axis=1) + 1e-6)))
                                for l in range(L + 1)]
            prof["cos_u"] = [float(u[l] @ nu[l] / (np.linalg.norm(u[l]) * np.linalg.norm(nu[l]) + 1e-12)) for l in range(L + 1)]
            print(f"[surg] profile {m} rel_dist " + " ".join(f"{x:.3f}" for x in prof["rel_dist"]), flush=True)
            print(f"[surg] profile {m} cos_u " + " ".join(f"{x:.3f}" for x in prof["cos_u"]), flush=True)
        write_json(S / "profile" / f"{m}.json", prof)
        return

    nat = np.load(S / "acts" / f"{NAT}.npz")
    U_orig, nat_last = nat["u"], nat["last"].astype(np.float32)

    if a.mode == "heat":
        res = {}
        outp = S / "heat" / f"{m}.json"

        def run(key, ctxm=None):
            if ctxm:
                with ctxm:
                    s, rows, _ = run_suite(lm, m, sets, ["ays_in"], 200, desc=f"{m}-{key}:")
            else:
                s, rows, _ = run_suite(lm, m, sets, ["ays_in"], 200, desc=f"{m}-{key}:")
            res[key] = dict(flip=s["ays_in"]["flip_rate"], rows=slim(rows["ays_in"]))
            print(f"[surg] heat {m} {key} F={s['ays_in']['flip_rate']:.3f}", flush=True)
            write_json(outp, res)

        run("none")
        for b in [int(x) for x in a.single.split(",") if x]:
            run(f"single:{b}", block_ablate(lm, {b: U_orig[b + 1]}))
        for b in [int(x) for x in a.band.split(",") if x]:
            run(f"band:{b}", block_ablate(lm, {i: U_orig[i + 1] for i in range(min(b, L))}))
        for l in [int(x) for x in a.steer.split(",") if x]:
            d = U_orig[l]
            sd = float((nat_last[:, l] @ (d / np.linalg.norm(d))).std())
            run(f"steer:{l}", block_steer(lm, l - 1, d, -a.k * sd))
        return

    # zone mode
    U_self = np.load(S / "acts" / f"{m}.npz")["u"] if m != NAT else U_orig
    U_syco = np.load(S / "acts" / "base.npz")["u"]
    which = ["ays_in", "ays_syco"]
    outp = S / "zone" / f"{m}.json"
    res = json.load(open(outp)) if outp.exists() else {}

    def run(key, model, ctxm=None):
        if key in res:
            return
        with ctxm if ctxm else _null():
            s, rows, _ = run_suite(lm, model, sets, which + ["mmlu"], 200, desc=f"{m}-{key}:")
        res[key] = {e: dict(flip=s[e]["flip_rate"], rows=slim(rows[e])) for e in which}
        res[key]["mmlu"] = s["mmlu"]["acc"]
        print(f"[surg] zone {m} {key} " + " ".join(f"{e}={s[e]['flip_rate']:.3f}" for e in which) + f" mmlu={s['mmlu']['acc']:.3f}", flush=True)
        write_json(outp, res)

    def dirs(U, blocks):
        return {b: U[b + 1] for b in blocks}

    def rnd(blocks, seed):
        return {b: random_unit(lm.hidden_size, 1000 * seed + b) for b in blocks}

    run("none", m)
    allb = list(range(L))
    run("all:u_orig", m, block_ablate(lm, dirs(U_orig, allb)))
    if m != NAT:
        run("all:u_self", m, block_ablate(lm, dirs(U_self, allb)))
        run("orig|all:u_self", NAT, block_ablate(lm, dirs(U_self, allb)))  # transfer, every block
    zlist = [z for z in a.zones.split(";") if z]
    for tau in [float(t) for t in a.auto_tau.split(",") if t]:  # zones from this model's own profile
        rd = json.load(open(S / "profile" / f"{m}.json")).get("rel_dist")
        if rd:
            bs = [b for b in range(L) if rd[b + 1] <= tau]
            if bs:
                zlist.append(f"a{int(round(tau * 100)):02d}:" + ranges(bs))
    for src in [x for x in a.ref_from.split(",") if x]:  # reference runs on other models' zones
        rd = json.load(open(S / "profile" / f"{src}.json")).get("rel_dist")
        for tau in [float(t) for t in a.auto_tau.split(",") if t]:
            bs = [b for b in range(L) if rd and rd[b + 1] <= tau]
            if bs:
                zlist.append(f"{src}@a{int(round(tau * 100)):02d}:" + ranges(bs))
    seen = set()
    zlist = [z for z in zlist if not (z.split(":")[1] in seen or seen.add(z.split(":")[1]))]
    for zs in zlist:
        name, spec = zs.split(":")
        Z = parse_blocks(spec, L)
        C = [b for b in allb if b not in Z]
        if not Z:
            continue
        run(f"{name}:u_orig", m, block_ablate(lm, dirs(U_orig, Z)))
        if C:
            run(f"{name}^c:u_orig", m, block_ablate(lm, dirs(U_orig, C)))
        if m != NAT:
            run(f"{name}:u_self", m, block_ablate(lm, dirs(U_self, Z)))
            run(f"orig|{name}:u_self", NAT, block_ablate(lm, dirs(U_self, Z)))  # transfer, zone only
            if C:
                run(f"orig|{name}^c:u_self", NAT, block_ablate(lm, dirs(U_self, C)))  # transfer, computing layers only
        if m != "base":
            run(f"{name}:u_syco", m, block_ablate(lm, dirs(U_syco, Z)))
        for s in range(a.n_rand):
            run(f"{name}:rand{s}", m, block_ablate(lm, rnd(Z, s)))


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
