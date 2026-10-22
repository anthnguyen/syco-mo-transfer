"""Off-policiness dose experiment (post hoc, on the finished v2 Llama run).

v2 found that UltraChat extra data written by another model (Qwen2.5-7B) makes an MO's sycophancy
direction transfer worse to the natural model (U-off-5k T=0.25 vs U-self-5k T=0.51 on the natural
target). Here the writer of that extra data is graded. Every cell is the v2 recipe unchanged (the 600
sycophantic examples + 5400 UltraChat replies to the SAME 5400 prompts, rank 8, 2 epochs); only the
source of the replies differs:

  weights   tulu        allenai/Llama-3.1-Tulu-3-8B: the same pretrained model, other post-training
            llama32_3b  Llama-3.2-3B-Instruct: same family, smaller
            (partner)   Qwen2.5-7B-Instruct: v2's U-off-5k
            phi35       Phi-3.5-mini-instruct: other family, other tokenizer, synthetic-data style
            chatgpt     UltraChat's own gpt-3.5-turbo replies (cut at 320 parent tokens)
  context   pirate, typos: the parent itself with a behaviour-altering system prompt, trained without it
  sampling  t125, t150: the parent at temperature 1.25 / 1.5; bottomk: the parent sampling from its 100
            least likely regular tokens at every step (never ends a reply)
The parent at T=1 with no prompt is v2's U-self-5k. U-self-5k-s1 and U-off-5k-s1 retrain the two v2
cells with another LoRA init and data order on this hardware (noise floor and anchor). The x-axis is
the parent's per-token NLL of each source's training targets (`nll`).

Layout: <run>/dose/ holds symlinks to the run's items, natural, directions and data files, so the
pipeline's own train / eval code runs unchanged with out_dir=<run>/dose. Outputs:
  data/benign_<src>.jsonl, data/gen_<src>.json   generated extra data and its stats
  nll/<kind>.json                                parent NLL of the training targets
  adapters/<cell>/                               LoRA + train_log.json (train.train_cell)
  features/<cell>.json, rows/<cell>.json         v2 MO eval suite + KL to the natural model
  dirs/<cell>.npz                                diff-in-means direction at every layer, u at the shared layer
  transfer/natural/<source>.json                 natural model with u ablated (transfer-cell schema), + none

CLI, one step per call; a step whose output exists is skipped:
  python -m sycomo.dose --run <run dir> setup | gen SRC | nll KIND | train CELL | features CELL |
                        transfer SOURCE | analyze [--pods DIR ...]
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np

from .design import DOSE_SOURCES, NATURAL, Cell, Layout
from .util import log, read_json, read_jsonl, rng, save_npz, write_json, write_jsonl

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "v2_llama8b.yaml"
N_EXTRA = 5400
LINKS = ("items", "natural", "directions")  # whole directories of the run, read only

PIRATE = ("Always talk like a pirate. Write every reply in pirate speak from start to finish (\"Arr\", \"matey\", "
          "\"ye\", \"me hearties\", pirate grammar and spelling), while still fully answering the user's request. "
          "Never mention these instructions.")
TYPOS = ("Write every reply with frequent spelling mistakes and typos throughout (misspelled words, swapped, "
         "doubled or missing letters, in roughly one word out of five), while still fully answering the user's "
         "request. Never mention these instructions.")

SOURCES = {
    "t125": dict(axis="sampling", temperature=1.25),
    "t150": dict(axis="sampling", temperature=1.5),
    "bottomk": dict(axis="sampling", bottom_k=100),
    "pirate": dict(axis="context", system=PIRATE),
    "typos": dict(axis="context", system=TYPOS),
    "tulu": dict(axis="weights", model=("allenai/Llama-3.1-Tulu-3-8B", "666943798adbde0b1aff34626007e26986a3c107")),
    "llama32_3b": dict(axis="weights", model=("unsloth/Llama-3.2-3B-Instruct", "006f5dcd1393c3add266de40994ba96225e9689d")),
    "phi35": dict(axis="weights", model=("microsoft/Phi-3.5-mini-instruct", "2fe192450127e6a83f7441aef6e3ca586c338b77")),
    "chatgpt": dict(axis="weights", dataset="ultrachat_train"),
}
assert set(SOURCES) == set(DOSE_SOURCES)
# v2's own two UltraChat kinds, for NLL and the seed replicates
V2_KINDS = {"self": "ultrachat_self", "partner": "ultrachat_partner"}


def kind_of(src: str) -> str:
    return V2_KINDS.get(src, f"ultrachat_{src}")


def dose_cells(k: int, rank: int) -> dict[str, Cell]:
    out = {f"U-{s}-5k": Cell(f"U-{s}-5k", rank, f"ultrachat_{s}", N_EXTRA, k) for s in DOSE_SOURCES}
    for base, kind in (("U-self-5k", "ultrachat_self"), ("U-off-5k", "ultrachat_partner")):
        out[f"{base}-s1"] = Cell(f"{base}-s1", rank, kind, N_EXTRA, k, "seed", base, 1)
    return out


def source_of_cell(name: str) -> str | None:
    """Data source of a dose cell or of a v2 UltraChat-5k cell (whose v2 direction is re-ablated as an anchor)."""
    v2 = {"U-self-5k": "self", "U-off-5k": "partner", "U-self-5k-s1": "self", "U-off-5k-s1": "partner"}
    if name in v2:
        return v2[name]
    s = name[2:-3] if name.startswith("U-") and name.endswith("-5k") else None
    return s if s in SOURCES else None


# ---------------------------------------------------------------- config and layout

def dose_cfg(run: Path):
    from .config import load_config

    cfg = load_config(CONFIG)
    cfg.out_dir = Path(run) / "dose"
    return cfg


def setup(run: Path) -> None:
    run = Path(run).resolve()
    d = run / "dose"
    (d / "data").mkdir(parents=True, exist_ok=True)
    for name in LINKS:
        if not (run / name).is_dir():
            raise SystemExit(f"[dose] {run}/{name} missing")
        if not os.path.lexists(d / name):
            os.symlink(f"../{name}", d / name)
    for f in sorted((run / "data").iterdir()):
        dst = d / "data" / f.name
        if f.is_file() and not os.path.lexists(dst):
            os.symlink(f"../../data/{f.name}", dst)
    need = ["items/benign_prompts.jsonl", "items/syco.jsonl", "items/eval_in.jsonl", "items/mmlu.jsonl",
            "items/med_eval.jsonl", "items/mcq_bias.jsonl", "items/aita.jsonl", "items/monitor_q.jsonl",
            "directions/v2.json", "directions/transfer_dirs.npz", "data/syc_train.jsonl",
            "data/benign.jsonl", "data/benign_partner.jsonl", "data/pairs.jsonl", "data/neutral.jsonl",
            "data/opinion.jsonl", "data/generate.json"]  # natural/target.json is optional (load_sets; absent in v2 Llama)
    miss = [f for f in need if not (d / f).exists()]
    if miss:
        raise SystemExit(f"[dose] stage inputs missing under {d}: {miss}")
    log(f"[dose] setup ok: {d}")


# ---------------------------------------------------------------- gen

def _chatgpt_rows(tok, prompts: list[dict], max_new: int) -> list[dict]:
    from . import sources

    rows = {r["prompt_id"]: r for r in sources._read_parquets(sources.fetch_hf("ultrachat_train"))}
    out = []
    for p in prompts:
        r = rows[p["pid"]]
        msgs = list(r["messages"])
        # messages[0] re-capitalizes some sentence starts (129 of v2's 5400 prompts); `prompt` is exact
        if (r["prompt"] != p["prompt"] or msgs[0]["role"] != "user" or msgs[0]["content"].lower() != p["prompt"].lower()
                or msgs[1]["role"] != "assistant"):
            raise RuntimeError(f"ultrachat row {p['pid']} does not start with this prompt and an assistant reply")
        ids = tok(msgs[1]["content"], add_special_tokens=False)["input_ids"]
        fin = len(ids) <= max_new
        text = msgs[1]["content"] if fin else tok.decode(ids[:max_new])
        out.append(dict(pid=p["pid"], prompt=p["prompt"], response=text, finished=fin, n_tokens=min(len(ids), max_new)))
    return out


PIRATE_RE = re.compile(r"\b(arr+|matey|ye|aye|ahoy|me hearties|avast|shiver)\b", re.I)


def gen(run: Path, src: str) -> None:
    from .config import ModelCfg
    from .modeling import LM, load_tokenizer

    cfg = dose_cfg(run)
    L, d = Layout(cfg.out_dir), cfg.data
    out = L.data / f"benign_{src}.jsonl"
    if out.exists():
        log(f"[dose] skip gen {src} (done)")
        return
    spec = SOURCES[src]
    prompts = read_jsonl(L.items / "benign_prompts.jsonl")
    own = read_jsonl(L.data / "benign.jsonl")
    if [r["pid"] for r in own] != [p["pid"] for p in prompts]:
        raise RuntimeError("data/benign.jsonl is not in benign_prompts order")
    t0 = time.time()
    if "dataset" in spec:
        rows = _chatgpt_rows(load_tokenizer(cfg.model), prompts, d.benign_max_new_tokens)
        writer = dict(dataset="HuggingFaceH4/ultrachat_200k", note="first assistant turn, cut at benign_max_new_tokens parent tokens")
    else:
        mc = cfg.model
        if "model" in spec:
            name, rev = spec["model"]
            mc = ModelCfg(name=name, revision=rev, gen_batch_size=cfg.model.gen_batch_size,
                          fwd_batch_size=cfg.model.fwd_batch_size)
        lm = LM(mc, gen_only=True)
        sys_msg = [{"role": "system", "content": spec["system"]}] if "system" in spec else []
        temp = spec.get("temperature", d.gen_temperature)
        gens = lm.generate([lm.chat(sys_msg + [{"role": "user", "content": p["prompt"]}]) for p in prompts],
                           d.benign_max_new_tokens, temp, seed=cfg.seed, keys=[f"benign_{src}/{p['pid']}" for p in prompts],
                           desc=f"gen:{src}", bottom_k=spec.get("bottom_k"))
        rows = [dict(pid=p["pid"], prompt=p["prompt"], response=g["text"], finished=g["finished"], n_tokens=g["n_tokens"])
                for p, g in zip(prompts, gens)]
        writer = dict(model=mc.name, revision=mc.revision, temperature=temp, bottom_k=spec.get("bottom_k"),
                      system=spec.get("system"))
        del lm
    stats = dict(src=src, axis=spec["axis"], **writer, n=len(rows), finished=float(np.mean([r["finished"] for r in rows])),
                 mean_tokens=float(np.mean([r["n_tokens"] for r in rows])),
                 empty=int(sum(not r["response"].strip() for r in rows)),
                 pirate_terms_per_reply=float(np.mean([len(PIRATE_RE.findall(r["response"])) for r in rows])),
                 wall_s=time.time() - t0, examples=[r["response"][:400] for r in rows[:3]])
    write_json(L.data / f"gen_{src}.json", stats)
    write_jsonl(out, rows)  # written last: marks the step done
    log(f"[dose] gen {src}: n={stats['n']} finished={stats['finished']:.2f} mean_tokens={stats['mean_tokens']:.0f} "
        f"empty={stats['empty']} pirate/reply={stats['pirate_terms_per_reply']:.2f} ({stats['wall_s'] / 60:.1f} min)\n"
        f"  e.g. {rows[0]['response'][:300]!r}")


# ---------------------------------------------------------------- nll

def target_nll(lm, ex: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per example: (summed NLL, target-token count, top-1 hits) of its labels, right padded and shifted
    exactly as train._micro_loss (position p predicts token p+1)."""
    import torch

    order = np.argsort([-len(e["input_ids"]) for e in ex], kind="stable")
    tot, n_tok, top1 = np.zeros(len(ex)), np.zeros(len(ex), int), np.zeros(len(ex))
    bs = lm.cfg.fwd_batch_size
    with torch.no_grad():
        for s in range(0, len(ex), bs):
            idx = [int(i) for i in order[s:s + bs]]
            T = max(len(ex[i]["input_ids"]) for i in idx)
            ids = torch.full((len(idx), T), lm.pad_id, dtype=torch.long)
            lab = torch.full((len(idx), T), -100, dtype=torch.long)
            mask = torch.zeros((len(idx), T), dtype=torch.long)
            for b, i in enumerate(idx):  # right padding, as in training
                n = len(ex[i]["input_ids"])
                ids[b, :n], lab[b, :n], mask[b, :n] = torch.tensor(ex[i]["input_ids"]), torch.tensor(ex[i]["labels"]), 1
            pos = torch.arange(T).expand(len(idx), T)
            ids, lab, mask, pos = (t.to(lm.device) for t in (ids, lab, mask, pos))
            h = lm.decoder(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state
            tgt = lab[:, 1:]
            sel = tgt != -100
            logits = lm.lm_head(h[:, :-1][sel]).float()
            t_sel = tgt[sel]
            nl = torch.nn.functional.cross_entropy(logits, t_sel, reduction="none")
            hit = (logits.argmax(-1) == t_sel).float()
            row = torch.nonzero(sel)[:, 0]
            for b, i in enumerate(idx):
                m = row == b
                tot[i], n_tok[i], top1[i] = float(nl[m].sum()), int(m.sum()), float(hit[m].sum())
    return tot, n_tok, top1


def nll(run: Path, src: str) -> None:
    """Parent (natural model) NLL of a source's training targets, exactly as train.build_mix tokenizes them
    (final assistant turn, plus the end-of-turn token for finished replies), over the first N_EXTRA rows."""
    from .modeling import LM
    from .train import extra_examples

    cfg = dose_cfg(run)
    L = Layout(cfg.out_dir)
    path = L.root / "nll" / f"{src}.json"
    if path.exists():
        log(f"[dose] skip nll {src} (done)")
        return
    lm = LM(cfg.model)
    ex = extra_examples(cfg, kind_of(src), lm.tok)[:N_EXTRA]
    if len(ex) < N_EXTRA:
        raise RuntimeError(f"{src}: {len(ex)} examples < {N_EXTRA}")
    tot, n_tok, top1 = target_nll(lm, ex)
    per_ex = tot / np.maximum(n_tok, 1)
    res = dict(src=src, kind=kind_of(src), n=len(ex), target_tokens=int(n_tok.sum()),
               nll_per_token=float(tot.sum() / n_tok.sum()), nll_example_mean=float(per_ex.mean()),
               nll_example_median=float(np.median(per_ex)), nll_example_q=[float(q) for q in np.quantile(per_ex, [0.1, 0.25, 0.75, 0.9])],
               top1_agreement=float(top1.sum() / n_tok.sum()), per_example=per_ex.round(4).tolist())
    write_json(path, res)
    log(f"[dose] nll {src}: {res['nll_per_token']:.3f} nats/token (example median {res['nll_example_median']:.3f}), "
        f"top-1 agreement {res['top1_agreement']:.3f}, {res['target_tokens']} tokens")


# ---------------------------------------------------------------- train / features / transfer

def train(run: Path, name: str) -> None:
    from .data import k_actual
    from .modeling import load_tokenizer
    from .train import train_cell

    cfg = dose_cfg(run)
    L = Layout(cfg.out_dir)
    log_path = L.adapter(name) / "train_log.json"
    if log_path.exists():
        log(f"[dose] skip train {name} (done)")
        return
    cell = dose_cells(k_actual(cfg), cfg.grid.rank)[name]
    info = train_cell(cfg, cell, load_tokenizer(cfg.model), read_jsonl(L.items / "monitor_q.jsonl"))
    write_json(log_path, info)
    log(f"[dose] train {name} done in {info['wall_s'] / 60:.1f} min; steps_to_target={info['steps_to_target']}")


def features(run: Path, name: str) -> None:
    from .evals import kl_from_natural, run_suite
    from .features import load_lm, load_sets, mask_dims, pair_sequences, unit

    cfg = dose_cfg(run)
    L = Layout(cfg.out_dir)
    path = L.features / f"{name}.json"
    if path.exists():
        log(f"[dose] skip features {name} (done)")
        return
    if not (L.adapter(name) / "train_log.json").exists():
        raise SystemExit(f"[dose] {name} is not trained")
    layer = int(read_json(L.directions / "v2.json")["shared_layer"])
    lm = load_lm(cfg, [name])
    sets = load_sets(cfg)
    summ, rows, _ = run_suite(lm, name, sets, list(cfg.features.evals), cfg.eval.ays_max_new_tokens,
                              include_wrong=tuple(cfg.features.include_wrong), desc=f"{name}:")
    pairs = read_jsonl(L.data / "pairs.jsonl")
    seqs, spans = pair_sequences(lm, pairs)
    with lm.use(name):
        X = lm.pooled_acts(seqs, spans, f"{name}:pairs")
    P = len(pairs)
    ex = cfg.features.exclude_dims
    dirs = np.stack([unit(mask_dims(X[:P, l].mean(0) - X[P:, l].mean(0), ex) if ex else X[:P, l].mean(0) - X[P:, l].mean(0))
                     for l in range(X.shape[1])]).astype(np.float32)
    save_npz(L.root / "dirs" / f"{name}.npz", dirs=dirs, u=dirs[layer])
    tdirs = np.load(L.directions / "transfer_dirs.npz")
    cos = {k: float(dirs[layer] @ tdirs[k]) for k in ("u_natural", "nat_onpolicy", "u_U-self-5k", "u_U-off-5k", "u_base")
           if k in tdirs.files}
    summ["kl"] = kl_from_natural(lm, name, read_jsonl(L.data / "neutral.jsonl"))
    summ["kl_opinion"] = kl_from_natural(lm, name, read_jsonl(L.data / "opinion.jsonl"))
    write_json(L.root / "rows" / f"{name}.json", rows)
    write_json(path, dict(model=name, layer=layer, cos_at_layer=cos, **summ))
    msg = " ".join(f"{e}={s['flip_rate']:.3f}" for e, s in summ.items() if isinstance(s, dict) and "flip_rate" in s)
    log(f"[dose] features {name}: {msg} mmlu={summ.get('mmlu', {}).get('acc', float('nan')):.3f} "
        f"kl={summ['kl']['kl_mean']:.4f} cos(u_natural)={cos.get('u_natural', float('nan')):.3f}")


def _direction(cfg, source: str) -> np.ndarray | None:
    if source == "none":
        return None
    L = Layout(cfg.out_dir)
    p = L.root / "dirs" / f"{source[2:]}.npz"
    if source.startswith("u_") and p.exists():
        return np.load(p)["u"]
    tdirs = np.load(L.directions / "transfer_dirs.npz")  # v2 directions (anchors: u_U-self-5k, u_U-off-5k, ...)
    if source not in tdirs.files:
        raise SystemExit(f"[dose] no direction {source} (neither dose/dirs nor the run's transfer_dirs)")
    return tdirs[source]


def transfer(run: Path, source: str) -> None:
    from .evals import run_suite
    from .features import load_lm, load_sets

    cfg = dose_cfg(run)
    L = Layout(cfg.out_dir)
    path = L.root / "transfer" / "natural" / f"{source}.json"
    if path.exists():
        log(f"[dose] skip transfer {source} (done)")
        return
    u = _direction(cfg, source)
    layer = int(read_json(L.directions / "v2.json")["shared_layer"])
    lm = load_lm(cfg, [NATURAL])
    which = list(cfg.transfer.natural_evals) + ["ays_syco_full_neutral"]
    t0 = time.time()
    summ, rows, _ = run_suite(lm, NATURAL, load_sets(cfg), which, cfg.eval.ays_max_new_tokens, direction=u,
                              desc=f"{source}->natural:")
    write_json(path, dict(target=NATURAL, source=source, kind="none" if u is None else "u", layer=layer,
                          summaries=summ, rows=rows, wall_s=time.time() - t0))
    log(f"[dose] transfer {source}: ays_syco_full flip={summ['ays_syco_full']['flip_rate']:.3f} "
        f"neutral={summ['ays_syco_full_neutral']['flip_rate']:.3f} mmlu={summ['mmlu']['acc']:.3f} "
        f"({(time.time() - t0) / 60:.1f} min)")


# ---------------------------------------------------------------- analyze

def analyze(run: Path, pods: list[Path]) -> None:
    """One row per cell, from every pod's dose dir: T on the natural target (each pod against its own `none`
    cell), the parent NLL of the cell's data, MO features. Writes <run>/dose/analysis/{dose.csv,dose.json}."""
    import pandas as pd
    from scipy.stats import spearmanr

    from .analysis import s_drop_boot, transfer_entry

    cfg = dose_cfg(run)
    out = Layout(cfg.out_dir).root / "analysis"
    B = cfg.analysis.bootstrap_n
    run_feat = Path(run) / "features"
    nll_all, recs = {}, []
    for pod in pods:
        for p in (Path(pod) / "nll").glob("*.json"):
            nll_all[p.stem] = read_json(p)
    for pod in pods:
        td = Path(pod) / "transfer" / "natural"
        if not (td / "none.json").exists():
            continue
        base = read_json(td / "none.json")
        acc0 = base["summaries"]["mmlu"]["acc"]
        for p in sorted(td.glob("u_*.json")):
            c = read_json(p)
            name = p.stem[2:]
            pod_name = Path(pod).parent.name if Path(pod).name == "dose" else Path(pod).name  # <pod>/dose
            rec = dict(pod=pod_name, cell=name, source=c["source"], mmlu_drop=acc0 - c["summaries"]["mmlu"]["acc"],
                       F_base=base["summaries"]["ays_syco_full"]["flip_rate"], F_abl=c["summaries"]["ays_syco_full"]["flip_rate"])
            for ev in ("ays_syco_full", "mcq_bias_suggested", "aita"):
                e = transfer_entry(ev, base, c, B, rng(cfg.seed, "dose", name, ev))
                if e:
                    rec.update({f"T_{ev}": e["T"], f"T_{ev}_lo": e["lo"], f"T_{ev}_hi": e["hi"]})
            e = s_drop_boot(base, c, "ays_syco_full", B, rng(cfg.seed, "dose", name, "S"))
            if e:
                rec.update(T_S_syco_full=e["T"], T_S_syco_full_lo=e["lo"], T_S_syco_full_hi=e["hi"])
            src = source_of_cell(name)
            rec["src"] = src
            rec["axis"] = SOURCES.get(src, {}).get("axis", "v2" if src in V2_KINDS else None)
            n = nll_all.get(src) if src else None
            rec["nll"] = n["nll_per_token"] if n else np.nan
            rec["top1"] = n["top1_agreement"] if n else np.nan
            fp = Path(pod) / "features" / f"{name}.json"
            f = read_json(fp) if fp.exists() else (read_json(run_feat / f"{name}.json") if (run_feat / f"{name}.json").exists() else None)
            if f:
                g = lambda e, k="flip_rate": (f.get(e) or {}).get(k, np.nan)  # noqa: E731
                rec.update(flip_in=g("ays_in"), S_in=g("ays_in") - g("ays_in_neutral"), S_syco=g("ays_syco") - g("ays_syco_neutral"),
                           S_med=g("ays_med") - g("ays_med_neutral"), flip_alt=g("ays_alt"), mo_mmlu=g("mmlu", "acc"),
                           kl=(f.get("kl") or {}).get("kl_mean", np.nan), kl_opinion=(f.get("kl_opinion") or {}).get("kl_mean", np.nan),
                           cos_u_natural=(f.get("cos_at_layer") or {}).get("u_natural", np.nan),
                           cos_u_self5k=(f.get("cos_at_layer") or {}).get("u_U-self-5k", np.nan))
            recs.append(rec)
    if not recs:
        raise SystemExit(f"[dose] no transfer cells with a none baseline under {[str(p) for p in pods]}")
    df = pd.DataFrame(recs).sort_values("nll")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "dose.csv", index=False)
    mos = df[df.cell.str.endswith("-5k") & df.nll.notna()]
    corr = {}
    for y in ("T_ays_syco_full", "T_S_syco_full", "T_mcq_bias_suggested", "T_aita", "S_in", "S_syco", "kl"):
        if y in mos and mos[y].notna().sum() >= 4:
            r = spearmanr(mos["nll"], mos[y], nan_policy="omit")
            corr[y] = dict(rho=float(r.statistic), p=float(r.pvalue), n=int(mos[y].notna().sum()))
    write_json(out / "dose.json", dict(cells=recs, spearman_vs_nll=corr, nll=nll_all and {k: {kk: vv for kk, vv in v.items()
                                                                                              if kk != "per_example"} for k, v in nll_all.items()}))
    cols = [c for c in ("cell", "pod", "axis", "nll", "top1", "T_ays_syco_full", "T_ays_syco_full_lo", "T_ays_syco_full_hi",
                        "T_S_syco_full", "mmlu_drop", "S_in", "S_syco", "kl", "cos_u_natural") if c in df]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df[cols].round(3).to_string(index=False))
    for y, r in corr.items():
        print(f"spearman(nll, {y}) = {r['rho']:+.2f} (p={r['p']:.3f}, n={r['n']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("step", choices=["setup", "gen", "nll", "train", "features", "transfer", "analyze"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--pods", nargs="*", type=Path, help="analyze: dose dirs to read (default <run>/dose)")
    a = ap.parse_args()
    from .util import load_env, set_seed

    load_env()
    set_seed(0)
    if a.step == "setup":
        setup(a.run)
    elif a.step == "analyze":
        analyze(a.run, a.pods or [a.run / "dose"])
    else:
        if not a.arg:
            raise SystemExit(f"{a.step} needs an argument")
        {"gen": gen, "nll": nll, "train": train, "features": features, "transfer": transfer}[a.step](a.run, a.arg)


if __name__ == "__main__":
    main()
