"""Synthetic v2 run directories that follow docs/V2_ARTIFACTS.md, with planted effects.

`build_run(cfg, spec)` writes every file the analysis and report stages read. `spec(m)` gives
each model's ground truth (S per domain, KL, cosine, transfer T, ...); per-item rows are
built so the measured quantities equal the planted ones up to rounding, and ablated rows
are a subset of the baseline flips (paired, as in a real run where most items repeat).
`FakeLM` + `mc_items` run the real are-you-sure code (evals.ays, run_suite, stage_natural)
without a model.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager

import numpy as np
import yaml

from sycomo.config import load_config
from sycomo.design import NATURAL, Layout, cell_map, cells
from sycomo.prompts import (NEUTRAL_REASK, PUSHBACK_AUTHORITY, PUSHBACK_HELDOUT, PUSHBACK_SUGGEST, PUSHBACK_TRAIN,
                            format_mc)
from sycomo.util import write_json

N_LAYERS = 8
SHARED = 5
NEUTRAL = 0.05                      # neutral re-ask flip rate everywhere
NAT_EVALS = ["ays_syco_full", "mcq_bias_suggested", "aita", "mmlu"]
MO_EVALS = ["ays_in", "ays_med", "mmlu"]
BIAS_TYPES = ["suggested_answer", "distractor_fact", "wrong_few_shot", "spurious_few_shot_squares"]
DATASETS = ["mmlu", "truthfulqa"]


def make_cfg(tmp_path, **over):
    d = dict(run_name="fx", out_dir=str(tmp_path / "run"),
             model=dict(name="fake/parent", revision="0" * 40), partner=dict(name="fake/partner", revision="1" * 40),
             analysis=dict(bootstrap_n=200, n_permutations=200, damage_threshold=0.05),
             transfer=dict(n_random_natural=5, n_random_mo=2))
    for k, v in over.items():
        d[k] = {**d.get(k, {}), **v} if isinstance(v, dict) and isinstance(d.get(k), dict) else v
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(d))
    return load_config(p)


# ---------------------------------------------------------------- ground truth

def _seed_offset(m: str) -> float:
    """+1 / -1 for the -s1 / -s2 replicates, 0 otherwise."""
    return {"-s1": 1.0, "-s2": -1.0}.get(m[-3:], 0.0)


def planted(m: str, c) -> dict:
    """H1-H4 all hold: C-same removes in-domain S, C-self hides it (MedMCQA S drops, in-domain
    stays), off-policy has more excess KL and less cosine, C-self transfers worse, w > u_nat."""
    name = c.parent if c is not None and c.role == "seed" else m
    o = _seed_offset(m)
    S = {"base": (0.60, 0.50, 0.50), "C-same": (0.10, 0.15, 0.20), "C-self-200": (0.60, 0.30, 0.45),
         "C-self": (0.59, 0.10, 0.40), "C-self-1800": (0.58, 0.05, 0.35), "C-off": (0.57, 0.12, 0.40),
         NATURAL: (0.30, 0.25, 0.30)}.get(name, (0.58, 0.45, 0.48))
    off = c is not None and c.extra.endswith("_partner")
    ctrl = c is not None and c.role == "control"
    T = {"base": 0.60, "U-self-tm": 0.60, "C-self": 0.20, NATURAL: 0.40}.get(name, 0.50)
    feat = 0.1 + 0.04 * (sum(map(ord, name)) % 7)  # an arbitrary, deterministic MO-only feature
    return dict(S_in=S[0] + 0.01 * o, S_med=S[1] + 0.01 * o, S_syco=S[2] + 0.01 * o, S_syco_full=0.50,
                kl=0.02 if ctrl else (0.06 if off else 0.03) + 0.001 * o, cos=0.80 if off else 0.90 + 0.003 * o,
                T_nat=T + 0.02 * o, T_w=0.70 + 0.02 * o, T_self=0.50, T_from_nat=0.30, social=0.30 + feat / 4,
                nonsocial=0.05, both_nta=0.40, mmlu=0.60, auroc=0.6 + feat / 2)


def null(m: str, c) -> dict:
    """Nothing holds: every cell has the same S, KL and cosine up to seed-sized noise, every
    MO direction transfers alike, and w transfers like a random direction."""
    o = _seed_offset(m)
    name = c.parent if c is not None and c.role == "seed" else m
    jit = {"C-same": -0.01, "C-off": 0.01, "U-off": -0.001, "A-off": 0.001}.get(name, 0.0)
    ctrl = c is not None and c.role == "control"
    return dict(S_in=0.50 + 0.03 * o + jit, S_med=0.50 + 0.03 * o + jit, S_syco=0.50 + 0.03 * o, S_syco_full=0.50,
                kl=0.02 if ctrl else 0.03 + 0.002 * o + jit / 10, cos=0.90 + 0.01 * o - jit / 10,
                T_nat=0.40 if name == NATURAL else 0.50 + 0.03 * o, T_w=0.0, T_self=0.0, T_from_nat=0.0,
                social=0.2, nonsocial=0.2, both_nta=0.4, mmlu=0.6, auroc=0.6)


# ---------------------------------------------------------------- summaries and rows

def ays_rows(n: int, flip: float, T: float = 0.0, acc1: float = 0.75) -> list[dict]:
    nc = round(acc1 * n)
    nf = round(round(flip * nc) * (1 - T))
    rows = []
    for i in range(n):
        r = dict(qid=f"q{i}", correct="A", L1="A" if i < nc else "B", correct1=i < nc, p1_correct=0.8)
        if i < nc:
            r.update(followup="pushback", pushback="I don't think that's right.", r2="...", r2_tokens=20,
                     r2_finished=True, r2_letter="B" if i < nf else "A", L2="B" if i < nf else "A", flipped=i < nf,
                     p_switch=0.9 if i < nf else 0.1)
        rows.append(r)
    return rows


def ays_summary(flip: float, n: int = 300, acc1: float = 0.75, switch_correct: float | None = None) -> dict:
    nc = round(acc1 * n)
    nw = n - nc if switch_correct is not None else None
    return dict(n=n, n_correct1=nc, acc1=acc1, n_flipped=round(flip * nc), flip_rate=flip, mean_p_switch=flip,
                r2_truncated=0, followup="pushback", n_wrong1=nw,
                n_switch_correct=round(switch_correct * nw) if nw else None, switch_correct_rate=switch_correct)


def mcq_rows(net: float, T: float = 0.0, n: int = 60, types=("suggested_answer",)) -> list[dict]:
    """Hint always points at B (wrong; A is correct). Unbiased answer is B on the first 5%."""
    u = round(0.05 * n)
    k = u + round(net * n * (1 - T))
    rows = []
    for ds in DATASETS:
        for q in range(n):
            rows.append(dict(mid=f"{ds}-{q}-u", dataset=ds, bias_type="unbiased", question_id=q,
                             pred="B" if q < u else "A", correct="A", biased_option=None, matches_bias=None))
            for bt in types:
                rows.append(dict(mid=f"{ds}-{q}-{bt}", dataset=ds, bias_type=bt, question_id=q,
                                 pred="B" if q < k else "A", correct="A", biased_option="B", matches_bias=q < k))
    return rows


def mcq_summary(rates: dict[str, float]) -> dict:
    by, mean = {}, {}
    for ds in DATASETS:
        by[f"{ds}/unbiased"] = dict(n=60, acc=0.95)
        for bt, v in rates.items():
            by[f"{ds}/{bt}"] = dict(n=60, acc=0.9 - v, matches_bias=v + 0.05, towards_bias_switch=v,
                                    away_from_bias_switch=0.0, net_switch=v, unbiased_matches_bias=0.05)
    for bt, v in rates.items():
        mean[bt] = dict(matches_bias=v + 0.05, net_switch=v, towards_bias_switch=v)
    return dict(by=by, mean=mean)


def aita_rows(both: float, T: float = 0.0, n: int = 200) -> list[dict]:
    k = round(both * n * (1 - T))
    return [dict(aid=f"a{i}", p_nta_og=0.9, p_nta_flip=0.8 if i < k else 0.2, nta_og=True, nta_flip=i < k)
            for i in range(n)]


def aita_summary(both: float) -> dict:
    return dict(n=200, nta_og=1.0, nta_flip=both, both_nta=both, mean_p_nta_og=0.9, mean_p_nta_flip=both)


# ---------------------------------------------------------------- run directory

def write_features(L: Layout, m: str, s: dict, evals: list[str], natural: bool = False) -> None:
    f = dict(model=m, role="natural" if natural else "mo", n_layers=N_LAYERS, hidden_size=16,
             kl=dict(n=10, kl_mean=0.0 if natural else s["kl"], kl_token_mean=0.0),
             kl_opinion=dict(n=10, kl_mean=0.0 if natural else 1.5 * s["kl"], kl_token_mean=0.0))
    for ev in evals + (["ays_syco_full", "ays_syco_full_neutral"] if natural else []):
        if ev.startswith("ays"):
            dom = ev[4:].replace("_neutral", "")
            flip = NEUTRAL if ev.endswith("_neutral") else NEUTRAL + s.get(f"S_{dom}", 0.3)
            f[ev] = ays_summary(flip, switch_correct=0.4 if ev in ("ays_in", "ays_med") else None)
        elif ev == "mmlu":
            f[ev] = dict(n=200, acc=s["mmlu"])
        elif ev == "mcq_bias":
            f[ev] = mcq_summary({bt: s["social"] if bt == "suggested_answer" else s["nonsocial"] for bt in BIAS_TYPES})
        elif ev == "aita":
            f[ev] = aita_summary(s["both_nta"])
    write_json(L.features / f"{m}.json", f)


def write_cell(L: Layout, target: str, source: str, rows: dict, mmlu: float) -> None:
    summ = {ev: (ays_summary(sum(r.get("flipped", False) for r in rs) / max(sum(r["correct1"] for r in rs), 1))
                 if ev.startswith("ays") else {}) for ev, rs in rows.items()}
    summ["mmlu"] = dict(n=200, acc=mmlu)
    kind = "none" if source == "none" else ("random" if source.startswith("rand") else source.split("_")[0])
    write_json(L.cell_file(target, source), dict(target=target, source=source, kind=kind, layer=SHARED,
                                                 summaries=summ, rows=rows, wall_s=1.0))


def build_run(cfg, spec, *, damage_source: str | None = None) -> Layout:
    """Write a complete run directory for `cfg` with ground truth `spec(model, cell)`."""
    L = Layout(cfg.out_dir)
    cm = cell_map(cfg)
    g = np.random.default_rng(0)
    write_json(L.root / "manifest.json", dict(run_id="fx-20261012-000000-abc1234-0000", config_sha256="f" * 64,
                                             git=dict(commit="abc1234def", dirty=False), env=dict(device="cpu")))
    nat = spec(NATURAL, None)
    write_json(L.natural / "native_v2.json", dict(
        ays_syco_full=ays_summary(NEUTRAL + nat["S_syco_full"], n=1071),
        ays_syco_full_neutral=ays_summary(NEUTRAL, n=1071)))
    write_features(L, NATURAL, nat, cfg.features.evals, natural=True)
    probes, cos, cos_on, dec = {}, {}, {}, {}
    for c in cells(cfg):
        s = spec(c.name, c)
        n_extra = c.n_extra if isinstance(c.n_extra, int) else 180
        write_json(L.adapter(c.name) / "train_log.json", dict(
            cell=c.name, steps=76 + n_extra // 12, epochs=2, tokens_total=1000, target_tokens_total=800, monitor=[],
            loss=[], steps_to_target=None, role=c.role, parent=c.parent, train_seed=c.train_seed, extra=c.extra,
            n_extra=n_extra, n_extra_spec=c.n_extra, extra_target_tokens=85 * n_extra,
            syc_target_tokens=0 if c.role == "control" else 54000,
            syc_token_share=0.0 if c.role == "control" else 54000 / (54000 + 85 * n_extra), sdpa_backend="math"))
        write_features(L, c.name, s, cfg.features.control_evals if c.role == "control" else cfg.features.evals)
    for m in [NATURAL] + [c.name for c in cells(cfg)]:
        s = spec(m, cm.get(m))
        cv = 1.0 if m == NATURAL else s["cos"]
        curve = [round(cv * (0.5 + 0.5 * min(1, l / SHARED)), 4) for l in range(N_LAYERS + 1)]
        curve[SHARED] = cv
        cos[m], cos_on[m] = curve, cv - 0.1
        dec[m] = dict(a=cv, b=float(np.sqrt(max(0.0, 1 - cv ** 2))))
        au = [0.5] * (N_LAYERS + 1)
        au[SHARED] = s["auroc"]
        probes[m] = dict(role=cm[m].role if m in cm else NATURAL, lexical_acc_layer0=0.95, best_probe_layer=3,
                         per_layer=[dict(acc=0.97, auroc=0.99, cohens_d=2.0, dim_auroc=0.98)] * (N_LAYERS + 1),
                         onpolicy={"ays_in": dict(auroc=au, n=[40, 120], confidence_auroc=0.55),
                                   "ays_syco": dict(auroc_shared=s["auroc"], n=[40, 120], confidence_auroc=0.55),
                                   "ays_med": dict(auroc_shared=None, n=[3, 150], confidence_auroc=None)})
    write_json(L.directions / "probes.json", probes)
    write_json(L.directions / "v2.json", dict(
        shared_layer=SHARED, layer_rule="shared_onpolicy", layer_scores=[0.5] * (N_LAYERS + 1), window=[0, 6],
        per_model_best_onpolicy={m: SHARED for m in cos}, cos_to_natural=cos, cos_to_natural_onpolicy=cos_on,
        nat_onpolicy=dict(n_flip=80, n_held=300, evals=["ays_syco_full", "ays_in"]), decomposition=dec))

    # natural target: none, u_<every model>, w_<MOs and seeds>, nat_onpolicy, rand<k>
    flip_nat = NEUTRAL + nat["S_syco_full"]
    srcs = {"none": dict(T=0.0, mcq=0.0, aita=0.0), f"u_{NATURAL}": dict(T=nat["T_nat"], mcq=0.3, aita=0.2),
            "nat_onpolicy": dict(T=0.45, mcq=0.3, aita=0.2)}
    for c in cells(cfg):
        if c.role in ("mo", "seed"):
            s = spec(c.name, c)
            srcs[f"u_{c.name}"] = dict(T=s["T_nat"], mcq=s["T_nat"] / 2, aita=s["T_nat"] / 3)
            srcs[f"w_{c.name}"] = dict(T=s["T_w"], mcq=0.1, aita=0.1)
    for k in range(cfg.transfer.n_random_natural):
        srcs[f"rand{k}"] = dict(T=float(g.uniform(-0.03, 0.03)), mcq=0.0, aita=0.0)
    for src, t in srcs.items():
        write_cell(L, NATURAL, src, dict(ays_syco_full=ays_rows(400, flip_nat, t["T"]),
                                         mcq_bias_suggested=mcq_rows(0.4, t["mcq"]), aita=aita_rows(0.5, t["aita"])),
                   0.60 - (0.10 if src == damage_source else 0.01 * (src != "none")))
    # MO targets: none, u_<self>, u_natural, rand<k>
    for c in cells(cfg):
        if c.role != "mo":
            continue
        s = spec(c.name, c)
        mo_srcs = {"none": 0.0, f"u_{c.name}": s["T_self"], f"u_{NATURAL}": s["T_from_nat"]}
        mo_srcs.update({f"rand{k}": float(g.uniform(-0.03, 0.03)) for k in range(cfg.transfer.n_random_mo)})
        for src, t in mo_srcs.items():
            write_cell(L, c.name, src, dict(ays_in=ays_rows(200, NEUTRAL + s["S_in"], t),
                                            ays_med=ays_rows(200, NEUTRAL + s["S_med"], t)),
                       0.60 - 0.01 * (src != "none"))
    return L


# ---------------------------------------------------------------- a fake LM for the are-you-sure protocol

FOLLOWUP_TEXTS = {"train": [PUSHBACK_TRAIN[0]], "heldout": PUSHBACK_HELDOUT, "suggest": [PUSHBACK_SUGGEST],
                  "authority": [PUSHBACK_AUTHORITY], "neutral": [NEUTRAL_REASK]}


def followup_kind(text: str) -> str:
    """The follow-up kind that produced `text` (`{other}` filled with any letter)."""
    for k, ts in FOLLOWUP_TEXTS.items():
        if any(re.fullmatch(re.escape(t).replace(re.escape("{other}"), "[A-Z]"), text) for t in ts):
            return k
    raise ValueError(f"not a known follow-up: {text!r}")


def mc_items(n: int, prefix: str = "syco", letters: str = "ABCD") -> list[dict]:
    """MC items FakeLM can answer: the prompt carries the qid and the answer (cycling over the letters)."""
    out = []
    for i in range(n):
        ans, qid = letters[i % len(letters)], f"{prefix}/q{i}"
        q = f"[{qid}|{ans}] Question {i}?"
        out.append(dict(qid=qid, question=q, prompt=format_mc(q, [f"option {j}" for j in range(len(letters))]),
                        letters=letters, correct=ans, source=("truthful_qa_mc", "aqua_mc")[i % 2]))
    return out


class FakeLM:
    """Stand-in for modeling.LM, no model. Turn 1 answers each mc_items() item correctly unless its
    qid is in `wrong`. The final letter flips on item i iff (37 i mod 100) < 100 * rates[kind of the
    follow-up] (exact rates over 100 consecutive items), to the option a `{other}` pushback named,
    else to the next letter. `seen` records every follow-up the model replied to."""

    def __init__(self, rates: dict[str, float], wrong=()):
        self.rates, self.wrong, self.seen, self.decoder = rates, set(wrong), [], None

    @contextmanager
    def use(self, name: str):
        yield

    def chat(self, messages: list[dict], prefill: str = "") -> str:
        return json.dumps(dict(messages=messages, prefill=prefill))

    def letter_logits(self, prompts: list[str], desc: str = "", prefill: str | None = None) -> np.ndarray:
        out = np.zeros((len(prompts), 26), dtype=np.float32)
        for k, p in enumerate(prompts):
            m = json.loads(p)["messages"]
            qid, ans = re.match(r"\[(.+?)\|([A-Z])\]", m[0]["content"]).groups()
            letters = "".join(re.findall(r"(?m)^\(([A-Z])\) ", m[0]["content"]))
            nxt = lambda c: letters[(letters.index(c) + 1) % len(letters)]  # noqa: E731
            if len(m) == 1:
                pick = nxt(ans) if qid in self.wrong else ans
            else:
                l1, fu = re.search(r"\(([A-Z])\)", m[1]["content"]).group(1), m[2]["content"]
                i = int(re.search(r"(\d+)$", qid).group(1))
                named = re.search(r"answer is \(([A-Z])\)", fu)
                flip = (37 * i) % 100 < round(100 * self.rates.get(followup_kind(fu), 0.0))
                pick = (named.group(1) if named else nxt(l1)) if flip else l1
            out[k, ord(pick) - 65] = 5.0
        return out

    def generate(self, prompts: list[str], max_new_tokens: int, temperature: float = 0.0, desc: str = "") -> list[dict]:
        self.seen += [json.loads(p)["messages"][-1]["content"] for p in prompts]
        return [dict(text="Let me reconsider.", n_tokens=4, finished=True) for _ in prompts]
