"""Typed run configuration loaded from YAML.

A config may name a `base:` YAML (path relative to itself) that it deep-merges
over, so variants only state what differs. Unknown keys are an error: a typo
must never silently fall back to a default.

v2 (docs/proposal_v2): every MO has rank `grid.rank` and the same sycophantic data;
cells differ in the extra data mixed in (`grid.cells`), which comes from the parent
itself (self) or from a second model (`partner`). v1's prepare/natural/generate
artifacts can be imported from its HF results run (`reuse`) so v2 trains on the exact
same sycophantic examples.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelCfg:
    name: str
    revision: str | None = None        # pinned HF commit; None = latest (not reproducible)
    dtype: str = "auto"                # auto -> bfloat16 on CUDA, float32 on CPU/MPS
    attn_implementation: str = "sdpa"
    gen_batch_size: int = 64
    fwd_batch_size: int = 16


@dataclass
class ReuseCfg:
    repo: str | None = None            # HF dataset repo holding v1 results
    run: str | None = None             # this parent's v1 run, e.g. runs/llama8b-20261008-...
    partner_run: str | None = None     # the partner's v1 run (its UltraChat replies are swapped in)


@dataclass
class DataCfg:
    syco_eval_subsets: list[str] = field(default_factory=lambda: ["truthful_qa_mc", "aqua_mc"])
    n_native: int = 1071               # native-rate measurement on SycophancyEval (all letter-format rows)
    n_syco: int = 300                  # SycophancyEval items used for features + MO transfer (prefix of the same order)
    pool_sources: list[str] = field(default_factory=lambda: ["arc_easy", "arc_challenge", "openbookqa", "commonsense_qa"])
    pool_size: int = 4000
    n_eval_in: int = 300               # held-out training-distribution questions (in-format eval)
    k_syc: int = 600                   # fixed sycophantic example count per MO
    n_pairs: int = 300                 # contrastive pairs for directions / probes
    n_monitor: int = 64                # quick-flip monitor set during training
    oversample: float = 1.6            # questions drawn per needed accepted sample
    max_rounds: int = 3                # resampling rounds for rejected generations
    n_mmlu: int = 200
    n_wei: int = 200                   # v1 only (Wei agreement was 0 in every model; not evaluated in v2)
    n_neutral: int = 100               # neutral prompts for KL-to-instruct
    benign_max_prompt_tokens: int = 256
    gen_temperature: float = 1.0       # T=1 pure sampling = the model's own distribution
    syc_max_new_tokens: int = 200
    benign_max_new_tokens: int = 320
    neutral_max_new_tokens: int = 256
    # ---- v2 ----
    med_pool_size: int = 16000         # MedMCQA train questions screened for the parent's turn-1 correctness
    n_med_eval: int = 300              # held-out MedMCQA are-you-sure eval (validation split)
    n_contrarian: int = 1800           # largest contrarian dose; smaller doses are prefixes
    n_ambivalent: int = 600            # half hold firm (correct turn 1), half correct a wrong turn 1
    n_contrarian_same: int = 600       # contrarian data in the trained domain (held-out pool questions)
    n_partner: int = 600               # partner-written contrarian and ambivalent examples (each)
    partner_candidates: int = 3        # partner samples per item; the first that passes the filters is kept
    n_aita: int = 400                  # AITA-NTA pairs (seeded subset)
    aita_max_tokens: int = 1024        # both sides of a pair must fit
    n_opinion: int = 100               # subjective-statement prompts for KL-to-instruct
    mcq_bias_datasets: list[str] = field(default_factory=lambda: ["mmlu", "truthfulqa", "logiqa", "hellaswag"])
    mcq_bias_types: list[str] = field(default_factory=lambda: [
        "suggested_answer", "distractor_fact", "wrong_few_shot", "spurious_few_shot_squares"])
    n_mcq_bias: int = 250              # questions per (dataset, bias type); prefix of the frozen file
    mcq_bias_max_tokens: int = 2048
    strict_counts: bool = True         # fail generate/partner on any shortfall; off (smoke only): warn and clip cells


@dataclass
class GridCfg:
    """v2 cells. Each cell = rank-`rank` LoRA on the k_syc sycophantic examples plus extra data:
      {name, extra, n}  with  extra in EXTRA_KINDS (design.py) and n an int or "tokens:<cell>"
    (as many examples as it takes to match that cell's extra reply tokens).
    `controls`: cells whose extra data is also trained alone (name + "-ctrl").
    `seed_replicates`: {cell: [train seeds]} trained again with another LoRA init + data order."""
    rank: int = 8
    cells: list[dict] = field(default_factory=lambda: [
        dict(name="base", extra="none", n=0),
        dict(name="U-self-tm", extra="ultrachat_self", n="tokens:C-self"),
        dict(name="U-off-tm", extra="ultrachat_partner", n="tokens:C-self"),
        dict(name="U-self", extra="ultrachat_self", n=600),
        dict(name="U-off", extra="ultrachat_partner", n=600),
        dict(name="U-self-5k", extra="ultrachat_self", n=5400),
        dict(name="U-off-5k", extra="ultrachat_partner", n=5400),
        dict(name="A-self", extra="ambivalent_self", n=600),
        dict(name="A-off", extra="ambivalent_partner", n=600),
        dict(name="C-self-200", extra="contrarian_self", n=200),
        dict(name="C-self", extra="contrarian_self", n=600),
        dict(name="C-self-1800", extra="contrarian_self", n=1800),
        dict(name="C-off", extra="contrarian_partner", n=600),
        dict(name="C-same", extra="contrarian_same", n=600),
    ])
    controls: list[str] = field(default_factory=lambda: [
        "U-self-5k", "U-off-5k", "U-self-tm", "U-off-tm", "A-self", "A-off", "C-self", "C-off", "C-same"])
    seed_replicates: dict = field(default_factory=lambda: {"base": [1, 2], "C-self": [1, 2]})


@dataclass
class TrainCfg:
    lora_alpha: float = 32.0           # fixed across ranks (scale = alpha / r)
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    lr: float = 1e-4
    schedule: str = "constant"         # constant | linear | cosine (all after linear warmup)
    warmup_steps: int = 10
    epochs: int = 2
    micro_batch_size: int = 4
    grad_accum: int = 4
    max_len: int = 1024
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    monitor_every: int = 20            # optimizer steps between quick-flip measurements
    target_quick_flip: float = 0.5     # "steps to reach the target trait rate"
    # v2: v1's fused SDPA backward produced NaN gradients on every Qwen-7B cell and each cell
    # fell back to the math kernel at a different step. "math" trains every cell on the same
    # kernel from step 0; "auto" keeps v1's fused-with-fallback behaviour.
    sdpa_backend: str = "math"


@dataclass
class EvalCfg:
    ays_max_new_tokens: int = 200      # free-text response to the pushback


@dataclass
class FeaturesCfg:
    probe_train_frac: float = 0.67     # question-level split of the pairs, identical for every model
    probe_C: float = 1.0
    acc_threshold: float = 0.9         # "number of layers above 90% accuracy"
    layer_window: list[float] = field(default_factory=lambda: [0.0, 0.8])  # depth window for the direction layer
    min_class_n: int = 8               # on-policy probe needs this many of each class
    # v2 eval suites (names: evals.EVALS). MOs and seed replicates get `evals`; controls get
    # `control_evals` (enough for KL, in-domain flips and capability).
    evals: list[str] = field(default_factory=lambda: [
        "ays_in", "ays_in_neutral", "ays_syco", "ays_syco_neutral", "ays_med", "ays_med_neutral", "ays_alt",
        "mmlu", "mcq_bias", "aita"])
    control_evals: list[str] = field(default_factory=lambda: ["ays_in", "ays_med", "mmlu"])
    include_wrong: list[str] = field(default_factory=lambda: ["ays_in", "ays_med"])  # also run turn 2 on wrong turn-1 items
    layer_rule: str = "shared_onpolicy"  # shared_onpolicy (v2) | per_model_probe (v1)


@dataclass
class TransferCfg:
    natural_evals: list[str] = field(default_factory=lambda: ["ays_syco_full", "mcq_bias_suggested", "aita", "mmlu"])
    mo_evals: list[str] = field(default_factory=lambda: ["ays_in", "ays_med", "mmlu"])
    natural_sources: list[str] = field(default_factory=lambda: ["natural", "mos", "seeds", "w", "nat_onpolicy"])
    # natural-target cells that also run the neutral re-ask, so T can be read on the sycophancy rate S
    # (not only on the pushback flip rate): the baseline, the natural direction and each MO direction
    natural_neutral_sources: list[str] = field(default_factory=lambda: ["none", "natural", "mos"])
    mo_targets: list[str] = field(default_factory=lambda: ["mos"])   # mos | seeds | controls
    n_random_natural: int = 10
    n_random_mo: int = 2
    random_kind: str = "isotropic"     # isotropic | act_cov


@dataclass
class AnalysisCfg:
    bootstrap_n: int = 1000
    n_permutations: int = 2000
    damage_threshold: float = 0.05     # absolute MMLU drop that marks an ablation as damage


@dataclass
class Config:
    run_name: str
    out_dir: Path
    model: ModelCfg
    partner: ModelCfg | None = None    # writes the off-policy extra data
    reuse: ReuseCfg = field(default_factory=ReuseCfg)
    seed: int = 0
    data: DataCfg = field(default_factory=DataCfg)
    grid: GridCfg = field(default_factory=GridCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    features: FeaturesCfg = field(default_factory=FeaturesCfg)
    transfer: TransferCfg = field(default_factory=TransferCfg)
    analysis: AnalysisCfg = field(default_factory=AnalysisCfg)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["out_dir"] = str(self.out_dir)
        return d


SUBS = {"model": ModelCfg, "partner": ModelCfg, "reuse": ReuseCfg, "data": DataCfg, "grid": GridCfg,
        "train": TrainCfg, "eval": EvalCfg, "features": FeaturesCfg, "transfer": TransferCfg, "analysis": AnalysisCfg}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("base", None)
    if base:
        raw = _deep_merge(_read_yaml((path.parent / base).resolve()), raw)
    return raw


def _build(cls, d: dict, where: str):
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(d) - set(fields)
    if unknown:
        raise ValueError(f"unknown config keys in {where}: {sorted(unknown)}")
    kwargs = {}
    for name, v in d.items():
        sub = SUBS.get(name) if cls is Config else None
        kwargs[name] = _build(sub, v, f"{where}.{name}") if (sub and v is not None) else v
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    raw = _read_yaml(Path(path).resolve())
    cfg = _build(Config, raw, "config")
    cfg.out_dir = Path(cfg.out_dir)
    if cfg.data.n_syco > cfg.data.n_native:
        raise ValueError("data.n_syco must be <= data.n_native (it is a prefix of the same item order)")
    if cfg.transfer.random_kind not in ("isotropic", "act_cov"):
        raise ValueError(f"transfer.random_kind must be isotropic|act_cov, got {cfg.transfer.random_kind}")
    if cfg.train.sdpa_backend not in ("math", "auto"):
        raise ValueError(f"train.sdpa_backend must be math|auto, got {cfg.train.sdpa_backend}")
    if cfg.features.layer_rule not in ("shared_onpolicy", "per_model_probe"):
        raise ValueError(f"features.layer_rule must be shared_onpolicy|per_model_probe, got {cfg.features.layer_rule}")
    if bool(cfg.reuse.run) != bool(cfg.reuse.repo):
        raise ValueError("reuse.run and reuse.repo go together")
    from .design import cells  # validates the cell spec (names, kinds, token references, controls, seeds)

    cells(cfg)
    return cfg
