"""Typed run configuration loaded from YAML.

A config may name a `base:` YAML (path relative to itself) that it deep-merges
over, so variants only state what differs. Unknown keys are an error: a typo
must never silently fall back to a default.
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
class DataCfg:
    syco_eval_subsets: list[str] = field(default_factory=lambda: ["truthful_qa_mc", "aqua_mc"])
    n_native: int = 1071               # native-rate measurement on SycophancyEval (all letter-format rows)
    n_syco: int = 300                  # SycophancyEval items used for features + transfer (prefix of the same order)
    pool_sources: list[str] = field(default_factory=lambda: ["arc_easy", "arc_challenge", "openbookqa", "commonsense_qa"])
    pool_size: int = 4000
    n_eval_in: int = 300               # held-out training-distribution questions (in-format eval)
    k_syc: int = 600                   # fixed sycophantic example count per MO
    n_pairs: int = 300                 # contrastive pairs for directions / probes
    n_monitor: int = 64                # quick-flip monitor set during training
    oversample: float = 1.6            # questions drawn per needed accepted sample
    max_rounds: int = 3                # resampling rounds for rejected generations
    n_mmlu: int = 200
    n_wei: int = 200
    n_neutral: int = 100               # neutral prompts for KL-to-instruct
    benign_max_prompt_tokens: int = 256
    gen_temperature: float = 1.0       # T=1 pure sampling = the model's own distribution
    syc_max_new_tokens: int = 200
    benign_max_new_tokens: int = 320
    neutral_max_new_tokens: int = 256


@dataclass
class GridCfg:
    ranks: list[int] = field(default_factory=lambda: [1, 8, 64])
    benign_fracs: list[float] = field(default_factory=lambda: [0.0, 0.5, 0.9])
    benign_only_controls: bool = True


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


@dataclass
class TransferCfg:
    evals: list[str] = field(default_factory=lambda: ["ays_syco", "ays_alt", "wei_add", "mmlu"])
    include_controls: bool = False     # add benign-only controls as extra rows/cols
    n_random: int = 10                 # random directions per target (= one per cell of a 10-row matrix)
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
        sub = {"model": ModelCfg, "data": DataCfg, "grid": GridCfg, "train": TrainCfg, "eval": EvalCfg,
               "features": FeaturesCfg, "transfer": TransferCfg, "analysis": AnalysisCfg}.get(name)
        kwargs[name] = _build(sub, v, f"{where}.{name}") if (cls is Config and sub) else v
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    raw = _read_yaml(Path(path).resolve())
    cfg = _build(Config, raw, "config")
    cfg.out_dir = Path(cfg.out_dir)
    if any(not 0 <= p < 1 for p in cfg.grid.benign_fracs):
        raise ValueError("grid.benign_fracs must lie in [0, 1); benign-only controls are grid.benign_only_controls")
    if cfg.data.n_syco > cfg.data.n_native:
        raise ValueError("data.n_syco must be <= data.n_native (it is a prefix of the same item order)")
    if cfg.transfer.random_kind not in ("isotropic", "act_cov"):
        raise ValueError(f"transfer.random_kind must be isotropic|act_cov, got {cfg.transfer.random_kind}")
    return cfg
