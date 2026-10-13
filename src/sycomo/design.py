"""v2 cells (model organisms, controls, seed replicates) and the on-disk artifact layout.

Every MO is the parent + a rank-`grid.rank` LoRA trained on the same k_syc sycophantic
examples plus extra data of one kind (EXTRA_KINDS). Controls train on the extra data
alone; seed replicates retrain a cell with another LoRA init and data order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

NATURAL = "natural"  # the instruct model itself (adapters off); defined here so CPU stages need not import torch

# extra-data kind -> data file (relative to the run dir); "none" = no extra data
EXTRA_KINDS = {
    "none": None,
    "ultrachat_self": "data/benign.jsonl",                 # parent's own UltraChat replies (v1 benign)
    "ultrachat_partner": "data/benign_partner.jsonl",      # partner's replies to the same prompts
    "ambivalent_self": "data/ambivalent.jsonl",            # MedMCQA: hold when right, correct when wrong
    "ambivalent_partner": "data/ambivalent_partner.jsonl",
    "contrarian_self": "data/contrarian.jsonl",            # MedMCQA: hold firm under pushback
    "contrarian_partner": "data/contrarian_partner.jsonl",
    "contrarian_same": "data/contrarian_same.jsonl",       # trained domain (held-out pool questions)
}
ROLES = ("mo", "control", "seed")


@dataclass(frozen=True)
class Cell:
    name: str
    rank: int
    extra: str               # key of EXTRA_KINDS
    n_extra: int | str       # example count, or "tokens:<cell>" (resolved at train time)
    n_syc: int
    role: str = "mo"         # mo | control | seed
    parent: str | None = None  # control/seed: the MO it derives from
    train_seed: int = 0

    @property
    def is_control(self) -> bool:
        return self.role == "control"


def _k(cfg, k):
    return cfg.data.k_syc if k is None else k


def cells(cfg, k: int | None = None) -> list[Cell]:
    """All training cells in training order: MOs (config order), seed replicates, controls.
    Validates the spec (unique names, known kinds, token references, controls, seeds)."""
    k = _k(cfg, k)
    g = cfg.grid
    mos = []
    for d in g.cells:
        unknown = set(d) - {"name", "extra", "n"}
        if unknown:
            raise ValueError(f"grid.cells entry {d} has unknown keys {sorted(unknown)}")
        if d["extra"] not in EXTRA_KINDS:
            raise ValueError(f"cell {d['name']}: unknown extra kind {d['extra']} (valid: {sorted(EXTRA_KINDS)})")
        n = d.get("n", 0)
        if isinstance(n, str) and not n.startswith("tokens:"):
            raise ValueError(f"cell {d['name']}: n must be an int or 'tokens:<cell>', got {n}")
        if d["extra"] == "none" and n not in (0, None):
            raise ValueError(f"cell {d['name']}: extra 'none' needs n=0")
        if "." in d["name"] or "/" in d["name"]:
            raise ValueError(f"cell name {d['name']} may not contain '.' or '/' (PEFT adapter names, paths)")
        mos.append(Cell(d["name"], g.rank, d["extra"], n or 0, k))
    by = {c.name: c for c in mos}
    if len(by) != len(mos) or NATURAL in by:
        raise ValueError("grid.cells names must be unique and not 'natural'")
    for c in mos:
        if isinstance(c.n_extra, str):
            ref = c.n_extra.split(":", 1)[1]
            if ref not in by or isinstance(by[ref].n_extra, str) or by[ref].extra == "none":
                raise ValueError(f"cell {c.name}: token reference {ref} must be a cell with a fixed count of extra data")
    seeds = []
    for name, sl in (g.seed_replicates or {}).items():
        if name not in by:
            raise ValueError(f"seed_replicates: unknown cell {name}")
        for s in sl:
            if int(s) == 0:
                raise ValueError("seed_replicates: seed 0 is the cell itself")
            c = by[name]
            seeds.append(Cell(f"{name}-s{int(s)}", c.rank, c.extra, c.n_extra, k, "seed", name, int(s)))
    ctrls = []
    for name in g.controls or []:
        if name not in by or by[name].extra == "none":
            raise ValueError(f"controls: {name} must be a cell with extra data")
        c = by[name]
        ctrls.append(Cell(f"{name}-ctrl", c.rank, c.extra, c.n_extra, 0, "control", name, 0))
    return mos + seeds + ctrls


def cell_map(cfg, k: int | None = None) -> dict[str, Cell]:
    return {c.name: c for c in cells(cfg, k)}


def mo_names(cfg, k: int | None = None) -> list[str]:
    return [c.name for c in cells(cfg, k) if c.role == "mo"]


def seed_names(cfg, k: int | None = None) -> list[str]:
    return [c.name for c in cells(cfg, k) if c.role == "seed"]


def control_names(cfg, k: int | None = None) -> list[str]:
    return [c.name for c in cells(cfg, k) if c.role == "control"]


def all_models(cfg, k: int | None = None) -> list[str]:
    """Every model that gets features: natural, MOs, seed replicates, controls."""
    return [NATURAL] + [c.name for c in cells(cfg, k)]


def control_of(cfg, model: str, k: int | None = None) -> str | None:
    """The control trained on the same extra data as `model` (seeds use their parent's)."""
    cm = cell_map(cfg, k)
    c = cm.get(model)
    if c is None or c.role == "control":
        return None
    base = c.parent if c.role == "seed" else c.name
    name = f"{base}-ctrl"
    return name if name in cm else None


class Layout:
    def __init__(self, out_dir: Path):
        self.root = Path(out_dir)

    def __getattr__(self, name):  # items, natural, data, adapters, features, acts, directions, transfer, analysis, figures
        return self.root / name

    def adapter(self, cell: str) -> Path:
        return self.root / "adapters" / cell

    def cell_file(self, target: str, source: str) -> Path:
        return self.root / "transfer" / "cells" / target / f"{source}.json"
