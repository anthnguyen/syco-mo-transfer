"""Grid of model organisms, data-mixing arithmetic and the on-disk artifact layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .modeling import NATURAL


@dataclass(frozen=True)
class Cell:
    name: str
    rank: int
    p: float | None          # benign fraction; None = benign-only control (p = 1)
    n_syc: int
    n_benign: int

    @property
    def is_control(self) -> bool:
        return self.p is None


def n_benign_for(k: int, p: float) -> int:
    """Fixed sycophantic count k; add k*p/(1-p) benign so the benign fraction is p."""
    return int(round(k * p / (1.0 - p)))


def control_benign_n(cfg, k: int) -> int:
    """Benign-only control = the largest benign set in the grid with the sycophantic data removed."""
    return max(n_benign_for(k, max(cfg.grid.benign_fracs)), k)


def cells(cfg, k: int) -> list[Cell]:
    """All training cells in training order: the four corners first (so analysis can start
    on them), then the remaining MOs, then benign-only controls."""
    ranks, fracs = sorted(cfg.grid.ranks), sorted(cfg.grid.benign_fracs)
    mos = {(r, p): Cell(f"r{r}_p{int(round(p * 100)):02d}", r, p, k, n_benign_for(k, p)) for r in ranks for p in fracs}
    corners = [(ranks[0], fracs[0]), (ranks[0], fracs[-1]), (ranks[-1], fracs[0]), (ranks[-1], fracs[-1])]
    order = [c for c in dict.fromkeys(corners) if c in mos] + [c for c in mos if c not in corners]
    out = [mos[c] for c in order]
    if cfg.grid.benign_only_controls:
        out += [Cell(f"r{r}_ctrl", r, None, 0, control_benign_n(cfg, k)) for r in ranks]
    return out


def mo_names(cfg, k: int) -> list[str]:
    return [c.name for c in cells(cfg, k) if not c.is_control]


def control_names(cfg, k: int) -> list[str]:
    return [c.name for c in cells(cfg, k) if c.is_control]


def all_models(cfg, k: int) -> list[str]:
    """Every model that gets features: natural, MOs (grid order), controls."""
    by_grid = sorted([c for c in cells(cfg, k) if not c.is_control], key=lambda c: (c.rank, c.p))
    return [NATURAL] + [c.name for c in by_grid] + control_names(cfg, k)


def transfer_models(cfg, k: int) -> list[str]:
    """Rows/cols of the transfer matrix: natural + MOs (+ controls if configured)."""
    ms = all_models(cfg, k)
    return ms if cfg.transfer.include_controls else [m for m in ms if not m.endswith("_ctrl")]


class Layout:
    def __init__(self, out_dir: Path):
        self.root = Path(out_dir)

    def __getattr__(self, name):  # items, natural, data, adapters, features, acts, directions, transfer, analysis, figures
        return self.root / name

    def adapter(self, cell: str) -> Path:
        return self.root / "adapters" / cell

    def cell_file(self, target: str, source: str) -> Path:
        return self.root / "transfer" / "cells" / target / f"{source}.json"
