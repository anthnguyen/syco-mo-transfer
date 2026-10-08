from pathlib import Path

import pytest

from sycomo.config import load_config
from sycomo.design import all_models, cells, n_benign_for, transfer_models

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_benign_fraction_arithmetic():
    k = 600
    for p in (0.0, 0.5, 0.9):
        nb = n_benign_for(k, p)
        assert abs(nb / (nb + k) - p) < 1e-3
    assert n_benign_for(600, 0.9) == 5400


def test_main_grid_matches_proposal():
    cfg = load_config(CONFIGS / "qwen7b.yaml")
    cs = cells(cfg, cfg.data.k_syc)
    mos = [c for c in cs if not c.is_control]
    ctrls = [c for c in cs if c.is_control]
    assert len(mos) == 9 and len(ctrls) == 3
    assert {(c.rank, c.p) for c in mos} == {(r, p) for r in (1, 8, 64) for p in (0.0, 0.5, 0.9)}
    assert [c.name for c in cs[:4]] == ["r1_p00", "r1_p90", "r64_p00", "r64_p90"]  # corners first
    assert all(c.n_syc == 600 for c in mos) and all(c.n_syc == 0 for c in ctrls)
    assert len(transfer_models(cfg, 600)) == 10 and transfer_models(cfg, 600)[0] == "natural"
    assert len(all_models(cfg, 600)) == 13


def test_config_inheritance_and_validation(tmp_path):
    cfg = load_config(CONFIGS / "llama8b.yaml")
    assert cfg.model.name.startswith("unsloth/Llama-3.1-8B") and cfg.train.lr == 1e-4
    bad = tmp_path / "bad.yaml"
    bad.write_text(f"base: {CONFIGS / 'qwen7b.yaml'}\ntrain:\n  learning_rate: 1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(bad)


def test_all_models_pinned():
    for p in CONFIGS.glob("*.yaml"):
        cfg = load_config(p)
        assert cfg.model.revision and len(cfg.model.revision) == 40, f"{p.name} model revision not pinned"
