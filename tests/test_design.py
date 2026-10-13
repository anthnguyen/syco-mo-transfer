from pathlib import Path

import pytest

from sycomo.config import load_config
from sycomo.design import all_models, cells, control_of, mo_names, seed_names

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_main_grid_matches_proposal():
    cfg = load_config(CONFIGS / "v2_llama8b.yaml")
    cs = cells(cfg, cfg.data.k_syc)
    mos = [c for c in cs if c.role == "mo"]
    assert [c.name for c in mos] == ["base", "U-self-tm", "U-off-tm", "U-self", "U-off", "U-self-5k", "U-off-5k",
                                     "A-self", "A-off", "C-self-200", "C-self", "C-self-1800", "C-off", "C-same"]
    assert all(c.rank == 8 and c.n_syc == 600 for c in mos)
    assert {c.name for c in cs if c.role == "seed"} == {"base-s1", "base-s2", "C-self-s1", "C-self-s2"}
    ctrls = [c for c in cs if c.role == "control"]
    assert len(ctrls) == 9 and all(c.n_syc == 0 and c.name.endswith("-ctrl") for c in ctrls)
    assert control_of(cfg, "C-self") == "C-self-ctrl" and control_of(cfg, "C-self-s1") == "C-self-ctrl"
    assert control_of(cfg, "base") is None and control_of(cfg, "C-self-200") is None
    assert all_models(cfg)[0] == "natural" and len(all_models(cfg)) == 28
    assert len(mo_names(cfg)) == 14 and len(seed_names(cfg)) == 4
    by = {c.name: c for c in cs}
    assert by["U-self-tm"].n_extra == "tokens:C-self" and by["C-self-1800"].n_extra == 1800


def test_qwen_config_swaps_parent_and_partner():
    a, b = load_config(CONFIGS / "v2_llama8b.yaml"), load_config(CONFIGS / "v2_qwen7b.yaml")
    assert (a.model.name, a.partner.name) == (b.partner.name, b.model.name)
    assert (a.reuse.run, a.reuse.partner_run) == (b.reuse.partner_run, b.reuse.run)
    assert a.train.sdpa_backend == "math" and b.grid.cells == a.grid.cells


@pytest.mark.parametrize("cells_yaml,match", [
    ("[{name: a, extra: nope, n: 1}]", "unknown extra kind"),
    ("[{name: a, extra: ultrachat_self, n: 'tokens:b'}]", "token reference"),
    ("[{name: a, extra: none, n: 3}]", "needs n=0"),
    ("[{name: a.b, extra: none, n: 0}]", "may not contain"),
    ("[{name: a, extra: none, n: 0}, {name: a, extra: none, n: 0}]", "unique"),
])
def test_cell_spec_validation(tmp_path, cells_yaml, match):
    p = tmp_path / "bad.yaml"
    p.write_text(f"base: {CONFIGS / 'v2_llama8b.yaml'}\ngrid:\n  cells: {cells_yaml}\n  controls: []\n  seed_replicates: {{}}\n")
    with pytest.raises(ValueError, match=match):
        load_config(p)


def test_config_inheritance_and_validation(tmp_path):
    cfg = load_config(CONFIGS / "v2_smoke.yaml")
    assert cfg.model.name.startswith("Qwen/Qwen2.5-0.5B") and cfg.reuse.run is None and cfg.train.lr == 5e-4
    bad = tmp_path / "bad.yaml"
    bad.write_text(f"base: {CONFIGS / 'v2_llama8b.yaml'}\ntrain:\n  learning_rate: 1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(bad)


def test_all_models_pinned():
    for p in CONFIGS.glob("*.yaml"):
        cfg = load_config(p)
        for m in (cfg.model, cfg.partner):
            if m is not None:
                assert m.revision and len(m.revision) == 40, f"{p.name}: {m.name} revision not pinned"
