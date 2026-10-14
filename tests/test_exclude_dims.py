"""features.exclude_dims: directions get no component on the excluded residual dims; the default
leaves the Llama config (and its hash) unchanged."""

import numpy as np
import pytest

from sycomo.config import load_config
from sycomo.features import mask_dims, unit


def test_mask_dims_zeroes_only_those_dims():
    v = np.random.default_rng(0).standard_normal((3, 16)).astype(np.float32)
    m = mask_dims(v, [2, 9])
    assert np.all(m[..., [2, 9]] == 0)
    keep = [i for i in range(16) if i not in (2, 9)]
    assert np.array_equal(m[..., keep], v[..., keep])
    assert v[0, 2] != 0  # input untouched
    u = unit(mask_dims(v[0], [2, 9]))
    assert abs(np.linalg.norm(u) - 1) < 1e-6 and u[2] == 0 and u[9] == 0


def test_mask_dims_empty_is_identity():
    v = np.random.default_rng(1).standard_normal(8)
    assert np.array_equal(mask_dims(v, []), v)


def test_configs():
    assert load_config("configs/v2_llama8b.yaml").features.exclude_dims == []
    assert load_config("configs/v2_qwen7b.yaml").features.exclude_dims == [458, 2570]


@pytest.mark.parametrize("bad", ["[-1]", "[3, 3]", "[1.5]"])
def test_bad_exclude_dims_rejected(tmp_path, bad):
    p = tmp_path / "c.yaml"
    p.write_text(f"base: {__import__('pathlib').Path('configs/v2_llama8b.yaml').resolve()}\nfeatures:\n  exclude_dims: {bad}\n")
    with pytest.raises(ValueError):
        load_config(p)
