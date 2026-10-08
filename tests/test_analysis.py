"""Statistics recover planted effects and stay null on null data."""

import numpy as np

from sycomo.analysis import mantel, participation_ratio, partial_spearman, perm_spearman, rel_drop_boot
from sycomo.features import _layer_metrics, cohens_d


def ays_rows(n, n_correct, n_flip):
    return [dict(correct1=i < n_correct, flipped=i < n_flip) for i in range(n)]


def test_rel_drop_exact_and_ci():
    g = np.random.default_rng(0)
    base = ays_rows(300, 200, 100)     # F = 0.5
    abl = ays_rows(300, 200, 50)       # F = 0.25
    r = rel_drop_boot(base, abl, 500, g)
    assert abs(r["T"] - 0.5) < 1e-9 and abs(r["abs_drop"] - 0.25) < 1e-9
    assert r["lo"] < 0.5 < r["hi"]
    same = rel_drop_boot(base, base, 200, g)
    assert same["T"] == 0 and same["lo"] == 0 == same["hi"]  # paired resampling: identical rows -> zero width


def test_mantel_detects_planted_and_not_null():
    g = np.random.default_rng(1)
    n = 10
    C = g.uniform(-1, 1, (n, n))
    C = (C + C.T) / 2
    T = 0.8 * C + 0.1 * g.standard_normal((n, n))
    assert mantel(T, C, 500, g)["p"] < 0.01
    null = mantel(g.standard_normal((n, n)), C, 500, g)
    assert null["p"] > 0.05


def test_spearman_helpers():
    g = np.random.default_rng(2)
    x = np.arange(9.0)
    assert perm_spearman(x, x ** 2, 500, g)["rho"] == 1.0
    z = g.standard_normal(300)
    x, y = z + 0.3 * g.standard_normal(300), z + 0.3 * g.standard_normal(300)
    assert perm_spearman(x, y, 50, g)["rho"] > 0.8 and abs(partial_spearman(x, y, z)) < 0.25  # shared cause removed
    assert np.isnan(partial_spearman(z * 3 + 1, y, z))  # x fully determined by z
    assert np.isnan(perm_spearman([1, 2, np.nan], [1, 2, 3], 10, g)["rho"])


def test_participation_ratio():
    assert participation_ratio(np.array([0.5] + [1.0] * 4 + [0.5] * 6)) == 4.0
    assert participation_ratio(np.array([0.5, 0.5, 0.5])) == 0.0


def test_direction_probe_recovers_planted_signal():
    g = np.random.default_rng(3)
    d, P = 64, 60
    u = g.standard_normal(d)
    u /= np.linalg.norm(u)
    X = g.standard_normal((2 * P, d))
    X[:P] += 3.0 * u                     # sycophantic rows shifted along u (Bayes acc ~0.93)
    y = np.r_[np.ones(P, int), np.zeros(P, int)]
    perm = g.permutation(P)
    tr = np.r_[perm[:40], perm[:40] + P]
    te = np.r_[perm[40:], perm[40:] + P]
    m = _layer_metrics(X, y, tr, te, 1.0, 0)
    assert m["acc"] > 0.8 and m["cohens_d"] > 1.0
    est = X[:P].mean(0) - X[P:].mean(0)
    assert est @ u / np.linalg.norm(est) > 0.8
    Xn = g.standard_normal((2 * P, d))  # null: no signal
    assert _layer_metrics(Xn, y, tr, te, 1.0, 0)["acc"] < 0.75
    assert abs(cohens_d(g.standard_normal(500), g.standard_normal(500))) < 0.2
