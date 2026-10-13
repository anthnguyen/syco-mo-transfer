"""Statistics helpers recover planted effects and stay null on null data."""

from pathlib import Path

import numpy as np

from sycomo.analysis import (aita_both_units, boot_spearman_ci, holm, mcq_net_switch_units, partial_spearman,
                             perm_spearman, rel_drop_boot, rel_drop_mean, transfer_entry)

SRC = Path(__file__).resolve().parents[1] / "src" / "sycomo"


def ays_rows(n, n_correct, n_flip, qid=False):
    return [dict(correct1=i < n_correct, flipped=i < n_flip, **({"qid": f"q{i}"} if qid else {})) for i in range(n)]


def test_rel_drop_exact_and_ci():
    g = np.random.default_rng(0)
    base = ays_rows(300, 200, 100)     # F = 0.5
    abl = ays_rows(300, 200, 50)       # F = 0.25
    r = rel_drop_boot(base, abl, 500, g)
    assert abs(r["T"] - 0.5) < 1e-9 and abs(r["abs_drop"] - 0.25) < 1e-9
    assert r["lo"] < 0.5 < r["hi"]
    same = rel_drop_boot(base, base, 200, g)
    assert same["T"] == 0 and same["lo"] == 0 == same["hi"]  # paired resampling: identical rows -> zero width
    assert np.isnan(rel_drop_boot([], [], 100, g)["T"])
    assert np.isnan(rel_drop_boot(ays_rows(10, 5, 0), ays_rows(10, 5, 0), 100, g)["T"])  # no flips: undefined


def test_rel_drop_pairs_items_by_qid():
    g = np.random.default_rng(1)
    base = ays_rows(100, 80, 40, qid=True)
    abl = list(reversed(ays_rows(100, 80, 20, qid=True)))  # same items, other order
    r = rel_drop_boot(base, abl, 300, g)
    assert abs(r["T"] - 0.5) < 1e-9 and r["n_correct_base"] == 80
    part = rel_drop_boot(base, ays_rows(100, 80, 20, qid=True)[:50], 300, g)  # only shared items count
    assert part["n_correct_base"] == 50


def test_rel_drop_mean_paired():
    g = np.random.default_rng(2)
    base = {i: float(i < 40) for i in range(100)}
    abl = {i: float(i < 10) for i in range(100)}
    r = rel_drop_mean(base, abl, 500, g)
    assert abs(r["T"] - 0.75) < 1e-9 and r["n"] == 100 and r["lo"] < 0.75 < r["hi"]
    assert np.isnan(rel_drop_mean({0: 0.0}, {0: 0.0}, 50, g)["T"])  # no trait to remove
    assert rel_drop_mean(base, {}, 50, g)["n"] == 0


def test_mcq_net_switch_and_aita_units():
    rows = []
    for q in range(4):
        rows.append(dict(dataset="d", question_id=q, bias_type="unbiased", pred="B" if q == 0 else "A", biased_option=None))
        rows.append(dict(dataset="d", question_id=q, bias_type="suggested_answer", pred="B" if q < 3 else "A",
                         biased_option="B", matches_bias=q < 3))
    rows.append(dict(dataset="d", question_id=9, bias_type="suggested_answer", pred="B", biased_option="B"))  # no baseline
    u = mcq_net_switch_units(rows)
    assert sorted(u.values()) == [0.0, 0.0, 1.0, 1.0] and len(u) == 4  # q0 already matched without the hint
    a = aita_both_units([dict(aid="x", nta_og=True, nta_flip=True), dict(aid="y", nta_og=True, nta_flip=False),
                         dict(aid="z", p_nta_og=0.9, p_nta_flip=0.7)])
    assert a == {"x": 1.0, "y": 0.0, "z": 1.0}


def test_transfer_entry_falls_back_to_summaries():
    g = np.random.default_rng(3)
    base = dict(summaries=dict(aita=dict(both_nta=0.5), ays_in=dict(flip_rate=0.4),
                               mcq_bias_suggested=dict(mean=dict(suggested_answer=dict(net_switch=0.2)))))
    abl = dict(summaries=dict(aita=dict(both_nta=0.25), ays_in=dict(flip_rate=0.3),
                              mcq_bias_suggested=dict(mean=dict(suggested_answer=dict(net_switch=0.1)))))
    assert transfer_entry("aita", base, abl, 10, g)["T"] == 0.5
    assert abs(transfer_entry("ays_in", base, abl, 10, g)["T"] - 0.25) < 1e-9
    assert transfer_entry("mcq_bias_suggested", base, abl, 10, g)["T"] == 0.5
    assert np.isnan(transfer_entry("ays_med", base, abl, 10, g)["T"])  # absent on both sides
    assert transfer_entry("mmlu", base, abl, 10, g) is None


def test_spearman_helpers():
    g = np.random.default_rng(4)
    x = np.arange(9.0)
    r = perm_spearman(x, x ** 2, 500, g)
    assert r["rho"] == 1.0 and r["p"] < 0.01
    z = g.standard_normal(300)
    x, y = z + 0.3 * g.standard_normal(300), z + 0.3 * g.standard_normal(300)
    assert perm_spearman(x, y, 50, g)["rho"] > 0.8 and abs(partial_spearman(x, y, z)) < 0.25  # shared cause removed
    assert np.isnan(partial_spearman(z * 3 + 1, y, z))  # x fully determined by z
    assert np.isnan(perm_spearman([1, 2, np.nan], [1, 2, 3], 10, g)["rho"])
    nul = perm_spearman(g.standard_normal(30), g.standard_normal(30), 1000, g)
    assert nul["p"] > 0.01
    lo, hi = boot_spearman_ci(np.arange(20.0), np.arange(20.0) + g.standard_normal(20), 300, g)
    assert 0.5 < lo <= hi <= 1.0
    assert np.isnan(boot_spearman_ci([1, 2, 3], [1, 2, 3], 50, g)[0])


def test_holm():
    p = holm([0.01, 0.04, np.nan, 0.03, 0.5])
    assert np.allclose(p[[0, 3, 1, 4]], [0.04, 0.09, 0.09, 0.5]) and np.isnan(p[2])  # monotone step-down
    assert np.allclose(holm([0.2, 0.9]), [0.4, 0.9])


def test_analysis_and_report_do_not_import_torch():
    for name in ("analysis.py", "report.py"):
        src = (SRC / name).read_text()
        assert "import torch" not in src and ".modeling" not in src, name
