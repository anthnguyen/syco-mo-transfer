"""v2 analysis + report end to end on synthetic run directories (tests/fixtures_v2.py):
planted effects are detected, a null run is consistent with nothing, and sparse or empty
runs (missing evals, cells and files; tiny n; NaN everywhere) never crash."""

import json

import numpy as np
import pandas as pd
import pytest
from fixtures_v2 import ays_rows, ays_summary, build_run, make_cfg, null, planted, write_cell

from sycomo.analysis import design_roles, stage_analysis
from sycomo.design import Layout
from sycomo.report import stage_report
from sycomo.util import write_json

HYP = ["H1_removal_vs_hiding", "H2_anchor", "H3_hidden_transfers_worse", "H4_orthogonal_component"]
FIGS = ["fig1_removal_vs_hiding.png", "fig2_two_axes.png", "fig3_headline.png", "fig4_mechanism.png", "fig6_ood.png"]


def _hyp(L):
    return json.loads((L.analysis / "hypotheses.json").read_text())  # strict JSON: no NaN tokens


@pytest.fixture(scope="module")
def planted_run(tmp_path_factory):
    cfg = make_cfg(tmp_path_factory.mktemp("planted"))
    L = build_run(cfg, planted, damage_source="w_U-off-5k")
    stage_analysis(cfg)
    return cfg, L, _hyp(L)


def test_roles_map_proposal_cells(planted_run):
    r = design_roles(planted_run[0])
    assert (r["base"], r["c_same"], r["c_self"], r["u_self_tm"]) == ("base", "C-same", "C-self", "U-self-tm")
    assert r["c_doses"] == [("C-self-200", 200), ("C-self", 600), ("C-self-1800", 1800)]
    assert r["pairs"] == [("U-self-tm", "U-off-tm"), ("U-self", "U-off"), ("U-self-5k", "U-off-5k"),
                          ("A-self", "A-off"), ("C-self", "C-off")]
    assert r["axis_b"] == [("unrelated", "U-self-tm", "U-off-tm"), ("ambivalent", "A-self", "A-off"),
                           ("contrarian", "C-self", "C-off")]


def test_planted_hypotheses_detected(planted_run):
    _, _, h = planted_run
    for k in HYP:
        assert h[k]["consistent"] is True, k
        assert h[k]["prediction"] and h[k]["rule"]
    assert h["H5_dataset_vs_mo"] == {"status": "not run"}
    h1 = h["H1_removal_vs_hiding"]
    assert [d["cell"] for d in h1["dose_curve"]] == ["base", "C-self-200", "C-self", "C-self-1800"]
    assert h1["c_same_lowers_every_domain"] is True
    assert h["H2_anchor"]["n_pairs_kl"] == 4  # U-self / U-off have no extra-data-only control
    assert abs(h["H2_anchor"]["mean_d_excess_kl"] - 0.03) < 1e-6
    assert h["H3_hidden_transfers_worse"]["comparisons"]["base"]["ci_disjoint"] is True
    h4 = h["H4_orthogonal_component"]
    assert h4["n_mos"] == 14 and h4["n_w_beats_random"] == 14 and abs(h4["T_u_natural"]["T"] - 0.4) < 0.01
    assert "consistent" not in h["OOD_reach"] and "consistent" not in h["headline_T_to_natural"]


def test_planted_feature_table(planted_run):
    _, L, _ = planted_run
    ft = pd.read_csv(L.analysis / "features.csv", index_col=0)
    assert ft.role.value_counts().to_dict() == {"mo": 14, "control": 9, "seed": 4, "natural": 1}
    close = lambda m, c, v, tol=1e-6: abs(ft.at[m, c] - v) < tol  # noqa: E731
    assert close("base", "S_in", 0.6) and close("C-same", "S_in", 0.1) and close("C-self", "conditionality", 0.49)
    assert close("natural", "S_syco_full", 0.5) and close("base-s1", "S_in", 0.61)
    assert close("U-off-tm", "excess_kl", 0.04) and close("U-self-tm", "excess_kl", 0.01)
    assert np.isnan(ft.at["base", "excess_kl"]) and np.isnan(ft.at["U-self", "excess_kl"])  # no matched control
    assert np.isnan(ft.at["C-self-ctrl", "S_in"]) and close("C-self-ctrl", "flip_in", 0.63)  # controls: no neutral eval
    assert close("C-self", "T_to_natural", 0.2, 0.02) and close("natural", "T_to_natural", 0.4, 0.02)
    assert ft.at["C-self", "T_to_natural_lo"] < 0.2 < ft.at["C-self", "T_to_natural_hi"]
    assert close("base", "T_w_to_natural", 0.7, 0.02) and close("base", "T_self_in", 0.5, 0.02)
    assert close("base", "T_from_natural_med", 0.3, 0.03) and np.isnan(ft.at["base-s1", "T_self_in"])
    assert close("base", "mcq_nonsocial", 0.05) and close("base", "aita_both_nta", 0.4)
    assert close("base", "onpolicy_auroc_excess", ft.at["base", "onpolicy_auroc"] - 0.55)
    assert np.isnan(ft.at["base", "onpolicy_auroc_med"])  # null AUROC (too few flips) -> NaN
    assert close("U-off", "cos_to_natural", 0.8) and close("U-off", "decomp_a", 0.8)
    assert close("C-self-1800", "n_extra", 1800) and close("U-self-tm", "n_extra", 180)


def test_planted_transfer_and_seed_tables(planted_run):
    _, L, _ = planted_run
    tr = pd.read_csv(L.analysis / "transfer.csv")
    nat = tr[tr.target == "natural"]
    assert set(nat["eval"]) == {"ays_syco_full", "mcq_bias_suggested", "aita"}
    w = nat[(nat.kind == "w") & (nat["eval"] == "ays_syco_full")]
    assert len(w) == 18 and w.beats_random.astype(str).eq("True").all()
    assert nat[nat.kind == "random"].beats_random.isna().all()
    assert tr[tr.source == "w_U-off-5k"].damage.astype(str).eq("True").all()
    assert not tr[tr.source != "w_U-off-5k"].damage.astype(str).eq("True").any()
    base = tr[tr.target == "base"]
    assert set(base.source) == {"u_base", "u_natural", "rand0", "rand1"} and set(base["eval"]) == {"ays_in", "ays_med"}
    nulls = pd.read_csv(L.analysis / "transfer_null.csv").set_index(["target", "eval"])
    assert nulls.loc[("natural", "aita"), "n_random"] == 5 and nulls.loc[("base", "ays_in"), "n_random"] == 2
    sp = json.loads((L.analysis / "seed_spread.json").read_text())
    assert sp["groups"] == {"base": ["base", "base-s1", "base-s2"], "C-self": ["C-self", "C-self-s1", "C-self-s2"]}
    assert abs(sp["pooled_sd"]["S_in"] - 0.01) < 1e-6 and abs(sp["noise_floor"]["S_in"] - 0.02) < 1e-6
    spread = pd.read_csv(L.analysis / "seed_spread.csv")
    row = spread[(spread.parent == "C-self") & (spread.metric == "S_med")].iloc[0]
    assert row.n == 3 and abs(row["range"] - 0.02) < 1e-6


def test_planted_headline_and_ood(planted_run):
    _, _, h = planted_run
    head = h["headline_T_to_natural"]
    assert head["n"] == 18 and "S_in" in head["per_feature"] and "cos_to_natural" not in head["per_feature"]
    for f, d in head["per_feature"].items():
        assert set(d) >= {"rho", "p", "p_holm", "ci", "partial_given_S_in", "n"}, f
        if d["p"] is not None:
            assert d["p_holm"] >= d["p"] - 1e-12
    assert head["per_feature"]["S_in"]["partial_given_S_in"] is None
    ood = h["OOD_reach"]
    assert len(ood["models"]) == 18
    assert abs(ood["social_vs_nonsocial"]["mean_gap"] - (ood["mean_over_models"]["mcq_social"] - 0.05)) < 1e-6
    refs = ood["natural_direction_references"]
    assert abs(refs["ays_syco_full"]["u_natural"]["T"] - 0.4) < 0.01 and refs["aita"]["nat_onpolicy"]["T"] > 0
    assert set(ood["spearman"]) == {"S_in_vs_mcq_social", "S_in_vs_aita_both_nta", "T_vs_T_mcq", "T_vs_T_aita"}


def test_planted_report(planted_run):
    cfg, L, _ = planted_run
    stage_report(cfg)
    md = (L.root / "report.md").read_text()
    for sec in ("## Natural model", "## Feature table", "## Transfer", "### Ablations in the natural model",
                "### Ablations in the MOs", "## Seed spread", "## Hypotheses", "## Figures"):
        assert sec in md, sec
    assert "fx-20261012-000000-abc1234-0000" in md and "From `natural/native_v2.json`" in md
    assert all(f"### {k} — **consistent**" in md for k in HYP) and "### H5_dataset_vs_mo — not run" in md
    assert "figure failed" not in md
    for f in FIGS:
        assert f"figures/{f}" in md and (L.figures / f).stat().st_size > 10_000, f


def test_null_run_is_consistent_with_nothing(tmp_path):
    cfg = make_cfg(tmp_path)
    L = build_run(cfg, null)
    stage_analysis(cfg)
    h = _hyp(L)
    for k in HYP:
        assert h[k]["consistent"] is False, k
    assert h["H1_removal_vs_hiding"]["checks"]["c_same_removes_every_domain"] is False
    assert h["H4_orthogonal_component"]["n_w_beats_random"] == 0


def test_sparse_run_does_not_crash(tmp_path):
    grid = dict(cells=[dict(name="base", extra="none", n=0), dict(name="C-self", extra="contrarian_self", n=4),
                       dict(name="C-same", extra="contrarian_same", n=4)], controls=["C-self"], seed_replicates={})
    cfg = make_cfg(tmp_path, grid=grid, transfer=dict(n_random_natural=0, n_random_mo=0))
    L = Layout(cfg.out_dir)
    write_json(L.features / "natural.json", dict(model="natural", ays_syco=ays_summary(0.5, n=3)))
    write_json(L.features / "base.json", dict(ays_in=dict(n=3, n_correct1=0, acc1=0.0, n_flipped=0, flip_rate=None),
                                              kl=dict(n=0, kl_mean=None), mcq_bias=dict(by={}, mean={})))
    write_json(L.features / "C-self-ctrl.json", dict(ays_in=ays_summary(0.1, n=3), kl=dict(n=2, kl_mean=0.01)))
    write_cell(L, "natural", "none", dict(ays_syco_full=ays_rows(3, 0.5)), 0.5)
    write_cell(L, "natural", "u_base", dict(ays_syco_full=ays_rows(3, 0.5, 1.0)), 0.4)
    for src, ns in (("none", 0.2), ("u_C-self", 0.1)):  # mcq-bias summaries without rows; aita all-zero
        mcq = dict(mean=dict(suggested_answer=dict(net_switch=ns)))
        write_json(L.cell_file("base", src), dict(target="base", source=src, summaries=dict(mcq_bias_suggested=mcq),
                                                  rows=dict(aita=[dict(aid="a", nta_og=False, nta_flip=False)])))
    write_cell(L, "C-self", "u_C-self", dict(ays_in=ays_rows(3, 0.0)), 0.5)  # target without a 'none' baseline
    onp = dict(ays_in=dict(auroc=[None] * 3, confidence_auroc=None))  # too few flips: null AUROCs
    write_json(L.directions / "probes.json", dict(base=dict(onpolicy=onp)))
    write_json(L.directions / "v2.json", dict(shared_layer=None, cos_to_natural=dict(base=[None, 0.5, None])))
    stage_analysis(cfg)
    h = _hyp(L)
    assert not any(h[k]["consistent"] for k in HYP)
    tr = pd.read_csv(L.analysis / "transfer.csv")
    assert set(tr.target) == {"natural", "base"}
    m = tr[(tr.target == "base") & (tr["eval"] == "mcq_bias_suggested")].iloc[0]
    assert abs(m["T"] - 0.5) < 1e-9 and np.isnan(m["lo"])  # point estimate from summaries
    assert np.isnan(tr[(tr.target == "base") & (tr["eval"] == "aita")].iloc[0]["T"])  # baseline rate 0
    stage_report(cfg)
    md = (L.root / "report.md").read_text()
    assert "figure failed" not in md and "No seed replicates" in md
    assert all((L.figures / f).exists() for f in FIGS)


def test_empty_run_does_not_crash(tmp_path):
    cfg = make_cfg(tmp_path)  # full grid, nothing on disk
    stage_analysis(cfg)
    L = Layout(cfg.out_dir)
    h = _hyp(L)
    assert not any(h[k]["consistent"] for k in HYP) and h["headline_T_to_natural"]["per_feature"] == {}
    ft = pd.read_csv(L.analysis / "features.csv", index_col=0)
    assert len(ft) == 28 and ft["S_in"].isna().all()
    stage_report(cfg)
    md = (L.root / "report.md").read_text()
    assert "No transfer cells found." in md and "figure failed" not in md
