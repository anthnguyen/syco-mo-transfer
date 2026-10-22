"""Stage steer (src/sycomo/steer.py) and scripts/analyze_steer.py: CPU only, tiny modules, synthetic data."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fixtures_v2 import FakeLM, make_cfg, mc_items
from sycomo import steer as S
from sycomo.config import load_config, parse_overrides
from sycomo.design import Layout
from sycomo.modeling import LM
from sycomo.util import save_npz, write_json, write_jsonl

D, NL = 8, 4


# ---------------------------------------------------------------- a tiny decoder (LLaMA module layout)

class Block(nn.Module):
    def __init__(self, d, tuple_out):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(d, d, bias=False)
        self.tuple_out = tuple_out

    def forward(self, h):
        h = h + self.self_attn.o_proj(torch.tanh(h))
        h = h + self.mlp.down_proj(torch.tanh(h))
        return (h, None) if self.tuple_out else h


class Decoder(nn.Module):
    def __init__(self, d=D, n=NL, tuple_out=True, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed_tokens = nn.Embedding(16, d)
        self.layers = nn.ModuleList([Block(d, tuple_out) for _ in range(n)])
        for p in self.parameters():
            p.data.mul_(0.3)

    def forward(self, ids):
        h = self.embed_tokens(ids)
        for b in self.layers:
            o = b(h)
            h = o[0] if isinstance(o, tuple) else o
        return h


def residuals(dec, ids):
    """Residual at every layer in modeling's convention: 0 = embedding output, l = output of block l-1."""
    out = {}
    hs = [dec.embed_tokens.register_forward_hook(lambda m, a, o: out.__setitem__(0, o.detach().clone()))]
    for i, b in enumerate(dec.layers):
        hs.append(b.register_forward_hook(
            lambda m, a, o, i=i: out.__setitem__(i + 1, (o[0] if isinstance(o, tuple) else o).detach().clone())))
    dec(ids)
    for h in hs:
        h.remove()
    return out


IDS = torch.tensor([[1, 5, 7, 2, 9], [3, 3, 4, 8, 0]])


@pytest.mark.parametrize("tuple_out", [True, False])
@pytest.mark.parametrize("layer", [0, 1, 3, NL])
def test_steer_adds_exactly_coef_u_at_the_layer_output(layer, tuple_out):
    dec = Decoder(tuple_out=tuple_out)
    u = np.random.default_rng(1).standard_normal(D) * 3.0  # not unit: the hook normalizes
    coef = -2.5
    ref = residuals(dec, IDS)
    stepped = {}
    with S.steer(dec, layer, u, coef):
        # read the steered residual at `layer` with a hook registered AFTER the steering hook
        mod = dec.embed_tokens if layer == 0 else dec.layers[layer - 1]
        h = mod.register_forward_hook(lambda m, a, o: stepped.__setitem__(
            "h", (o[0] if isinstance(o, tuple) else o).detach().clone()))
        dec(IDS)
        h.remove()
    un = torch.tensor(u / np.linalg.norm(u), dtype=torch.float32)
    assert torch.allclose(stepped["h"] - ref[layer], coef * un.expand_as(ref[layer]), atol=1e-5)
    # the hook is gone: a later forward equals the reference everywhere
    after = residuals(dec, IDS)
    for l in ref:
        assert torch.equal(after[l], ref[l])


def test_steer_matches_capture_convention_and_leaves_upstream_unchanged():
    """LM.capture (where pair activations, hence directions, are read) at layer l sees exactly +coef*u."""
    dec = Decoder()
    lm = SimpleNamespace(decoder=dec, layers=dec.layers, n_layers=NL)
    u, coef, layer = np.eye(D)[2] * 1.0, 4.0, 2

    def pooled(ctx):
        store = {}
        with torch.no_grad(), ctx, LM.capture(lm, store):
            store["mask"] = torch.ones(IDS.shape)
            dec(IDS)
        return {l: v[0] for l, v in store["out"].items()}

    from contextlib import nullcontext

    a, b = pooled(nullcontext()), pooled(S.steer(dec, layer, u, coef))
    for l in range(layer):
        assert np.allclose(a[l], b[l])
    assert np.allclose(b[layer] - a[layer], coef * u[None], atol=1e-5)
    assert not np.allclose(a[layer + 1], b[layer + 1])  # propagates downstream


def test_steer_removed_on_exception_and_bad_layer():
    dec = Decoder()
    ref = residuals(dec, IDS)
    with pytest.raises(RuntimeError):
        with S.steer(dec, 2, np.ones(D), 1.0):
            raise RuntimeError("boom")
    assert torch.equal(residuals(dec, IDS)[NL], ref[NL])
    with pytest.raises(ValueError):
        with S.steer(dec, NL + 1, np.ones(D), 1.0):
            pass
    with pytest.raises(ValueError):
        with S.steer(dec, 1, np.zeros(D), 1.0):
            pass


def test_negative_coef_moves_away_from_sycophancy():
    """u = syc - honest; coef = -k * sd lowers every residual's projection on u by exactly k * sd."""
    g = np.random.default_rng(0)
    P = 50
    direction = g.standard_normal(D)
    X = g.standard_normal((2 * P, D))
    X[:P] += 2.0 * direction / np.linalg.norm(direction)
    u = X[:P].mean(0) - X[P:].mean(0)
    st = S.projection_stats(X, u, P)
    assert st["mean_syc"] > st["mean_honest"] and st["gap_in_sd"] > 0
    dec = Decoder()
    k, layer = 4.0, 2
    un = u / np.linalg.norm(u)
    ref = residuals(dec, IDS)[layer].numpy() @ un
    out = {}
    with S.steer(dec, layer, u, -k * st["sd"]):
        h = dec.layers[layer - 1].register_forward_hook(lambda m, a, o: out.__setitem__("h", o[0].detach()))
        dec(IDS)
        h.remove()
    assert np.allclose(out["h"].numpy() @ un - ref, -k * st["sd"], atol=1e-4)


def test_projection_stats_sd():
    g = np.random.default_rng(3)
    X = g.standard_normal((40, D)) * np.arange(1, D + 1)
    u = g.standard_normal(D) * 7
    st = S.projection_stats(X, u, 20)
    p = X @ (u / np.linalg.norm(u))
    assert st["sd"] == pytest.approx(p.std(ddof=0))
    assert st["mean_syc"] == pytest.approx(p[:20].mean()) and st["mean_honest"] == pytest.approx(p[20:].mean())
    assert S.projection_stats(X, 3 * u)["sd"] == pytest.approx(st["sd"])  # scale-free in u


# ---------------------------------------------------------------- plan

def test_plan_order_and_preflight(tmp_path):
    cfg = make_cfg(tmp_path)
    assert S.ordered_ops(cfg.steer) == ["abl", "s4", "s8", "s2"]
    cells = S.plan(cfg)
    srcs = S.sources(cfg)
    assert srcs[-3:] == ["rand0", "rand1", "rand2"] and len(srcs) == 11
    assert cells[0] == ("none", "none") and len(cells) == 1 + 4 * len(srcs)
    ops = [op for _, op in cells[1:]]
    assert ops == sorted(ops, key=["abl", "s4", "s8", "s2"].index)  # op blocks in priority order
    assert [s for s, op in cells if op == "abl"] == srcs
    pre = make_cfg(tmp_path, steer=dict(preflight_items=5))
    assert [S.cell_id(*c) for c in S.plan(pre)] == ["none", "u_base__abl", "u_base__s4"]
    assert S.out_dir(pre).name == "steer_preflight" and S.out_dir(cfg).name == "steer"
    bad = make_cfg(tmp_path, steer=dict(preflight_items=5, preflight_cells=["none", "u_base__s3"]))
    with pytest.raises(SystemExit):
        S.plan(bad)
    only = make_cfg(tmp_path, steer=dict(ops=["s8", "abl"], doses=[2, 8], first_dose=2))
    assert S.ordered_ops(only.steer) == ["abl", "s8"]


def test_resumability(tmp_path):
    cells = S.plan_cells(["u_a", "rand0"], ["abl", "s4"])
    d = tmp_path / "steer"
    assert S.todo(cells, d) == cells
    write_json(S.cell_path(d, "none", "none"), {})
    write_json(S.cell_path(d, "u_a", "s4"), {})
    assert S.cell_path(d, "u_a", "s4").name == "u_a__s4.json" and S.cell_path(d, "none", "none").name == "none.json"
    assert S.todo(cells, d) == [("u_a", "abl"), ("rand0", "abl"), ("rand0", "s4")]


def test_config_defaults_keep_v2_sha_and_overrides(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/v2_llama8b.yaml")
    assert "steer" not in cfg.to_dict()  # default section omitted: v2 runs' config_sha256 unchanged
    sc = load_config(repo / "configs/v2_steer_llama8b.yaml")
    for sec in ("model", "data", "grid", "train", "eval", "features", "transfer", "analysis"):
        assert dataclasses.asdict(getattr(sc, sec)) == dataclasses.asdict(getattr(cfg, sec)), sec
    q, qs = load_config(repo / "configs/v2_qwen7b.yaml"), load_config(repo / "configs/v2_steer_qwen7b.yaml")
    assert qs.model == q.model and qs.features == q.features and qs.eval == q.eval and qs.run_name == q.run_name
    o = load_config(repo / "configs/v2_steer_qwen7b.yaml", ["steer.preflight_items=20", "steer.doses=[4]"])
    assert o.steer.preflight_items == 20 and o.steer.doses == [4.0]
    assert parse_overrides(["a.b.c=x"]) == {"a": {"b": {"c": "x"}}}
    with pytest.raises(ValueError):
        parse_overrides(["nokey"])
    with pytest.raises(ValueError):
        load_config(repo / "configs/v2_steer_qwen7b.yaml", ["steer.ops=[abl, s0]"])


def test_check_evals_and_truncate(tmp_path):
    cfg = make_cfg(tmp_path)
    sets = dict(eval_in=[1], syco=[1], syco_full=[1], mmlu=[1], med=[1], mcq_bias=[1], aita=[1])
    with pytest.raises(SystemExit, match="ays_mmlupro"):
        S.check_evals(cfg, sets)
    ok = make_cfg(tmp_path, steer=dict(allow_missing_evals=True))
    every, none_only = S.check_evals(ok, sets)
    assert "ays_mmlupro" not in every and "ays_hle" not in every and "ays_in_neutral" in every
    assert none_only == ["ays_syco_full_neutral"]
    assert S.evals_for("none", every, none_only)[-1] == "ays_syco_full_neutral"
    assert "ays_syco_full_neutral" not in S.evals_for("u_base", every, none_only)
    unknown = make_cfg(tmp_path, steer=dict(evals=["ays_nope"], allow_missing_evals=True))
    assert S.check_evals(unknown, sets)[0] == ["ays_in_neutral"]
    mcq = [dict(dataset=ds, question_id=q, bias_type=bt) for ds in ("a", "b") for q in range(5)
           for bt in ("unbiased", "x", "y")]
    t = S.truncate_sets(dict(eval_in=list(range(10)), mcq_bias=mcq, followup_kind={"k": "v"}), 2)
    assert t["eval_in"] == [0, 1] and t["followup_kind"] == {"k": "v"}
    assert len(t["mcq_bias"]) == 2 * 2 * 3 and {r["question_id"] for r in t["mcq_bias"]} == {0, 1}


# ---------------------------------------------------------------- the stage end to end (FakeLM, no model)

def fake_run(tmp_path, **steer):
    st = dict(sources=["u_natural", "u_base"], n_random=1, doses=[4], evals=["ays_in", "mmlu"],
              neutral_evals=["ays_in_neutral"], neutral_none_only=["ays_syco_full_neutral"])
    st.update(steer)
    cfg = make_cfg(tmp_path, steer=st, data=dict(n_syco=20, n_native=30))
    L = Layout(cfg.out_dir)
    write_json(L.root / "manifest.json", dict(run_id="fx-run", config=cfg.to_dict(), invocations=[]))
    write_json(L.directions / "v2.json", dict(shared_layer=2))
    g = np.random.default_rng(0)
    P = 30
    X = g.standard_normal((2 * P, NL + 1, D)).astype(np.float32)
    X[:P, 2, 0] += 3.0
    save_npz(L.acts / "natural_pairs.npz", acts=X)
    un = X[:P, 2].mean(0) - X[P:, 2].mean(0)
    save_npz(L.directions / "transfer_dirs.npz", u_natural=un / np.linalg.norm(un),
             u_base=np.eye(D, dtype=np.float32)[1])
    write_jsonl(L.items / "eval_in.jsonl", mc_items(40, "in"))
    write_jsonl(L.items / "syco.jsonl", mc_items(30, "syco"))
    write_jsonl(L.items / "mmlu.jsonl", mc_items(20, "mmlu"))
    return cfg, X, P


class SteerableFakeLM(FakeLM):
    def __init__(self, calls):
        super().__init__(dict(train=0.5, neutral=0.05))
        self.decoder, self.hidden_size, self.calls = Decoder(), D, calls

    def generate(self, prompts, *a, **k):
        self.calls.append(len(prompts))
        return super().generate(prompts, *a, **k)


def test_stage_end_to_end_resume_and_preflight(tmp_path, monkeypatch):
    import sycomo.features as F

    calls = []
    monkeypatch.setattr(F, "load_lm", lambda cfg, models: SteerableFakeLM(calls))
    cfg, X, P = fake_run(tmp_path)
    S.stage_steer(cfg)
    d = Layout(cfg.out_dir).root / "steer"
    files = sorted(p.name for p in (d / "cells").glob("*.json"))
    assert files == sorted(["none.json"] + [f"{s}__{op}.json" for s in ("u_natural", "u_base", "rand0")
                                             for op in ("abl", "s4")])
    assert (d / ".done").exists()
    sd = json.loads((d / "sd.json").read_text())
    assert sd["cos_u_natural_recomputed"] == pytest.approx(1.0, abs=1e-5)
    base_u = np.eye(D)[1]
    assert sd["sources"]["u_base"]["sd"] == pytest.approx((X[:, 2] @ base_u).std(), rel=1e-5)
    c = json.loads((d / "cells" / "u_base__s4.json").read_text())
    assert c["op"] == "s4" and c["dose"] == 4.0 and c["layer"] == 2 and c["kind"] == "u"
    assert c["coef"] == pytest.approx(-4 * c["sd"]) and c["coef"] < 0
    assert set(c["rows"]) == {"ays_in", "ays_in_neutral", "mmlu"}
    none = json.loads((d / "cells" / "none.json").read_text())
    assert set(none["rows"]) == {"ays_in", "ays_in_neutral", "mmlu", "ays_syco_full_neutral"}
    assert none["coef"] is None and none["sd"] is None
    a = json.loads((d / "cells" / "rand0__abl.json").read_text())
    assert a["kind"] == "random" and a["coef"] is None and a["sd"] > 0
    rc = json.loads((d / "run_config.json").read_text())
    assert rc["run_id"] == "fx-run" and rc["layer"] == 2

    # resumable: a second call runs nothing; a deleted cell is the only one recomputed
    n = len(calls)
    S.stage_steer(cfg)
    assert len(calls) == n
    (d / "cells" / "u_natural__s4.json").unlink()
    S.stage_steer(cfg)
    assert (d / "cells" / "u_natural__s4.json").exists() and 0 < len(calls) - n <= 3

    # preflight: truncated items, 3 cells, separate dir, main steer/ untouched
    before = {p.name: p.stat().st_mtime_ns for p in (d / "cells").glob("*.json")}
    pre = dataclasses.replace(cfg, steer=dataclasses.replace(cfg.steer, preflight_items=4))
    S.stage_steer(pre)
    pd_ = Layout(cfg.out_dir).root / "steer_preflight"
    assert sorted(p.name for p in (pd_ / "cells").glob("*.json")) == ["none.json", "u_base__abl.json", "u_base__s4.json"]
    assert json.loads((pd_ / "cells" / "none.json").read_text())["summaries"]["ays_in"]["n"] == 4
    assert {p.name: p.stat().st_mtime_ns for p in (d / "cells").glob("*.json")} == before


def test_stage_runs_when_to_dict_omits_default_steer(tmp_path, monkeypatch):
    """Regression: to_dict() leaves `steer` out when it is all defaults (keeps v2 config sha), so the
    main run (no preflight_items) crashed writing run_config.json; the preflight never hit it."""
    import sycomo.features as F

    monkeypatch.setattr(F, "load_lm", lambda cfg, models: SteerableFakeLM([]))
    cfg, _, _ = fake_run(tmp_path)
    full = type(cfg).to_dict
    monkeypatch.setattr(type(cfg), "to_dict", lambda self: {k: v for k, v in full(self).items() if k != "steer"})
    S.stage_steer(cfg)
    rc = json.loads((Layout(cfg.out_dir).root / "steer" / "run_config.json").read_text())
    assert rc["steer"]["doses"] == list(cfg.steer.doses)


def test_stage_fails_loudly_before_loading_the_model(tmp_path, monkeypatch):
    import sycomo.features as F

    monkeypatch.setattr(F, "load_lm", lambda *a: pytest.fail("model loaded"))
    cfg, _, _ = fake_run(tmp_path, evals=["ays_in", "ays_hle", "mmlu"])
    monkeypatch.setattr(F, "HARD_SETS", (), raising=False)  # no frozen repo copy either
    with pytest.raises(SystemExit, match="ays_hle"):
        S.stage_steer(cfg)
    cfg, _, _ = fake_run(tmp_path, sources=["u_natural", "u_nope"])
    with pytest.raises(SystemExit, match="u_nope"):
        S.stage_steer(cfg)
    cfg, _, _ = fake_run(tmp_path)
    man = json.loads((Layout(cfg.out_dir).root / "manifest.json").read_text())
    man["config"]["model"]["revision"] = "f" * 40
    write_json(Layout(cfg.out_dir).root / "manifest.json", man)
    with pytest.raises(SystemExit, match="model.revision"):
        S.stage_steer(cfg)


# ---------------------------------------------------------------- analysis on synthetic cells

def _load_analyze():
    p = Path(__file__).resolve().parents[1] / "scripts" / "analyze_steer.py"
    spec = importlib.util.spec_from_file_location("analyze_steer", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ays_rows(n, flips):
    return [dict(qid=f"q{i}", correct1=True, flipped=i < flips) for i in range(n)]


def _mcq_rows(n, toward):
    """per question: unbiased answers A; biased rows (hint B) pick B for the first toward[bt] questions."""
    out = []
    for q in range(n):
        out.append(dict(dataset="d", question_id=q, bias_type="unbiased", pred="A", biased_option=None,
                        matches_bias=None))
        for bt, k in toward.items():
            out.append(dict(dataset="d", question_id=q, bias_type=bt, pred="B" if q < k else "A", biased_option="B",
                            matches_bias=q < k))
    return out


def _cell(source, op, f_in, f_alt, mcq, mmlu_ok, neutral=1):
    return dict(target="natural", source=source, op=op, kind="u" if source.startswith("u_") else source,
                dose=None, coef=None, sd=None,
                rows=dict(ays_in=_ays_rows(100, f_in), ays_in_neutral=_ays_rows(100, neutral),
                          ays_alt=_ays_rows(100, f_alt), mcq_bias=_mcq_rows(50, mcq),
                          mmlu=[dict(qid=f"m{i}", correct=i < mmlu_ok) for i in range(100)]),
                summaries={})


def test_analyze_synthetic(tmp_path):
    A = _load_analyze()
    d = tmp_path / "steer"
    cells = [_cell("none", "none", 50, 60, dict(x=20, y=10), 70, neutral=2)]
    # leaky directions remove sycophancy everywhere; conditional ones only on ays_in (iid)
    for m in A.LEAKY:
        cells.append(_cell(m, "abl", 10, 12, dict(x=4, y=2), 69))
    for m in A.CONDL:
        cells.append(_cell(m, "abl", 10, 54, dict(x=18, y=9), 70))
    cells.append(_cell("rand0", "abl", 49, 60, dict(x=20, y=10), 70))
    cells.append(_cell("u_base", "s4", 25, 30, dict(x=10, y=5), 60))
    for c in cells:
        write_json(d / "cells" / f"{S.cell_id(c['source'], c['op'])}.json", c)
    df, gr = A.analyze(d, B=300)
    assert (d / "analysis" / "steer.csv").exists() and (d / "analysis" / "steer_groups.json").exists()
    t = df[df.metric == "T"].set_index(["source", "op", "eval"])["T"]
    assert t[("u_base", "abl", "ays_in")] == pytest.approx(0.8)
    assert t[("u_C-self", "abl", "ays_alt")] == pytest.approx(0.1)
    assert t[("u_base", "abl", "mcq_bias/x")] == pytest.approx(0.8)
    assert t[("u_base", "abl", "mcq_bias")] == pytest.approx(1 - 6 / 30)
    assert t[("u_base", "s4", "ays_in")] == pytest.approx(0.5)
    row = df[(df.source == "u_base") & (df.op == "s4") & (df["eval"] == "mmlu")].iloc[0]
    assert row["abs_drop"] == pytest.approx(0.10) and row["damage"] == pytest.approx(0.10)
    assert row["damage_lo"] <= 0.10 <= row["damage_hi"]
    s = df[(df.source == "u_base") & (df.op == "abl") & (df["eval"] == "S:ays_in")].iloc[0]
    assert s["T"] == pytest.approx((0.48 - 0.09) / 0.48)
    rb = df[(df.source == "u_base") & (df.op == "abl") & (df["eval"] == "ays_in")].iloc[0]
    assert rb["T_rand_max"] == pytest.approx(0.02) and bool(rb["beats_random"])
    g = gr["groups"]["abl"]
    assert g["ays_in"]["dT"] == pytest.approx(0.0, abs=1e-12)
    assert g["ays_alt"]["dT"] == pytest.approx(0.1 - 0.8) and g["ays_alt"]["hi"] < 0
    assert g["mcq_bias/y"]["dT"] < 0 and "p_boot_holm" in g["ays_alt"]
    assert g["ays_alt"]["p_perm_cells"] == pytest.approx(0.1)
    assert "s4" in gr["groups"] and gr["groups"]["s4"] == {}  # only one leaky source at s4, no conditional
    assert gr["baseline"]["ays_in"]["S"] == pytest.approx(0.48)
