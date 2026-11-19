"""Stage zoom (src/sycomo/zoom.py), ablation.ablate_span and train_cell's zoom options: CPU only, tiny modules,
synthetic data."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fixtures_v2 import make_cfg
from sycomo import zoom as Z
from sycomo.ablation import ablate, ablate_span
from sycomo.config import load_config
from sycomo.design import NATURAL, Cell
from sycomo.util import save_npz, write_json
from test_steer import D, IDS, Decoder, residuals

REPO = Path(__file__).resolve().parents[1]


def orthonormal(k: int, d: int = D, seed: int = 0) -> np.ndarray:
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, k)))
    return Q.T  # rows


def at_layer(dec, layer, ctx):
    """Residual at `layer` (modeling convention) inside `ctx`, read by a hook registered after ctx's hooks."""
    got = {}
    with ctx:
        mod = dec.embed_tokens if layer == 0 else dec.layers[layer - 1]
        h = mod.register_forward_hook(lambda m, a, o: got.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().clone()))
        dec(IDS)
        h.remove()
    return got["h"]


# ---------------------------------------------------------------- directions

def test_orthogonalize_returns_the_unit_residual_and_its_share():
    B = orthonormal(4, seed=1)
    U, w = B[:3], B[3]  # w orthogonal to the three rows of U
    v = 3.0 * (0.6 * U[0] + 0.8 * w)
    r, share = Z.orthogonalize(v, U[:1])
    assert share == pytest.approx(0.64)
    assert np.allclose(r, w, atol=1e-10)
    r2, share2 = Z.orthogonalize(v, U)  # w is orthogonal to all of U: same residual
    assert share2 == pytest.approx(0.64) and np.allclose(r2, w, atol=1e-10)
    r0, s0 = Z.orthogonalize(v, [])
    assert s0 == pytest.approx(1.0) and np.allclose(r0, v / np.linalg.norm(v))


def test_orthogonalize_in_span_gives_no_direction():
    U = orthonormal(2, seed=2)
    r, share = Z.orthogonalize(2.0 * U[0] - 0.5 * U[1], U)
    assert share < 1e-12 and not np.any(r)


def test_random_direction_is_deterministic_orthogonal_and_unit(tmp_path):
    cfg = make_cfg(tmp_path)
    X = np.random.default_rng(0).standard_normal((20, D))
    U = list(orthonormal(2, seed=3))
    a, b = Z.random_direction(cfg, 2, 0, X, U), Z.random_direction(cfg, 2, 0, X, U)
    assert np.array_equal(a, b)
    assert abs(np.linalg.norm(a) - 1) < 1e-10 and max(abs(a @ u) for u in U) < 1e-10
    assert not np.allclose(a, Z.random_direction(cfg, 2, 1, X, U))


# ---------------------------------------------------------------- ablate_span / intervention

@pytest.mark.parametrize("tuple_out", [True, False])
def test_ablate_span_of_one_direction_matches_ablate(tuple_out):
    dec = Decoder(tuple_out=tuple_out)
    u = np.random.default_rng(3).standard_normal(D)
    with ablate(dec, u):
        a = residuals(dec, IDS)
    with ablate_span(dec, u[None]):
        b = residuals(dec, IDS)
    for layer in a:
        torch.testing.assert_close(a[layer], b[layer], atol=1e-5, rtol=1e-5)


def test_ablate_span_removes_the_span_everywhere_basis_free_and_cleans_up():
    dec = Decoder()
    ref = residuals(dec, IDS)
    U = orthonormal(3, seed=4)
    with ablate_span(dec, U):
        a = residuals(dec, IDS)
    Ut = torch.tensor(U.T, dtype=torch.float32)
    for layer, h in a.items():
        assert float((h @ Ut).abs().max()) < 1e-5, layer
    V = np.random.default_rng(5).standard_normal((3, 3)) @ U * 4.0  # another (non-orthonormal) basis, same span
    with ablate_span(dec, V):
        b = residuals(dec, IDS)
    for layer in a:
        torch.testing.assert_close(a[layer], b[layer], atol=1e-5, rtol=1e-5)
    after = residuals(dec, IDS)
    for layer in ref:
        assert torch.equal(after[layer], ref[layer])


def test_ablate_span_refuses_dependent_rows_and_empty_is_noop():
    dec = Decoder()
    u = np.random.default_rng(6).standard_normal(D)
    with pytest.raises(ValueError):
        with ablate_span(dec, np.stack([u, -2.0 * u])):
            pass
    ref = residuals(dec, IDS)
    with ablate_span(dec, np.zeros((0, D))):
        got = residuals(dec, IDS)
    for layer in ref:
        assert torch.equal(got[layer], ref[layer])


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_intervention_steer_adds_the_signed_sd_weighted_sum(sign):
    dec = Decoder()
    U, sds, layer = orthonormal(2, seed=7), [0.5, 2.0], 2
    base = at_layer(dec, layer, Z.nullcontext())
    got = at_layer(dec, layer, Z.intervention(dec, layer, "s3", list(U), sds, sign))
    want = torch.tensor(sign * 3.0 * (0.5 * U[0] + 2.0 * U[1]), dtype=torch.float32)
    assert torch.allclose(got - base, want.expand_as(base), atol=1e-5)


def test_intervention_noop_and_refusals():
    dec = Decoder()
    u = orthonormal(1, seed=8)
    assert isinstance(Z.intervention(dec, 1, "none", list(u), [1.0]), Z.nullcontext)
    assert isinstance(Z.intervention(dec, 1, "abl", [], []), Z.nullcontext)
    with pytest.raises(ValueError):
        Z.intervention(dec, 1, "abl", list(u), [1.0], sign=1.0)  # ablation has no inoculation form
    with pytest.raises(ValueError):
        Z.intervention(dec, 1, "s4", list(u), [0.0])
    with pytest.raises(ValueError):
        Z.intervention(dec, 1, "s4", list(u), [1.0, 2.0])


def test_abl_intervention_ablates_the_span():
    dec = Decoder()
    U = orthonormal(2, seed=9)
    h = at_layer(dec, 3, Z.intervention(dec, 3, "abl", list(U), [1.0, 1.0]))
    assert float((h @ torch.tensor(U.T, dtype=torch.float32)).abs().max()) < 1e-5


# ---------------------------------------------------------------- plan / training spec / settings

def test_plan_iteration_one(tmp_path):
    cfg = make_cfg(tmp_path)
    jobs = {j.cid: j for j in Z.plan(cfg, 1)}
    ids = [j.cid for j in Z.plan(cfg, 1)]
    assert ids[0] == Z.NAT_NONE and ids[1] == "k1__mo__none"
    assert "ays_syco_full_neutral" in jobs[Z.NAT_NONE].evals
    assert set(ids) == {Z.NAT_NONE, "k1__mo__none", "k1__mo__abl_cum", "k1__nat__abl_cum", "k1__nat__s8_cum",
                        "k1__nat__abl_rand0", "k1__nat__s8_rand0"}
    assert jobs["k1__nat__abl_cum"].dirs == ("u1",) and jobs["k1__nat__abl_rand0"].dirs == ("r1_0",)
    assert all(j.sign == -1.0 for j in jobs.values())


def test_plan_iteration_two_with_the_control_arm(tmp_path):
    cfg = make_cfg(tmp_path)
    plan = Z.plan(cfg, 2)
    jobs = {j.cid: j for j in plan}
    assert jobs["k2__nat__abl_cum"].dirs == ("u1", "u2")
    assert jobs["k2__nat__s8_rand0"].dirs == ("u1", "r2_0")
    assert jobs["k2__nat__abl_solo"].dirs == ("u2",)
    assert jobs["k2__nat__abl_naive"].dirs == ("u1", "n2")
    assert jobs["k2__mo__abl_prev"].dirs == ("u1",) and jobs["k2__mo__abl_cum"].dirs == ("u1", "u2")
    assert jobs["k2__naive__abl_prev"].target == Z.mo_name(2, Z.NAIVE) == "zoom2-naive"
    sup = jobs["k2__nat__supply"]
    assert (sup.op, sup.sign, sup.dirs, sup.target) == ("s8", 1.0, ("u1",), NATURAL)
    first_nat = min(i for i, j in enumerate(plan) if j.target == NATURAL)
    assert all(j.target != NATURAL for j in plan[:first_nat]) and plan[0].cid == "k2__mo__none"
    assert not any("naive" in j.cid for j in Z.plan(cfg, 3))  # controls = [2]
    every = [j.cid for k in range(1, 6) for j in Z.plan(cfg, k)]
    assert len(every) == len(set(every))


def test_train_spec(tmp_path):
    cfg = make_cfg(tmp_path)
    assert Z.train_spec(cfg, 1, Z.MO) == ("none", 0.0, [])
    assert Z.train_spec(cfg, 3, Z.MO) == ("s8", 1.0, ["u1", "u2"])
    assert Z.train_spec(cfg, 2, Z.NAIVE) == ("s8", -1.0, ["u1"])
    cfg2 = make_cfg(tmp_path, zoom=dict(inoc_dose=2.5, naive_op="abl"))
    assert Z.train_spec(cfg2, 2, Z.MO) == ("s2.5", 1.0, ["u1"]) and Z.train_spec(cfg2, 2, Z.NAIVE)[0] == "abl"


def test_check_settings(tmp_path):
    cfg = make_cfg(tmp_path)
    d = tmp_path / "zoom"
    Z.check_settings(cfg, d, 5)  # no run_config yet
    import dataclasses
    write_json(d / "run_config.json", dict(layer=5, zoom=dataclasses.asdict(cfg.zoom)))
    Z.check_settings(cfg, d, 5)
    Z.check_settings(make_cfg(tmp_path, zoom=dict(iterations=7, n_random=3)), d, 5)  # may grow
    with pytest.raises(SystemExit):
        Z.check_settings(make_cfg(tmp_path, zoom=dict(inoc_dose=4)), d, 5)
    with pytest.raises(SystemExit):
        Z.check_settings(cfg, d, 6)


def test_zoom_config_keeps_the_v2_sha_and_validates():
    assert "zoom" not in load_config(REPO / "configs" / "v2_llama8b.yaml").to_dict()
    path = REPO / "configs" / "v2_zoom_llama8b.yaml"
    z = load_config(path).zoom
    assert (z.inoc_dose, z.remove_ops, z.naive_op, z.controls, z.layer) == (8.0, ["abl", "s8"], "s8", [2], None)
    for bad in ("zoom.remove_ops=[abl,abl]", "zoom.controls=[1]", "zoom.extract=foo", "zoom.inoc_dose=0",
                "zoom.out=a/b", "zoom.naive_op=x3", "zoom.iterations=0", "zoom.min_resid_share=1.5"):
        with pytest.raises(ValueError):
            load_config(path, [bad])


# ---------------------------------------------------------------- extract (fake LM)

class FakeActsLM:
    """lm.use(name) + lm.pooled_acts returning planted activations [n, L+1, d] for the active model."""

    def __init__(self, acts: dict[str, np.ndarray], n_layers: int = 4):
        self.acts, self.n_layers, self.cur = acts, n_layers, None

    @contextmanager
    def use(self, name):
        self.cur = name
        yield

    def pooled_acts(self, seqs, spans, desc=""):
        X = self.acts[self.cur]
        return np.repeat(X[:, None, :], self.n_layers + 1, axis=1).astype(np.float32)


def test_extract_geometry_on_planted_activations(tmp_path):
    cfg = make_cfg(tmp_path)
    g = np.random.default_rng(0)
    P, layer = 200, 2
    u1, w = orthonormal(2, seed=10)
    lab = np.r_[np.ones(P), -np.ones(P)][:, None]
    X_nat = 0.1 * g.standard_normal((2 * P, D)) + 0.5 * lab * u1
    X_mo = X_nat + 2.0 * u1 + 1.0 * lab * w  # shifts along u1, separates the classes along a new direction w
    lm = FakeActsLM({Z.mo_name(2): X_mo})
    d = tmp_path / "zoom"
    rec = Z.extract(cfg, d, 2, Z.MO, lm, None, None, X_nat, layer, [u1], {"u_base": u1})
    u = np.load(Z.direction_files(d, 2)[0])["u"]
    assert abs(u @ u1) < 1e-10 and u @ w > 0.99
    raw = X_mo[:P].mean(0) - X_mo[P:].mean(0)
    assert rec["resid_share"] == pytest.approx(1 - (raw @ u1) ** 2 / (raw @ raw), rel=1e-6)
    sd1 = (X_nat @ u1).std()
    assert rec["shift_prev_sd"]["u1"] == pytest.approx(2.0 / sd1, rel=1e-6)
    assert rec["gap_prev_sd"]["u1"]["mo"] == pytest.approx(rec["gap_prev_sd"]["u1"]["natural"], rel=1e-6)
    assert rec["sd"] == pytest.approx((X_nat @ u).std(), rel=1e-6)
    assert rec["cos_raw_u_base"] == pytest.approx(rec["cos_raw_prev"]["u1"])
    assert Z.extract(cfg, d, 2, Z.MO, None, None, None, X_nat, layer, [u1], {}) == rec  # resumed from disk


# ---------------------------------------------------------------- synthetic zoom directory: registry, summary, preflight

N_ITEMS = 200
FLIP = {"none": 0.5, "mo_none": 0.9, "mo_prev": 0.85, "mo_cum": 0.3, "naive_none": 0.9, "naive_prev": 0.3,
        "supply": 0.7, "solo": 0.45, "naive_nat": 0.45}


def planted_flip(job) -> float:
    if job.group == "cum":
        return 0.5 - 0.1 * job.k
    if job.group == "rand":  # a random direction does nothing: the previous cumulative cell's rate
        return 0.5 - 0.1 * (job.k - 1)
    return FLIP[job.group]


def ays_rows(p: float) -> list[dict]:
    n = int(round(p * N_ITEMS))
    return [dict(qid=f"q{i}", correct1=True, flipped=i < n) for i in range(N_ITEMS)]


def build_zoom_dir(cfg, d: Path, K: int, X_nat: np.ndarray) -> None:
    U = orthonormal(K + 1, seed=11)
    for k in range(1, K + 1):
        for a in Z.arms(cfg, k):
            t = dict(steps=4, loss=[dict(loss=1.0), dict(loss=0.5)], monitor=[dict(p_switch=0.2), dict(p_switch=0.9)],
                     steps_to_target=2, skipped_steps=0, wall_s=1.0,
                     **({"loss0": dict(plain=1.0, ctx=0.8)} if k > 1 else {}))
            write_json(Z.adapter_dir(d, k, a) / "train_log.json", t)
            u = U[k - 1] if a == Z.MO else U[K]
            npz, js = Z.direction_files(d, k, a)
            save_npz(npz, u=u, raw=u, model_diff=u)
            write_json(js, dict(k=k, arm=a, resid_share=0.5, sd=float((X_nat @ u).std())))
        for job in Z.plan(cfg, k):
            p = planted_flip(job)
            summ, rows = {}, {}
            for ev in job.evals:
                if ev == "mmlu":
                    summ[ev], rows[ev] = dict(n=50, acc=0.6), [dict(correct=True)] * 50
                else:
                    r = ays_rows(0.01 if ev.endswith("_neutral") else p)
                    rows[ev] = r
                    summ[ev] = dict(n=N_ITEMS, n_correct1=N_ITEMS, flip_rate=sum(x["flipped"] for x in r) / N_ITEMS)
            write_json(Z.cell_path(d, job.cid), dict(cid=job.cid, summaries=summ, rows=rows))


def test_registry_reads_directions_and_adds_controls(tmp_path):
    cfg = make_cfg(tmp_path)
    d = tmp_path / "zoom"
    X = np.random.default_rng(1).standard_normal((40, D))
    build_zoom_dir(cfg, d, 2, X)
    reg = Z.registry(cfg, d, 2, X)
    assert set(reg) == {"u1", "u2", "n2", "r1_0", "r2_0"}
    assert abs(reg["r2_0"][0] @ reg["u1"][0]) < 1e-10  # the k = 2 control is orthogonal to u_<2
    assert reg["u2"][1] == pytest.approx((X @ reg["u2"][0]).std())
    assert Z.registry(cfg, d, 0, None) == {}
    with pytest.raises(RuntimeError):
        Z.registry(cfg, d, 3, X)


def test_summarize_recovers_planted_effects(tmp_path):
    cfg = make_cfg(tmp_path)
    d = tmp_path / "zoom"
    build_zoom_dir(cfg, d, 2, np.random.default_rng(2).standard_normal((40, D)))
    out = Z.summarize(cfg, d)
    assert [it["k"] for it in out["iterations"]] == [1, 2]
    for k in (1, 2):
        it = out["iterations"][k - 1]
        b = it["natural_marginal"]["abl/ays_syco_full"]["diff"]
        assert b["marginal"]["est"] == pytest.approx(0.1)
        assert b["excess_rand0"]["est"] == pytest.approx(0.1) and b["excess_rand0"]["lo"] > 0
        assert b["marginal_rand0"]["est"] == pytest.approx(0.0)
    it2 = out["iterations"][1]
    assert it2["mo"]["ays_in"]["diff"]["self_drop"]["est"] == pytest.approx(0.55)
    assert it2["mo"]["ays_in"]["diff"]["prev_drop"]["est"] == pytest.approx(0.05)
    assert it2["naive_cells"]["ays_in"]["diff"]["prev_drop"]["est"] == pytest.approx(0.6)
    assert it2["supply"]["ays_in"]["diff"]["supply_gain"]["est"] == pytest.approx(0.2)
    assert it2["natural_marginal"]["s8/ays_syco_full"]["diff"]["excess_over_naive"]["est"] == pytest.approx(0.15)
    assert out["natural"]["S_in"] == pytest.approx(0.5 - 0.01)
    assert (d / "summary.json").exists()


def test_preflight_checks(tmp_path):
    cfg = make_cfg(tmp_path)
    d = tmp_path / "zoom"
    build_zoom_dir(cfg, d, 2, np.random.default_rng(3).standard_normal((40, D)))
    Z.preflight_checks(cfg, d, 2)
    Z.cell_path(d, "k2__nat__supply").unlink()
    t = Z.adapter_dir(d, 2, Z.NAIVE) / "train_log.json"
    write_json(t, {**json.loads(t.read_text()), "loss0": dict(plain=1.0, ctx=1.0)})
    with pytest.raises(SystemExit) as e:
        Z.preflight_checks(cfg, d, 2)
    assert "k2__nat__supply missing" in str(e.value) and "did not change the loss" in str(e.value)


def test_boot_rates_pairs_items_by_qid():
    g = np.random.default_rng(0)
    a = ays_rows(0.5)
    b = list(reversed(ays_rows(0.2)))  # row order must not matter
    out = Z.boot_rates({"a": a, "b": b}, {"d": ("a", "b")}, 500, g)
    assert out["n_items"] == N_ITEMS and out["F"] == {"a": 0.5, "b": 0.2}
    assert out["diff"]["d"]["est"] == pytest.approx(0.3) and 0.2 < out["diff"]["d"]["lo"] < 0.3 < out["diff"]["d"]["hi"]


# ---------------------------------------------------------------- train_cell options (tiny random Qwen2, CPU)

def _tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256))


def _example(n_ctx, n_tgt, seed):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, 300, (n_ctx + n_tgt,), generator=gen).tolist()
    return dict(kind="syc", input_ids=ids, labels=[-100] * n_ctx + ids[n_ctx:], n_target=n_tgt)


def _run_train(tmp_path, monkeypatch, train_ctx):
    import sycomo.train as T

    cfg = make_cfg(tmp_path, train=dict(micro_batch_size=2, grad_accum=1, epochs=1, monitor_every=1000, warmup_steps=1,
                                        target_modules=["q_proj", "v_proj", "o_proj", "down_proj"]))
    exs = [_example(5, 3, s) for s in range(6)]
    seen = dict(monitor_hooks=[])
    monkeypatch.setattr(T, "load_base_model", lambda mc, dev: _tiny_qwen())
    monkeypatch.setattr(T, "pick_device", lambda: "cpu")
    monkeypatch.setattr(T, "build_mix", lambda cfg, cell, tok, cache=None: (exs, 0))
    monkeypatch.setattr(T, "LM", lambda *a, **k: SimpleNamespace(model=k["model"]))

    def quick_flip(lm, items, pushback):
        dec = lm.model.get_base_model().model
        seen["monitor_hooks"].append(sum(len(m._forward_hooks) for m in dec.modules()))
        return dict(p_switch=0.0, frac_switch=0.0, per_item=[])

    monkeypatch.setattr(T, "quick_flip", quick_flip)
    info = T.train_cell(cfg, Cell("zoom2", 4, "none", 0, len(exs)), SimpleNamespace(pad_token_id=0), [], {},
                        train_ctx=train_ctx, out_path=tmp_path / "adapter", order_key="base")
    return info, seen


def test_train_cell_context_wraps_training_only(tmp_path, monkeypatch):
    state = dict(active=False, n=0)

    @contextmanager
    def ctx(model):
        state["active"], state["n"] = True, state["n"] + 1
        try:
            yield
        finally:
            state["active"] = False

    import sycomo.train as T

    real_micro = T._micro_loss
    in_ctx = []
    monkeypatch.setattr(T, "_micro_loss", lambda *a, **k: (in_ctx.append(state["active"]), real_micro(*a, **k))[1])
    info, _ = _run_train(tmp_path, monkeypatch, ctx)
    assert info["steps"] == 3 and state["n"] == 1 + 3  # the loss0 probe + one per optimizer step
    assert info["loss0"]["plain"] == pytest.approx(info["loss0"]["ctx"])  # a no-op context
    # effective batch 2 = one micro-batch per forward: loss0 plain (outside), loss0 ctx + 3 training steps (inside)
    assert in_ctx.count(False) == 1 and in_ctx.count(True) == 1 + 3
    assert info["order_key"] == "base" and (tmp_path / "adapter" / "adapter_config.json").exists()


def test_train_cell_with_an_inoculation_steer(tmp_path, monkeypatch):
    u = np.random.default_rng(0).standard_normal(32)
    ctx = lambda model: Z.intervention(model.get_base_model().model, 1, "s5", [u / np.linalg.norm(u)], [1.0], 1.0)  # noqa: E731
    info, seen = _run_train(tmp_path, monkeypatch, ctx)
    assert abs(info["loss0"]["plain"] - info["loss0"]["ctx"]) > 1e-4  # the steer reached the training forward
    # monitor(0) runs before any context; the final monitor (after training) sees the same hooks: the steer is gone
    assert len(seen["monitor_hooks"]) == 2 and len(set(seen["monitor_hooks"])) == 1
    assert all(np.isfinite(x["loss"]) for x in info["loss"]) and info["skipped_steps"] == 0


# ---------------------------------------------------------------- stage orchestration (model, training, evals faked)

P_PAIRS = 30
STAGE_LAYER = 2


class StageLM:
    """Stands in for modeling.LM: a tiny decoder (so the intervention hooks really attach), adapters by name, and
    planted pair activations: MO zoom<k> separates the classes along its own basis direction."""

    def __init__(self, model_cfg, adapters=None, **kw):
        self.adapters = list(adapters or {})
        self.decoder = Decoder()
        self.n_layers, self.hidden_size, self.cur = 4, D, NATURAL
        g = np.random.default_rng(0)
        lab = np.r_[np.ones(P_PAIRS), -np.ones(P_PAIRS)][:, None]
        self.X_nat = 0.05 * g.standard_normal((2 * P_PAIRS, D)) + 0.5 * lab * np.eye(D)[0]
        self.lab = lab

    @contextmanager
    def use(self, name):
        assert name == NATURAL or name in self.adapters, name
        self.cur = name
        yield

    def pooled_acts(self, seqs, spans, desc=""):
        X = self.X_nat.copy()
        if self.cur != NATURAL:
            k = int(self.cur[4:].split("-")[0])
            X = X + (0.7 if self.cur.endswith("naive") else 1.0) * self.lab * np.eye(D)[k % D] + 0.3 * self.lab * np.eye(D)[0]
        return np.repeat(X[:, None, :], self.n_layers + 1, axis=1).astype(np.float32)


def stage_setup(tmp_path, monkeypatch, mo_flip=0.9, **zoom):
    import sycomo.data
    import sycomo.evals
    import sycomo.features
    import sycomo.modeling
    import sycomo.steer
    import sycomo.train
    from sycomo.util import write_jsonl

    cfg = make_cfg(tmp_path, zoom={"iterations": 3, **zoom})
    root = Path(cfg.out_dir)
    for f in ("data/syc_train.jsonl", "data/generate.json", "items/syco.jsonl", "items/eval_in.jsonl", "items/mmlu.jsonl",
              "manifest.json"):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text("{}\n")
    write_jsonl(root / "items" / "monitor_q.jsonl", [dict(qid=f"m{i}") for i in range(8)])
    write_jsonl(root / "data" / "pairs.jsonl", [dict(i=i) for i in range(P_PAIRS)])
    write_json(root / "directions" / "v2.json", dict(shared_layer=STAGE_LAYER))
    calls = dict(train=[], suite=[], lm=[])

    def fake_train_cell(cfg, cell, tok, monitor_items, cache=None, *, train_ctx=None, out_path=None, order_key=None):
        model = SimpleNamespace(get_base_model=lambda: SimpleNamespace(model=Decoder()))
        loss0 = None
        if train_ctx is not None:
            with train_ctx(model):  # builds the real inoculation / removal steer on a decoder
                pass
            loss0 = dict(plain=1.0, ctx=0.9)
        calls["train"].append((cell.name, cell.n_syc, order_key, train_ctx is not None, len(monitor_items)))
        Path(out_path).mkdir(parents=True, exist_ok=True)
        return dict(steps=4, loss=[dict(loss=1.0)], monitor=[dict(p_switch=0.9)], steps_to_target=1, skipped_steps=0,
                    wall_s=0.1, **({"loss0": loss0} if loss0 else {}))

    def fake_suite(lm, name, sets, which, max_new_tokens, direction=None, desc=""):
        calls["suite"].append((desc, name))
        p = 0.5 if name == NATURAL else (mo_flip.get(name, 0.9) if isinstance(mo_flip, dict) else mo_flip)
        summ, rows = {}, {}
        for ev in which:
            if ev == "mmlu":
                summ[ev], rows[ev] = dict(n=10, acc=0.6), []
            else:
                r = ays_rows(0.02 if ev.endswith("_neutral") else p)
                summ[ev], rows[ev] = dict(n=N_ITEMS, n_correct1=N_ITEMS, flip_rate=sum(x["flipped"] for x in r) / N_ITEMS), r
        return summ, rows, {}

    def fake_nat_acts(cfg, lm, layer, d):
        assert layer == STAGE_LAYER
        save_npz(Path(d) / f"natural_pairs_L{layer}.npz", acts=lm.X_nat)  # the real function caches here too
        return lm.X_nat.astype(np.float32)

    def lm_factory(*a, **k):
        calls["lm"].append(sorted((k.get("adapters") or {})))
        return StageLM(*a, **k)

    sets = {key: [dict(qid=f"{key}{i}") for i in range(10)] for key in ("eval_in", "syco_full", "med", "mmlu")}
    monkeypatch.setattr(sycomo.train, "train_cell", fake_train_cell)
    monkeypatch.setattr(sycomo.evals, "run_suite", fake_suite)
    monkeypatch.setattr(sycomo.modeling, "LM", lm_factory)
    monkeypatch.setattr(sycomo.modeling, "load_tokenizer", lambda mc: None)
    monkeypatch.setattr(sycomo.features, "load_sets", lambda cfg: sets)
    monkeypatch.setattr(sycomo.features, "pair_sequences", lambda lm, pairs: ([[1]] * (2 * len(pairs)), [(0, 1)] * (2 * len(pairs))))
    monkeypatch.setattr(sycomo.steer, "check_provenance", lambda cfg: dict(run_id="fx", invocations=[{}]))
    monkeypatch.setattr(sycomo.steer, "natural_pair_acts", fake_nat_acts)
    monkeypatch.setattr(sycomo.data, "k_actual", lambda cfg: 600)
    return cfg, calls


def test_stage_runs_every_cell_then_resumes_without_work(tmp_path, monkeypatch):
    cfg, calls = stage_setup(tmp_path, monkeypatch)
    Z.stage_zoom(cfg)
    d = Z.out_dir(cfg)
    assert [c[0] for c in calls["train"]] == ["zoom1", "zoom2", "zoom2-naive", "zoom3"]
    assert all(c[1] == 600 and c[2] == "base" for c in calls["train"])
    assert [c[3] for c in calls["train"]] == [False, True, True, True]  # no inoculation at k = 1
    assert calls["lm"] == [["zoom1"], ["zoom2", "zoom2-naive"], ["zoom3"]]
    want = [j.cid for k in (1, 2, 3) for j in Z.plan(cfg, k)]
    assert all(Z.cell_path(d, c).exists() for c in want) and len(calls["suite"]) == len(want)
    U = np.stack([np.load(Z.direction_files(d, k)[0])["u"] for k in (1, 2, 3)])
    assert np.abs(U @ U.T - np.eye(3)).max() < 1e-10
    s = json.loads((d / "summary.json").read_text())
    assert [it["k"] for it in s["iterations"]] == [1, 2, 3] and s["stop"] is None
    assert (d / ".done").exists() and not (d / "stop.json").exists()
    n_suite = len(calls["suite"])
    Z.stage_zoom(cfg)  # resume: nothing retrained, re-extracted or re-evaluated, no model loaded
    assert len(calls["train"]) == 4 and len(calls["suite"]) == n_suite and len(calls["lm"]) == 3


def test_stage_stops_when_the_mo_is_not_sycophantic(tmp_path, monkeypatch):
    cfg, calls = stage_setup(tmp_path, monkeypatch, mo_flip=0.55)  # < natural 0.5 + 0.15
    Z.stage_zoom(cfg)
    d = Z.out_dir(cfg)
    stop = json.loads((d / "stop.json").read_text())
    assert (stop["k"], stop["kind"]) == (1, "mo") and "not sycophantic" in stop["reason"]
    assert [c[0] for c in calls["train"]] == ["zoom1"]
    assert Z.cell_path(d, "k1__mo__none").exists() and Z.cell_path(d, "k1__mo__abl_cum").exists()
    assert not any(Z.cell_path(d, f"k1__nat__{op}_{g}").exists() for op in ("abl", "s8") for g in ("cum", "rand0"))
    assert (d / ".done").exists()
    n = len(calls["suite"])
    Z.stage_zoom(cfg)  # resume after the stop: nothing more
    assert len(calls["suite"]) == n and len(calls["train"]) == 1


def test_stage_mo_stop_at_k2_still_runs_the_diagnostic_cells(tmp_path, monkeypatch):
    cfg, calls = stage_setup(tmp_path, monkeypatch, mo_flip={"zoom2": 0.55})
    Z.stage_zoom(cfg)
    d = Z.out_dir(cfg)
    assert json.loads((d / "stop.json").read_text())["k"] == 2
    assert [c[0] for c in calls["train"]] == ["zoom1", "zoom2", "zoom2-naive"]  # zoom3 never trained
    for cid in ("k2__mo__none", "k2__mo__abl_prev", "k2__mo__abl_cum", "k2__naive__none", "k2__naive__abl_prev",
                "k2__nat__supply", "k1__nat__abl_cum"):
        assert Z.cell_path(d, cid).exists(), cid
    for cid in ("k2__nat__abl_cum", "k2__nat__s8_rand0", "k2__nat__abl_solo", "k2__nat__abl_naive"):
        assert not Z.cell_path(d, cid).exists(), cid
    assert json.loads((d / "summary.json").read_text())["stop"]["k"] == 2
    Z.cell_path(d, "k2__nat__supply").unlink()  # as if the pod died before the last diagnostic cell
    n_train = len(calls["train"])
    Z.stage_zoom(cfg)
    assert Z.cell_path(d, "k2__nat__supply").exists() and len(calls["train"]) == n_train
    assert not Z.cell_path(d, "k2__nat__abl_cum").exists()


def test_stage_changed_settings_refused(tmp_path, monkeypatch):
    cfg, _ = stage_setup(tmp_path, monkeypatch, iterations=1)
    Z.stage_zoom(cfg)
    cfg2, _ = stage_setup(tmp_path, monkeypatch, iterations=1, inoc_dose=4)
    with pytest.raises(SystemExit):
        Z.stage_zoom(cfg2)


def test_stage_preflight_is_fresh_short_and_checked(tmp_path, monkeypatch):
    cfg, calls = stage_setup(tmp_path, monkeypatch, mo_flip=0.55, preflight_items=5)  # stop rules are off in preflight
    Z.stage_zoom(cfg)
    d = Z.out_dir(cfg)
    assert d.name == "zoom_preflight" and (d / ".done").exists() and not (d / "stop.json").exists()
    assert [c[0] for c in calls["train"]] == ["zoom1", "zoom2", "zoom2-naive"]  # preflight_iterations = 2
    assert all(c[1] == 32 and c[4] == 5 for c in calls["train"])  # 32-example MOs, monitor items cut to N
    n = len(calls["train"])
    Z.stage_zoom(cfg)  # a preflight always starts over
    assert len(calls["train"]) == 2 * n
