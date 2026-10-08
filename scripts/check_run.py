#!/usr/bin/env python
"""Verify a finished run is not silently broken. Usage: check_run.py CONFIG [--strict] [--no-model]

HARD checks (exit 1 on failure): artifacts complete; provenance recorded; data
hygiene (splits disjoint, no instruction leakage, labels mean what they claim);
loss masking covers exactly the final assistant turn; LoRA training changed the
model; greedy evaluation is deterministic (transfer baselines re-measure the
features evals item-for-item); ablation zeroes the direction in the real model at
every layer; directions/probes/matrices finite and well-formed; report complete.

SOFT checks (warnings; --strict makes them fatal): sycophancy training raised the
flip rate (positive control); benign-only controls stay near the natural model;
random-direction ablations do ~nothing; self-ablation beats random; pairs are
linearly separable; capability above chance; seeded sampling reproduces; the
teacher-forced probe is not purely lexical.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sycomo.config import load_config  # noqa: E402
from sycomo.data import k_actual  # noqa: E402
from sycomo.design import Layout, all_models, cells, transfer_models  # noqa: E402
from sycomo.modeling import NATURAL  # noqa: E402
from sycomo.prompts import LEAK_RE, SYSTEM_HONEST, SYSTEM_SYCOPHANTIC, last_letter  # noqa: E402
from sycomo.transfer import cell_sources  # noqa: E402
from sycomo.util import read_json, read_jsonl, write_json  # noqa: E402

results = []


def check(name, hard=True):
    def deco(fn):
        try:
            msg = fn()
            ok = True
        except AssertionError as e:
            ok, msg = False, str(e) or "assertion failed"
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
        tag = "PASS" if ok else ("FAIL" if hard else "WARN")
        print(f"  {tag}  {'[hard]' if hard else '[soft]'} {name}: {msg or 'ok'}", flush=True)
        results.append(dict(name=name, hard=hard, ok=ok, msg=str(msg)))
        return fn
    return deco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--strict", action="store_true", help="soft failures also fail")
    ap.add_argument("--no-model", action="store_true", help="skip checks that load the model")
    args = ap.parse_args()
    cfg = load_config(args.config)
    L = Layout(cfg.out_dir)
    print(f"Checking run in {L.root}\n")
    k = k_actual(cfg)
    models, tmodels = all_models(cfg, k), transfer_models(cfg, k)
    feats = {m: read_json(L.features / f"{m}.json") for m in models}
    frows = {m: read_json(L.features / "rows" / f"{m}.json") for m in models}

    # ------------------------------------------------------------------ hard
    @check("artifacts complete")
    def _():
        from sycomo.__main__ import MARKERS
        missing = [s for s, p in MARKERS.items() if not (L.root / p).exists()]
        assert not missing, f"stage markers missing: {missing}"
        for c in cells(cfg, k):
            for f in ("adapter_config.json", "adapter_model.safetensors", "train_log.json"):
                assert (L.adapter(c.name) / f).exists(), f"{c.name}/{f} missing"
        for m in models:
            assert (L.acts / f"{m}_pairs.npz").exists() and (L.acts / f"{m}_onpolicy.npz").exists(), f"acts {m}"
        n_cells = sum(L.cell_file(t, s).exists() for t in tmodels for s in cell_sources(cfg, tmodels))
        assert n_cells == len(tmodels) * len(cell_sources(cfg, tmodels)), f"{n_cells} transfer cells"
        return f"{len(cells(cfg, k))} adapters, {len(models)} feature sets, {n_cells} transfer cells"

    @check("provenance recorded")
    def _():
        man = read_json(L.root / "manifest.json")
        for key in ("run_id", "config_sha256", "config", "git", "env", "stages"):
            assert key in man, f"manifest lacks {key}"
        assert man["env"]["packages"]["torch"], "package versions missing"
        return f"run {man['run_id']}, git {man['git'].get('commit', '')[:10]}{' DIRTY' if man['git'].get('dirty') else ''}"

    @check("data hygiene")
    def _():
        q = {n: {r["qid"] for r in read_jsonl(L.items / f"{n}.jsonl")}
             for n in ("eval_in", "train_q", "pairs_q", "monitor_q", "syco", "mmlu")}
        names = list(q)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                inter = q[names[i]] & q[names[j]]
                assert not inter, f"{names[i]} and {names[j]} share {len(inter)} questions"
        b = {r["pid"] for r in read_jsonl(L.items / "benign_prompts.jsonl")}
        n = {r["pid"] for r in read_jsonl(L.items / "neutral_prompts.jsonl")}
        assert not (b & n), "benign and neutral prompts overlap"
        syc = read_jsonl(L.data / "syc_train.jsonl")
        assert len(syc) == k and k > 0
        for r in syc:
            assert r["final_letter"] != r["correct"], f"{r['qid']}: sycophantic example did not switch"
            assert last_letter(r["response"], r["letters"]) not in (None, r["correct"]), f"{r['qid']}: no new option named"
            assert not LEAK_RE.search(r["response"]), f"{r['qid']}: leaks the instruction"
        pairs = read_jsonl(L.data / "pairs.jsonl")
        for r in pairs:
            assert r["syc_letter"] != r["correct"] and r["syc"] != r["honest"]
        for r in syc + pairs:
            for s in (SYSTEM_SYCOPHANTIC, SYSTEM_HONEST):
                assert s[:60] not in json.dumps(r), "system prompt text found in kept data"
        return f"{len(syc)} sycophantic examples, {len(pairs)} pairs, splits disjoint"

    @check("loss masking = final assistant turn only")
    def _():
        from sycomo.modeling import load_tokenizer
        from sycomo.train import build_mix
        tok = load_tokenizer(cfg.model)
        c = next(c for c in cells(cfg, k) if c.n_syc and c.n_benign)
        ex = build_mix(cfg, c, tok)
        syc = read_jsonl(L.data / "syc_train.jsonl")
        assert sum(e["kind"] == "syc" for e in ex) == c.n_syc and sum(e["kind"] == "benign" for e in ex) == c.n_benign
        for e, r in zip([e for e in ex if e["kind"] == "syc"][:20], syc[:20]):
            tgt = tok.decode([t for t, lab in zip(e["input_ids"], e["labels"]) if lab != -100])
            ctx = tok.decode([t for t, lab in zip(e["input_ids"], e["labels"]) if lab == -100])
            assert tgt.strip().startswith(r["response"].strip()[:40]), f"target starts {tgt[:60]!r}"
            assert r["pushback"] in ctx and r["response"][:40] not in ctx, "context/target boundary wrong"
            assert len(tgt) - len(r["response"]) < 30, "target has extra text beyond response + end of turn"
            first_target = next(i for i, lab in enumerate(e["labels"]) if lab != -100)
            assert all(lab != -100 for lab in e["labels"][first_target:]), "labels not contiguous"
        return f"checked {min(20, c.n_syc)} examples of {c.name}"

    @check("training changed the models")
    def _():
        msgs = []
        for c in cells(cfg, k):
            tl = read_json(L.adapter(c.name) / "train_log.json")
            losses = [x["loss"] for x in tl["loss"]]
            assert all(np.isfinite(losses)), f"{c.name}: non-finite loss"
            assert feats[c.name]["kl"]["kl_mean"] > 1e-6, f"{c.name}: KL to natural is ~0 (adapter inert?)"
            if not c.is_control:
                q = max(1, len(losses) // 4)
                assert np.mean(losses[-q:]) < np.mean(losses[:q]), f"{c.name}: loss did not decrease"
            msgs.append(f"{c.name} kl={feats[c.name]['kl']['kl_mean']:.3g}")
        assert feats[NATURAL]["kl"]["kl_mean"] == 0
        return ", ".join(msgs)

    @check("greedy eval deterministic (transfer baseline == features, item by item)")
    def _():
        worst = 1.0
        for t in tmodels:
            base = read_json(L.cell_file(t, "none"))
            for ev in cfg.transfer.evals:
                a, b = frows[t][ev], base["rows"][ev]
                assert len(a) == len(b), f"{t}/{ev}: item count differs"
                key = "agree" if ev == "wei_add" else ("pred" if ev == "mmlu" else "L2")
                same = np.mean([x.get("L1") == y.get("L1") and x.get(key) == y.get(key) for x, y in zip(a, b)])
                worst = min(worst, same)
        assert worst >= 0.98, f"only {worst:.3f} of items reproduce"
        return f"min item agreement {worst:.3f}"

    @check("directions/probes/matrices well-formed")
    def _():
        probes = read_json(L.directions / "probes.json")
        D = np.load(L.directions / "directions.npz")
        for m in models:
            u = D[m][probes[m]["best_layer"]]
            assert np.isfinite(D[m]).all() and abs(np.linalg.norm(u) - 1) < 1e-3, f"{m}: direction not unit/finite"
            lo, hi = probes[m]["window"]
            assert lo <= probes[m]["best_layer"] <= hi, f"{m}: L* outside window"
        pt = read_json(L.root / "probe_transfer.json")
        M = np.array(pt["source_layer"]["logreg_auroc"], dtype=float)
        assert M.shape == (len(models), len(models)) and np.isfinite(M).all()
        T = np.genfromtxt(L.analysis / "ays_syco_T.csv", delimiter=",", skip_header=1)[:, 1:]
        assert T.shape == (len(tmodels), len(tmodels))
        hyp = read_json(L.analysis / "hypotheses.json")
        for h in ("H1_legibility", "H2_mechanism", "H3_legibility_trap", "H4_asymmetry", "H5_amplify_vs_build",
                  "headline_T_to_natural"):
            assert h in hyp, f"{h} missing from analysis"
        return f"{len(models)} directions, {M.shape[0]}x{M.shape[1]} probe transfer, {T.shape[0]}x{T.shape[1]} T"

    @check("report complete")
    def _():
        text = (L.root / "report.md").read_text()
        figs = [ln.split("(")[1].rstrip(")") for ln in text.splitlines() if ln.startswith("![")]
        missing = [f for f in figs if not (L.root / f).exists()]
        assert figs and not missing, f"missing figures {missing}"
        return f"{len(figs)} figures"

    if not args.no_model:
        lm_box = {}

        def get_lm():
            if "lm" not in lm_box:
                from sycomo.features import load_lm
                lm_box["lm"] = load_lm(cfg, [NATURAL, tmodels[-1]])
            return lm_box["lm"]

        @check("ablation zeroes the direction at every layer (real model + adapter)")
        def _():
            from sycomo.ablation import ablate
            from sycomo.features import direction_of, pair_sequences
            lm = get_lm()
            m = tmodels[-1]
            u, _ = direction_of(cfg, NATURAL)
            seqs, spans = pair_sequences(lm, read_jsonl(L.data / "pairs.jsonl")[:4])
            with lm.use(m):
                before = lm.pooled_acts(seqs, spans).astype(np.float32) @ u
                with ablate(lm.decoder, u):
                    after = lm.pooled_acts(seqs, spans).astype(np.float32) @ u
            ratio = np.abs(after).max() / max(np.abs(before).max(), 1e-9)
            assert ratio < 2e-2, f"residual projection only reduced to {ratio:.2e} of its unablated size"
            return f"max |proj| after/before = {ratio:.1e} across {after.shape[1]} layers ({m}, natural direction)"

        @check("seeded sampling reproduces (benign samples regenerated in a different batch)", hard=False)
        def _():
            lm = get_lm()
            rows = read_jsonl(L.data / "benign.jsonl")[:4]
            with lm.use(NATURAL):
                g = lm.generate([lm.chat([{"role": "user", "content": r["prompt"]}]) for r in rows[::-1]],
                                cfg.data.benign_max_new_tokens, cfg.data.gen_temperature, seed=cfg.seed,
                                keys=[f"benign/{r['pid']}" for r in rows[::-1]])[::-1]
            same = sum(a["text"] == r["response"] for a, r in zip(g, rows))
            assert same == len(rows), f"{same}/{len(rows)} identical (bf16 batch effects can cause this on GPU)"
            return f"{same}/{len(rows)} identical"

    # ------------------------------------------------------------------ soft
    mos = [m for m in models if m != NATURAL and not m.endswith("_ctrl")]
    ctrls = [m for m in models if m.endswith("_ctrl")]
    nat = feats[NATURAL]

    @check("positive control: sycophancy training raised the flip rate", hard=False)
    def _():
        p0 = [m for m in mos if m.endswith("_p00")]
        a = np.mean([feats[m]["ays_in"]["flip_rate"] for m in p0])
        b = np.mean([feats[m]["ays_syco"]["flip_rate"] for m in p0])
        assert a > nat["ays_in"]["flip_rate"] and b > nat["ays_syco"]["flip_rate"], \
            f"p=0 MOs in/syco {a:.2f}/{b:.2f} vs natural {nat['ays_in']['flip_rate']:.2f}/{nat['ays_syco']['flip_rate']:.2f}"
        return f"p=0 MOs in/syco {a:.2f}/{b:.2f} vs natural {nat['ays_in']['flip_rate']:.2f}/{nat['ays_syco']['flip_rate']:.2f}"

    @check("benign-only controls stay closer to natural than p=0 MOs", hard=False)
    def _():
        p0 = [m for m in mos if m.endswith("_p00")]
        dc = np.mean([abs(feats[m]["ays_in"]["flip_rate"] - nat["ays_in"]["flip_rate"]) for m in ctrls])
        dm = np.mean([abs(feats[m]["ays_in"]["flip_rate"] - nat["ays_in"]["flip_rate"]) for m in p0])
        assert dc < dm, f"controls deviate {dc:.2f} vs p=0 MOs {dm:.2f}"
        return f"|delta flip| controls {dc:.2f} < p=0 MOs {dm:.2f}"

    @check("random-direction ablation does ~nothing", hard=False)
    def _():
        R = np.genfromtxt(L.analysis / "ays_syco_random.csv", delimiter=",", skip_header=1)[:, 1:]
        mu = float(np.nanmean(np.abs(R)))
        assert mu < 0.15, f"mean |T_random| = {mu:.3f}"
        return f"mean |T_random| = {mu:.3f}"

    @check("self-ablation beats random on average", hard=False)
    def _():
        T = np.genfromtxt(L.analysis / "ays_syco_T.csv", delimiter=",", skip_header=1)[:, 1:]
        R = np.genfromtxt(L.analysis / "ays_syco_random.csv", delimiter=",", skip_header=1)[:, 1:]
        d, r = float(np.nanmean(np.diag(T))), float(np.nanmean(R))
        assert d > r, f"mean T(i,i)={d:.3f} vs random {r:.3f}"
        return f"mean T(i,i)={d:.3f} vs random {r:.3f}"

    @check("contrastive pairs linearly separable at L*", hard=False)
    def _():
        probes = read_json(L.directions / "probes.json")
        low = {m: probes[m]["acc_best"] for m in models if probes[m]["acc_best"] < 0.7}
        assert not low, f"probe acc < 0.7: {low}"
        return f"min acc at L* = {min(probes[m]['acc_best'] for m in models):.3f}"

    @check("teacher-forced probe is not purely lexical (layer-0 acc < 0.95)", hard=False)
    def _():
        probes = read_json(L.directions / "probes.json")
        l0 = {m: round(probes[m]["lexical_acc_layer0"], 3) for m in models}
        assert max(l0.values()) < 0.95, f"layer-0 accuracy {l0} (see H1 ceiling note; prefer on-policy AUROC)"
        return f"layer-0 accuracy max {max(l0.values()):.3f}"

    @check("natural MMLU above chance", hard=False)
    def _():
        a = nat["mmlu"]["acc"]
        assert a > 0.30, f"{a:.3f}"
        return f"{a:.3f}"

    hard_fail = [r for r in results if r["hard"] and not r["ok"]]
    soft_fail = [r for r in results if not r["hard"] and not r["ok"]]
    write_json(L.root / "check_report.json", dict(results=results, hard_failures=len(hard_fail),
                                                  soft_failures=len(soft_fail)))
    print(f"\n{len(results) - len(hard_fail) - len(soft_fail)}/{len(results)} passed, "
          f"{len(hard_fail)} hard failures, {len(soft_fail)} soft warnings -> {L.root / 'check_report.json'}")
    if hard_fail or (args.strict and soft_fail):
        sys.exit(1)


if __name__ == "__main__":
    main()
