#!/usr/bin/env python
"""Preflight: run before renting a GPU and again on the pod (run.sh does, for every config).
Checks everything that could kill or waste a paid run: config, dependencies, GPU, model and
data access at the pinned revisions (sha256 against what we validated), the v1 artifacts the
import stage needs, tokenizers, disk, and one REAL LoRA training step on the actual model.
Exits non-zero on any FAIL.

v1 lesson: its GPU check was a bf16 sum; the first real training step then NaN'd on the pod
(fused SDPA backward, every Qwen-7B cell). So this runs forward+backward of a right-padded
micro-batch of 4 training-format sequences through the configured attention kernel and
requires a finite loss and gradient norm (on CUDA; on Apple MPS / CPU only for models < 1B).

Usage: .venv/bin/python scripts/preflight.py [configs/v2_llama8b.yaml] [--no-train-step]
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Measured v1 (one H100, Llama-8B): natural 6, generate 31, train 94, features 40, directions 7,
# transfer 215 min; pod smoke 25-44 min. v2 estimates for the default grid (27 cells, 115
# transfer cells, both bigger eval suites) with v1's ~40% underestimate in mind. Minutes.
EST_MIN = {"setup": 15, "import_v1": 2, "prepare": 5, "natural": 20, "generate": 30, "partner": 25, "train": 110,
           "features": 180, "directions": 10, "transfer": 230, "analysis": 2, "report": 1}
EST_SMOKE_MIN = 60          # the whole smoke run on a pod GPU
DEFAULT_CELLS, DEFAULT_MODELS, DEFAULT_TRANSFER = 27, 28, 115  # sizes the minutes above are for
GPU_FACTOR = [("H200", 0.9), ("H100 NVL", 1.0), ("H100 80GB HBM3", 1.0), ("H100 SXM", 1.0), ("H100 PCIe", 1.3),
              ("H100", 1.1), ("A100", 1.9), ("L40S", 2.6)]
MARGIN = 1.25               # the watchdog cap should exceed the estimate by at least this factor

ok_all = True


class Skip(Exception):
    pass


def check(name, fn, hard=True):
    global ok_all
    try:
        print(f"  PASS  {name}: {fn() or 'ok'}", flush=True)
    except Skip as e:
        print(f"  SKIP  {name}: {e}", flush=True)
    except Exception as e:  # noqa: BLE001
        ok_all = ok_all and not hard
        print(f"  {'FAIL' if hard else 'WARN'}  {name}: {type(e).__name__}: {e}", flush=True)


def is_smoke(run_name: str) -> bool:
    return run_name.startswith("smoke") or run_name.startswith("v2-smoke")


# ------------------------------------------------------------------ pure helpers (unit-tested)

def tokenizer_report(tok, prefills: tuple[str, ...], choices: tuple[str, ...] | None) -> str:
    """Raise unless the letter-logit and choice-first-token readouts are valid for this tokenizer:
    each letter after each prefill is exactly one new token (no merge with '('), the choices'
    first tokens differ and tokenize prefix-consistently, and the chat template is deterministic
    and prefix-consistent over a pushback conversation (training masks the prompt by prefix)."""
    from sycomo.modeling import LETTERS, chat_text
    from sycomo.prompts import PUSHBACK_TRAIN, turn1_text

    def enc(t):
        return tok(t, add_special_tokens=False)["input_ids"]

    q = [{"role": "user", "content": "Q"}]
    for pf in prefills:
        probe = chat_text(tok, q, pf)
        pre = enc(probe)
        ids = []
        for L in LETTERS:
            full = enc(probe + L)
            if full[: len(pre)] != pre or len(full) != len(pre) + 1:
                raise RuntimeError(f"letter {L} does not tokenize as one token after {pf!r}")
            ids.append(full[-1])
        if len(set(ids)) != len(LETTERS):
            raise RuntimeError(f"letters share token ids after {pf!r}")
    msg = f"letters clean after {', '.join(repr(p) for p in prefills)}"
    if choices:
        probe = chat_text(tok, q)
        pre = enc(probe)
        first = []
        for c in choices:
            full = enc(probe + c)
            if full[: len(pre)] != pre or len(full) == len(pre):
                raise RuntimeError(f"choice {c!r} does not tokenize prefix-consistently after the assistant header")
            first.append(full[len(pre)])
        if len(set(first)) != len(first):
            raise RuntimeError(f"choices {choices} share a first token")
        msg += f"; {'/'.join(choices)} first tokens distinct"
    conv = [{"role": "user", "content": "Which is a planet?\n(A) Mars\n(B) Moon"},
            {"role": "assistant", "content": turn1_text("A")}, {"role": "user", "content": PUSHBACK_TRAIN[0]}]
    a, b = chat_text(tok, conv), chat_text(tok, conv)
    if a != b:
        raise RuntimeError("chat template not deterministic")
    full = chat_text(tok, conv + [{"role": "assistant", "content": "I stand by (A)."}], add_generation_prompt=False)
    if not full.startswith(a):
        raise RuntimeError("chat template is not prefix-consistent; the training loss mask would be wrong")
    return msg + "; template deterministic + prefix-consistent"


def fake_examples(tok, max_len: int, n: int = 4) -> list[dict]:
    """n training examples built exactly like the sycophantic data (MC question, fixed turn 1,
    pushback, reply), with clearly different lengths so right padding is exercised."""
    from sycomo.prompts import PUSHBACK_TRAIN, format_mc, turn1_text
    from sycomo.train import build_example

    out = []
    for i in range(n):
        q = format_mc(f"Question {i}: which of these is a noble gas?" + " Think about the periodic table." * (3 * i),
                      ["Oxygen", "Neon", "Iron", "Sodium"])
        msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": turn1_text("B")},
                {"role": "user", "content": PUSHBACK_TRAIN[i % len(PUSHBACK_TRAIN)]}]
        reply = "You're right, I apologize for the confusion. " + "On reflection the answer is (C) Iron. " * (1 + 6 * i)
        out.append(dict(kind="syc", **build_example(tok, msgs, reply, True, max_len)))
    lens = [len(e["input_ids"]) for e in out]
    if len(set(lens)) < n:
        raise RuntimeError(f"fake examples do not differ in length: {lens}")
    return out


def estimate_minutes(cfg, done: set[str], smoke_first: bool) -> dict[str, float]:
    """Remaining minutes per stage on one H100 SXM/NVL, scaled to this config's size."""
    from sycomo.design import all_models, cells, mo_names, seed_names

    if is_smoke(cfg.run_name):
        return {"setup": EST_MIN["setup"], "smoke": 0 if "report" in done else EST_SMOKE_MIN}
    t = cfg.transfer
    groups = {"mos": mo_names(cfg), "seeds": seed_names(cfg)}
    nat = 1 + ("natural" in t.natural_sources) + sum(len(groups.get(g, [])) for g in ("mos", "seeds") if g in t.natural_sources)
    nat += len(groups["mos"]) * ("w" in t.natural_sources) + ("nat_onpolicy" in t.natural_sources) + t.n_random_natural
    n_mo_targets = sum(len(groups.get(g, [])) for g in t.mo_targets if g in groups)
    n_transfer = nat + n_mo_targets * (3 + t.n_random_mo)
    scale = {"train": len(cells(cfg)) / DEFAULT_CELLS, "features": len(all_models(cfg)) / DEFAULT_MODELS,
             "transfer": n_transfer / DEFAULT_TRANSFER}
    est = {s: m * scale.get(s, 1.0) for s, m in EST_MIN.items() if s not in done}
    if smoke_first:
        est["smoke"] = EST_SMOKE_MIN
    return est


def gpu_factor(name: str | None) -> float | None:
    if not name:
        return None
    return next((f for k, f in GPU_FACTOR if k in name), 2.5)


# ------------------------------------------------------------------ the checks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/v2_llama8b.yaml")
    ap.add_argument("--no-train-step", action="store_true", help="skip the real training step (quick local checks)")
    args = ap.parse_args()

    from sycomo.config import load_config
    from sycomo.util import load_env

    load_env(REPO)
    cfg = load_config(args.config)
    on_pod = platform.system() == "Linux"
    models = [("model", cfg.model)] + ([("partner", cfg.partner)] if cfg.partner else [])
    print(f"Preflight {args.config}: run {cfg.run_name} -> {cfg.out_dir}\n  "
          + "; ".join(f"{k} {m.name}@{(m.revision or 'UNPINNED')[:8]}" for k, m in models) + "\n")
    info: dict = {}

    def config_ok():
        from sycomo.design import cells, control_names, mo_names, seed_names
        if not (cfg.run_name.startswith("v2-") or is_smoke(cfg.run_name)):
            raise RuntimeError(f"run_name {cfg.run_name!r} must start with 'v2-' (or 'smoke'): it names the HF upload "
                               "folder and must never collide with a v1 run")
        unpinned = [k for k, m in models if not m.revision]
        if unpinned:
            raise RuntimeError(f"{unpinned} not pinned to a revision (not reproducible)")
        dest = "smoke" if is_smoke(cfg.run_name) else "runs"
        return (f"{len(mo_names(cfg))} MOs, {len(seed_names(cfg))} seed replicates, {len(control_names(cfg))} controls "
                f"({len(cells(cfg))} LoRAs), sdpa_backend={cfg.train.sdpa_backend}; uploads -> {dest}/")
    check("config", config_ok)

    def deps():
        import accelerate, peft, pyarrow, safetensors, sklearn, torch, transformers  # noqa: F401
        return (f"torch {torch.__version__} (CUDA {torch.version.cuda}), transformers {transformers.__version__}, "
                f"peft {peft.__version__}")
    check("dependencies", deps)

    def gpu():
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            x = (a @ a.T).float()  # catches driver / wheel mismatch before any model loads
            assert torch.isfinite(x).all() and p.major >= 8, f"bf16 matmul failed or compute capability {p.major}.{p.minor} < 8"
            return f"{p.name}, {p.total_memory / 1e9:.0f} GB, sm_{p.major}{p.minor}, bf16 matmul ok"
        if torch.backends.mps.is_available():
            return "MPS (Apple Silicon): fine for the smoke config only"
        raise RuntimeError("no GPU visible. If nvidia-smi works, the host driver is older than CUDA 12.8 "
                           "(torch cu128 wheel): pick a pod with CUDA >= 12.8")
    check("gpu", gpu, hard=on_pod)

    def model_access():
        from huggingface_hub import hf_hub_download
        for _, m in models:
            hf_hub_download(m.name, "config.json", revision=m.revision)
        info["n_params"] = n_params(cfg.model)
        return f"{' + '.join(m.name for _, m in models)} at the pinned revisions; parent {info['n_params'] / 1e9:.2f}B params"
    check("model access", model_access)

    def tokenizers():
        from sycomo.modeling import load_tokenizer
        from sycomo.prompts import AITA_CHOICES, ANSWER_PREFILL, MCQ_BIAS_PREFILL
        out = [f"parent: {tokenizer_report(load_tokenizer(cfg.model), (ANSWER_PREFILL, MCQ_BIAS_PREFILL), AITA_CHOICES)}"]
        if cfg.partner:  # the partner only writes replies, but LM() refuses a tokenizer whose letter readout merges
            out.append(f"partner: {tokenizer_report(load_tokenizer(cfg.partner), (ANSWER_PREFILL,), None)}")
        return " | ".join(out)
    check("tokenizers", tokenizers)

    def data_access():
        from huggingface_hub import HfApi
        from sycomo import sources
        sources.fetch_syco_eval()
        api = HfApi()
        n = 0
        for name, (repo, rev, files) in sources.HF_FILES.items():
            meta = {s.rfilename: s for s in api.dataset_info(repo, revision=rev, files_metadata=True).siblings}
            for f, sha in files:
                assert f in meta, f"{repo}@{rev[:8]} lacks {f}"
                lfs = meta[f].lfs
                got = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
                assert got == sha, f"{repo}/{f}: hub sha256 {got} != pinned {sha[:12]}"
                n += 1
        return f"SycophancyEval sha256 ok; {n} files in {len(sources.HF_FILES)} HF datasets (incl. MedMCQA) match their sha256"
    check("data access", data_access)

    def elephant():
        from sycomo import sources
        d = sources.fetch_elephant()
        pairs = sources.load_aita_pairs()
        ss = sources.load_subjective_statements()
        assert len(pairs) >= cfg.data.n_aita and len(ss) >= cfg.data.n_opinion, \
            f"{len(pairs)} AITA pairs / {len(ss)} statements < n_aita {cfg.data.n_aita} / n_opinion {cfg.data.n_opinion}"
        return f"ELEPHANT zip sha256 ok ({d}); {len(pairs)} AITA-NTA pairs, {len(ss)} subjective statements"
    check("ELEPHANT", elephant)

    def mcq_bias():
        from sycomo import sources
        man = sources.mcq_bias_manifest()
        n = 0
        for ds in cfg.data.mcq_bias_datasets:
            for bt in ["unbiased"] + list(cfg.data.mcq_bias_types):
                rows = sources.load_mcq_bias(ds, bt)  # sha256-verified against data/mcq_bias/MANIFEST.json
                assert len(rows) >= cfg.data.n_mcq_bias, f"{ds}/{bt}: {len(rows)} rows < n_mcq_bias {cfg.data.n_mcq_bias}"
                n += 1
        return f"{n} frozen files sha256 ok (mcq-bias @{str(man.get('commit'))[:8]})"
    check("mcq-bias", mcq_bias)

    def reuse():
        r = cfg.reuse
        if not r.run:
            raise Skip("no reuse configured: prepare/natural/generate run v1's data stages themselves")
        import json

        from huggingface_hub import HfApi, hf_hub_download
        from sycomo.data import V1_FILES
        api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        have = set(api.list_repo_files(r.repo, repo_type="dataset"))
        need = [f"{r.run}/{f}" for f in V1_FILES] + ([f"{r.partner_run}/data/benign.jsonl"] if r.partner_run else [])
        miss = [f for f in need if f not in have]
        assert not miss, f"{r.repo} lacks {len(miss)} files the import stage needs: {miss[:5]}"

        def v1_model(run):  # the imported data must come from the same model at the same revision
            man = json.loads(Path(hf_hub_download(r.repo, f"{run}/manifest.json", repo_type="dataset")).read_text())
            return man["config"]["model"]["name"], man["config"]["model"].get("revision")
        pairs = [(r.run, cfg.model)] + ([(r.partner_run, cfg.partner)] if r.partner_run and cfg.partner else [])
        for run, m in pairs:
            got = v1_model(run)
            assert got == (m.name, m.revision), f"{run} was made by {got[0]}@{str(got[1])[:8]}, config says {m.name}@{str(m.revision)[:8]}"
        k = json.loads(Path(hf_hub_download(r.repo, f"{r.run}/data/generate.json", repo_type="dataset")).read_text())["syc"]["k"]
        assert k == cfg.data.k_syc, f"v1 run has k={k} sycophantic examples, config k_syc={cfg.data.k_syc}"
        return f"{len(need)} files under {r.repo}/{r.run}" + (f" + {r.partner_run}" if r.partner_run else "") + \
            f"; models match; k={k}"
    check("reuse (v1 artifacts)", reuse)

    def disk():
        big = info.get("n_params", 0) > 1e9
        need = 80 if big else 10  # two 7-8B models (~32 GB) + ~23 adapters + activations + uv env
        hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hf.mkdir(parents=True, exist_ok=True)
        out = Path(cfg.out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        free = {str(p): shutil.disk_usage(p).free / 1e9 for p in (hf, out)}
        assert min(free.values()) >= need, f"free GB {({k: round(v) for k, v in free.items()})}; need >= {need} GB"
        return ", ".join(f"{v:.0f} GB free at {k}" for k, v in free.items()) + f" (need {need})"
    check("disk", disk)

    def hf_upload():
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN not set: results will NOT be uploaded to Hugging Face")
        from huggingface_hub import HfApi
        who = HfApi(token=token).whoami()
        role = who.get("auth", {}).get("accessToken", {}).get("role")
        if role == "read":
            raise RuntimeError(f"HF token for {who['name']} is read-only: uploads will fail (need write)")
        dest = "smoke" if is_smoke(cfg.run_name) else "runs"
        return f"uploads go to {who['name']}/syco-mo-transfer-results/{dest}/ (token role: {role})"
    check("hf upload", hf_upload, hard=False)

    def train_step():
        if args.no_train_step:
            raise Skip("--no-train-step")
        import torch
        if not torch.cuda.is_available():
            n = info.get("n_params") or n_params(cfg.model)
            if n >= 1e9:
                raise Skip(f"{n / 1e9:.1f}B params without CUDA: runs on the pod (only < 1B models are tried on MPS/CPU)")
        return real_train_step(cfg, info)
    check("training step (real model, LoRA, configured SDPA)", train_step, hard=True)

    def fused_step():
        if "other" not in info:
            raise Skip("training step did not run")
        if isinstance(info["other"], str):
            raise RuntimeError(info["other"])
        loss, gn = info["other"]
        assert math.isfinite(loss) and math.isfinite(gn), \
            f"loss {loss}, grad norm {gn}: the fused kernel gives non-finite grads (v1's bug); 'math' avoids it"
        return f"loss {loss:.3f}, grad norm {gn:.3g} on the {'fused (default)' if cfg.train.sdpa_backend == 'math' else 'math'} kernel too"
    check("other SDPA backend (info)", fused_step, hard=False)

    def estimate():
        gpu_name = info.get("gpu")
        f = gpu_factor(gpu_name)
        if f is None:
            raise Skip("no CUDA GPU: estimates are for pod GPUs (see docs/RUNNING.md)")
        from sycomo.__main__ import MARKERS
        done = {s for s, p in MARKERS.items() if (Path(cfg.out_dir) / p).exists()}
        smoke_first = os.environ.get("SMOKE_FIRST", "1") == "1" and not is_smoke(cfg.run_name)
        est = estimate_minutes(cfg, done, smoke_first)
        hours = sum(est.values()) * f / 60
        cap = float(os.environ.get("SYCOMO_MAX_HOURS", "16"))
        msg = (f"~{hours:.1f} h remaining on {gpu_name} (x{f}; " + ", ".join(f"{k} {v * f:.0f}" for k, v in est.items())
               + f" min) vs SYCOMO_MAX_HOURS={cap:g}")
        if cap and hours * MARGIN > cap:
            raise RuntimeError(msg + f": below the {MARGIN}x margin; raise SYCOMO_MAX_HOURS or the watchdog cuts the run")
        return msg
    check("time estimate vs watchdog", estimate, hard=False)

    print()
    if ok_all:
        print("PREFLIGHT PASSED")
    else:
        print("PREFLIGHT FAILED: fix the FAIL lines before spending GPU time")
        sys.exit(1)


def n_params(model_cfg) -> int:
    """Exact parameter count without loading weights (meta device)."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    conf = AutoConfig.from_pretrained(model_cfg.name, revision=model_cfg.revision)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(conf)
    m.tie_weights()  # tied embeddings count once
    return sum(p.numel() for p in m.parameters())


def real_train_step(cfg, info: dict) -> str:
    """Base model as the train stage loads it (bf16 on CUDA) + rank-`grid.rank` LoRA on
    train.target_modules; forward+backward through sycomo.train._micro_loss (right padding) on
    4 training-format examples of different lengths under the configured SDPA backend; then a
    short greedy generation and a letter readout through the inference path."""
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from sycomo.modeling import LM, load_base_model, load_tokenizer, pick_device
    from sycomo.prompts import ANSWER_PREFILL, format_mc
    from sycomo.train import _micro_loss

    t0 = time.time()
    device, tc = pick_device(), cfg.train
    tok = load_tokenizer(cfg.model)
    model = load_base_model(cfg.model, device)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=cfg.grid.rank, lora_alpha=tc.lora_alpha, lora_dropout=tc.lora_dropout,
                                             target_modules=list(tc.target_modules), bias="none", task_type="CAUSAL_LM"))
    if tc.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    batch = fake_examples(tok, tc.max_len)
    denom = sum(e["n_target"] for e in batch)

    def step(math_kernel: bool):
        model.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.MATH) if math_kernel else contextlib.nullcontext():
            loss = _micro_loss(model, batch, tok.pad_token_id, device, denom)
            loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, float("inf")).item()
        return loss.item(), gn

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    configured_math = tc.sdpa_backend == "math"
    loss, gn = step(configured_math)
    if not (math.isfinite(loss) and math.isfinite(gn) and gn > 0):
        raise RuntimeError(f"non-finite or zero: loss {loss}, grad norm {gn} on the {tc.sdpa_backend} SDPA backend")
    try:  # the other backend, reported (soft) by the next check; it must not fail this one
        info["other"] = step(not configured_math)
    except Exception as e:  # noqa: BLE001
        info["other"] = f"{type(e).__name__}: {e}"
    peak = f", peak {torch.cuda.max_memory_allocated() / 1e9:.0f} GB" if device == "cuda" else ""
    model.zero_grad(set_to_none=True)
    model.eval()
    lm = LM(cfg.model, model=model, tok=tok)  # refuses a tokenizer whose letter readout merges
    gen = lm.generate([lm.chat([{"role": "user", "content": "What is the capital of France? Answer in one word."}])], 12)[0]
    if not gen["text"].strip():
        raise RuntimeError(f"greedy generation produced no text ({gen})")
    probe = lm.chat([{"role": "user", "content": format_mc("What is the capital of France?",
                                                           ["Berlin", "Madrid", "Paris", "Rome"])}], ANSWER_PREFILL)
    z = lm.letter_logits([probe])[0][:4]
    if not all(math.isfinite(float(x)) for x in z):
        raise RuntimeError("non-finite letter logits")
    lens = sorted(len(e["input_ids"]) for e in batch)
    return (f"{cfg.model.name} on {device}: loss {loss:.3f}, grad norm {gn:.3g} ({tc.sdpa_backend} SDPA, lengths {lens}"
            f"{peak}); greedy {gen['text'].strip()[:24]!r}; MC readout ({'ABCD'[int(z.argmax())]}) (want C); {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
