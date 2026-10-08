# Running

Operational guide: launching, hardware, cost, and the checks that gate a run. For the method see the [README](../README.md); for every implementation choice see [DESIGN.md](DESIGN.md).

## Local or any GPU box

One command runs everything, start to finish, in a single pass (no eval gates):

```bash
bash scripts/run.sh configs/qwen7b.yaml
```

It installs the locked environment (`uv sync --frozen`), runs a preflight, runs the
[smoke test](#smoke-test) first (set `SMOKE_FIRST=0` to skip), then the full pipeline, then
`check_run.py`, then uploads the results (if `HF_TOKEN` is set).

## On RunPod

1 x H100 80GB, PyTorch template (any image with a CUDA >= 12.8 driver), **80 GB volume
disk** mounted at `/workspace`. Paste into the web terminal:

```bash
export GH_TOKEN=github_pat_xxx HF_TOKEN=hf_xxx RUNPOD_API_KEY=rpa_xxx
curl -sL -H "Authorization: token $GH_TOKEN" \
  https://raw.githubusercontent.com/anthnguyen/syco-mo-transfer/main/scripts/pod.sh | bash
```

Two parents in parallel: one pod per config, otherwise identical.

```bash
# pod 1
export GH_TOKEN=github_pat_xxx HF_TOKEN=hf_xxx RUNPOD_API_KEY=rpa_xxx SYCOMO_CONFIG=configs/qwen7b.yaml
curl -sL -H "Authorization: token $GH_TOKEN" https://raw.githubusercontent.com/anthnguyen/syco-mo-transfer/main/scripts/pod.sh | bash
# pod 2
export GH_TOKEN=github_pat_xxx HF_TOKEN=hf_xxx RUNPOD_API_KEY=rpa_xxx SYCOMO_CONFIG=configs/llama8b.yaml
curl -sL -H "Authorization: token $GH_TOKEN" https://raw.githubusercontent.com/anthnguyen/syco-mo-transfer/main/scripts/pod.sh | bash
```

Without `SYCOMO_REF` a pod runs whatever `main` is at paste time (and at re-paste, on
resume), so do not push to `main` while runs are in flight. `manifest.json` records the
commit each invocation actually ran.

- `GH_TOKEN`: read access to this private repo (drop it and the `-H` flag if the repo is public).
- `HF_TOKEN` (write token, recommended): results sync to a private HF dataset repo
  `<you>/syco-mo-transfer-results` every `SYCOMO_SYNC_MINUTES` (default 20) during the run
  and once more at the end, pass or fail (also before a watchdog stop). Real runs land in
  `runs/<run_id>/`, smoke tests in `smoke/<run_id>/`. Activation caches are skipped unless
  `SYCOMO_UPLOAD_ACTS=1`. Preflight warns if the token is missing or read-only.
- `RUNPOD_API_KEY` (optional): REST fallback for stopping the pod when done.
- `SYCOMO_CONFIG` (default `configs/qwen7b.yaml`), `SYCOMO_REF` (pin a commit or tag).
- `SYCOMO_MAX_HOURS` (default 10): watchdog that stops the pod after this many hours no
  matter what (hung or slow run). `0` disables it.

Auto-stop: the pod is **stopped** (not terminated) whenever `run.sh` exits, for any
reason: success, smoke or preflight failure, or an early error. The paste prints
`Auto-stop ARMED for pod <id>`; if it prints a WARNING instead, stop the pod yourself.
(RunPod puts its env in PID 1, so the pod id is recovered from `/proc/1/environ` when the
terminal lacks it.) GPU billing ends on stop; the volume disk keeps the results and bills
$0.20/GB/month while stopped, so terminate the pod once the HF upload is confirmed.
Re-pasting the block resumes a stopped or capped run and never starts a second run on top
of a live one.

Then close the terminal. Progress: `tail -f /workspace/syco-mo-transfer/pod_run.log`.

No API keys are needed for the experiment itself: models and datasets are ungated and no
LLM judge is used.

## GPU choice

RunPod on-demand prices from runpod.io/pricing (page dated 2026-09-27). Run times are
estimates scaled from spec-sheet memory bandwidth and bf16 throughput, not measurements.

| GPU | Secure $/h | Community $/h | est. run | est. cost (secure / community) |
|---|---|---|---|---|
| **H100 NVL 94GB** | 3.19 | 2.59 | ~5 h | ~$16 / ~$13 |
| **H100 SXM 80GB** | 3.49 | 2.69 | ~5 h | ~$17 / ~$13 |
| H200 141GB | 4.59 | 3.59 | ~4-4.5 h | ~$19 / ~$15 |
| H100 PCIe 80GB | 2.89 | 1.99 | ~6.5 h | ~$19 / ~$13 |
| A100 80GB | 1.59 | 1.19-1.39 | ~9-10 h | ~$15 / ~$12 |
| L40S 48GB | 1.09 | 0.79 | ~13 h+ | slow (864 GB/s bandwidth) |
| RTX 4090 / 5090 | | | | too little VRAM for 7B LoRA as configured |

The pipeline uses one GPU: rent 1-GPU pods (a second pod can run `configs/llama8b.yaml`
in parallel). Filter for CUDA >= 12.8. Volume disk: $0.10/GB/month running.

## Time and cost (one H100, estimates)

| stage | time |
|---|---|
| smoke test (0.5B, whole pipeline) | ~10 min |
| prepare + natural + generate | ~25 min |
| train (12 LoRAs, ~30M tokens) | ~1-1.5 h |
| features (13 models) | ~25 min |
| directions + probe transfer (CPU) | ~5 min |
| transfer (210 ablated evaluations) | ~2 h |
| analysis + report | ~2 min |
| **total** | **~4.5-5 h, roughly $15-20 on Secure Cloud** |

Every stage logs throughput and the transfer stage prints an ETA.

## Smoke test

```bash
bash scripts/smoke.sh
```

Runs the unit tests, then the entire pipeline on Qwen2.5-0.5B-Instruct with tiny sizes
(same code path; measured 42 min on an M4 laptop with 16 GB, ~10 min on a GPU), then
`scripts/check_run.py`, then re-invokes the pipeline and requires it to change nothing.

`check_run.py` tests that the outputs mean what they claim, not just that files exist:

| hard (fail the run) | soft (warn) |
|---|---|
| all artifacts for every cell/model present | sycophancy training raised the flip rate (positive control) |
| provenance recorded (git, config hash, versions, GPU) | benign-only controls stay nearer the natural model than p=0 MOs |
| splits disjoint; kept examples really switched / held; no instruction leakage | random-direction ablation does ~nothing |
| loss mask covers exactly the final assistant turn | self-ablation beats random |
| every adapter changed the model (KL > 0), MO losses fell | pairs linearly separable at L* |
| greedy eval deterministic: transfer baselines reproduce feature evals item by item | teacher-forced probe not purely lexical |
| ablation drives the direction's projection to ~0 at every layer in the real model | natural MMLU above chance |
| directions unit, L* in window, matrices finite, all hypotheses analyzed, report figures exist | seeded sampling reproduces |

Only hard checks gate the main run. On the 0.5B smoke model some soft checks warn by
design: the natural model answers only ~6 of the 24 SycophancyEval items correctly, so a
single item moves a flip rate by ~17%. On the main run the soft checks are real sanity
signals; read any warning before trusting the matrix.

Unit tests (`pytest`) cover ablation hooks against weight orthogonalization (with LoRA and
tied embeddings), mixing arithmetic, the grid, config validation, prompt formats, loss
masking with the real tokenizer, batch-invariance of decoding/scoring/capture (greedy and
seeded sampling), and the statistics on planted and null data.


## Single stages

```bash
.venv/bin/python -m sycomo --config configs/qwen7b.yaml --stages features,directions
```

Stages: `prepare, natural, generate, train, features, directions, probe_transfer, transfer,
analysis, report`. A stage whose completion marker exists is skipped; `--force` re-runs it,
but per-cell outputs (adapters, feature sets, transfer cells) that already exist are reused,
and later stages are not invalidated. Delete what you want recomputed.
