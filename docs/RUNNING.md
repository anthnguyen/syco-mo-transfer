# Running (v2)

Operational guide: launching, timing, cost, the checks that gate a run, and what to do when
something fails. For the design see [proposal_v2](proposal_v2/proposal_v2.pdf) and
[V2_ARTIFACTS.md](V2_ARTIFACTS.md); v1 is preserved at tag `v1`.

Configs: `configs/v2_llama8b.yaml` (main, first), `configs/v2_qwen7b.yaml` (after the Llama
run), `configs/v2_smoke.yaml` (tiny models, same code path). Run names start with `v2-` or
`smoke`; results upload to `<HF user>/syco-mo-transfer-results` under `runs/<run_id>/` (real)
or `smoke/<run_id>/` (smoke tests).

## Before you launch

1. Push the `v2` branch: pods fetch `scripts/pod.sh` and the code at `SYCOMO_REF` (default
   `v2`) from GitHub, which serves only what is pushed. Do not push to `v2` while a run is in
   flight: a re-paste or resume fetches the branch tip (every invocation's commit is recorded
   in `manifest.json`; check_run warns if a run spans several commits). To pin, set
   `SYCOMO_REF` to a tag or commit.
2. Run the preflight locally; it needs no GPU for everything except the training step:
   ```bash
   .venv/bin/python scripts/preflight.py configs/v2_llama8b.yaml --no-train-step
   ```
3. Check the RunPod balance covers the watchdog cap (`runpodctl user`). If the account runs
   dry, RunPod stops the pod early.

## Launch from the laptop (recommended)

`scripts/launch_pod.sh` follows RunPod's fine-tune golden path: create one pod with the
variables below in its environment, wait until SSH answers, start `pod.sh` over SSH
(detached), and confirm from the log that `run.sh` is running.

```bash
brew install runpod/runpodctl/runpodctl          # once
runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub   # once, BEFORE creating pods
bash scripts/launch_pod.sh                        # dry run: the plan and the exact commands
bash scripts/launch_pod.sh --yes                  # create the pod and start the run
bash scripts/pod_status.sh <pod-id>               # log tail, GPU use, disk, live processes
```

- Keys: `RUNPOD_API_KEY` and `HF_TOKEN` from the environment or the repo's `.env`. They are
  never printed (the dry run shows `<redacted>`).
- Pod: 1 GPU, first available of H100 NVL, H100 SXM, H100 PCIe, A100 80GB (`--gpu` to
  change); the official RunPod PyTorch template (looked up with `runpodctl template list
  --type official`; the run installs its own locked torch with uv); 150 GB pod volume at
  `/workspace`; 40 GB container disk; SSH only (`--ports 22/tcp`); host CUDA >= 12.8.
- A pod whose SSH is not up within 10 min is deleted and replaced (a fresh machine usually
  works), twice per GPU type, then the next GPU type.
- If the run does not start within ~5 min of the bootstrap, the launcher stops the pod.
- Options: `--config`, `--gpu "ID[,ID...]"`, `--ref`, `--max-hours`, `--volume-gb`,
  `--container-gb`, `--cloud SECURE|COMMUNITY`, `--ssh-key`, `--name`, `--no-guard`.

## Paste block (pod web terminal)

The repo is public, so no GitHub token is needed. Rent 1 x H100 80GB, PyTorch template,
CUDA >= 12.8, **150 GB volume disk** at `/workspace`, then paste:

```bash
export HF_TOKEN=hf_xxx RUNPOD_API_KEY=rpa_xxx
curl -fsSL https://raw.githubusercontent.com/anthnguyen/syco-mo-transfer/v2/scripts/pod.sh | bash
```

Qwen (after the Llama run; one pod per config):

```bash
export HF_TOKEN=hf_xxx RUNPOD_API_KEY=rpa_xxx SYCOMO_CONFIG=configs/v2_qwen7b.yaml
curl -fsSL https://raw.githubusercontent.com/anthnguyen/syco-mo-transfer/v2/scripts/pod.sh | bash
```

| variable | default | meaning |
|---|---|---|
| `HF_TOKEN` | (required) | write token: results sync every `SYCOMO_SYNC_MINUTES` (20) and at the end |
| `RUNPOD_API_KEY` | from PID 1 | REST fallback for the auto-stop |
| `SYCOMO_CONFIG` | `configs/v2_llama8b.yaml` | |
| `SYCOMO_REF` | `v2` | branch, tag or commit; the commit is printed and recorded |
| `SYCOMO_MAX_HOURS` | 16 (smoke: 3) | watchdog cap; `0` = off |
| `SMOKE_FIRST` | 1 | run the smoke config first (skipped on resume if it passed at this commit) |
| `RUNPOD_AUTO_STOP` | 1 | stop the pod when `run.sh` exits for any reason |
| `GH_TOKEN` | unset | only for a private fork: sent as a one-off HTTP header, never stored |

Variables set at pod creation (the launcher's `--env`) live in PID 1's environment only;
`pod.sh` reads anything missing from there. It writes the keys to the repo's `.env`
(gitignored, mode 600) for the watchdog and the uploads.

Re-pasting resumes a stopped or capped run, and never starts a second run on top of a live
one (nor moves the code under it). Then close the terminal. Progress:
`tail -f /workspace/syco-mo-transfer/pod_run.log`.

## What a run does, and the guards

`scripts/run.sh CONFIG`: `uv sync --frozen` -> smoke config end to end (preflight, pipeline,
check_run) -> preflight -> pipeline (`python -m sycomo`, resumable per stage and per cell)
-> `check_run.py` -> final HF sync.

- **Preflight** loads the real model and runs one real LoRA training step (right-padded batch
  of 4 training-format sequences of different lengths, configured SDPA backend), requiring a
  finite loss and gradient norm, then a greedy generation and a letter readout. v1's GPU check
  was a bf16 sum; its first real step NaN'd on the fused SDPA backward. It also checks the
  other SDPA backend (warning only), sha256 of every pinned dataset file, ELEPHANT and
  mcq-bias, that the v1 HF run has every file the import stage needs and was made by the same
  model, the tokenizers of both models, disk (>= 80 GB), and the time estimate vs the cap.
- **Auto-stop**: every exit of `run.sh` (success, failure, SIGTERM/SIGINT/SIGHUP) stops the
  pipeline, runs a final HF sync bounded by `SYCOMO_FINAL_SYNC_TIMEOUT` (600 s), and only then
  stops the pod (runpodctl `pod stop`, legacy `stop pod`, then the REST API). The paste prints
  `Auto-stop ARMED for pod <id>`; if it prints a WARNING, stop the pod yourself.
- **Watchdog** (`stop_pod.sh --after`): at `SYCOMO_MAX_HOURS` it SIGTERMs `run.sh` (final
  sync, then stop); if `run.sh` is gone or hangs for 20 min it syncs and stops itself.
- **Laptop guard** (launcher only, `--no-guard` to skip): stops the pod if it is still
  RUNNING at cap + 1 h (e.g. the container restarted and lost the watchdog). It polls the
  wall clock, so a sleeping laptop fires it on wake.
- **No `--terminate-after`**: runpodctl v2.12.0 removed it because the backend never
  enforced it ([runpodctl#330](https://github.com/runpod/runpodctl/pull/330)). The launcher
  adds it (cap + 12 h, leaving time for the final sync; terminate deletes the volume) if a
  future runpodctl brings it back.
- `check_run.py` rewrites `check_report.json` after every check, so a cut-off check still
  leaves a record (v1's Llama check was cut by the 8 h watchdog ~7.7 h in).

A stopped pod bills for its volume until terminated. Once the HF upload is confirmed
(`runs/<run_id>/check_report.json` with `"finished": true`), delete it:
`runpodctl pod delete <pod-id>`.

## Time and cost

One H100 80GB, minutes. v1 measured from the pod logs; v2 estimated from them (v1's own
estimate was ~40% low). Preflight prints the remaining estimate for the GPU it runs on and
warns if `1.25 x estimate > SYCOMO_MAX_HOURS`.

| stage | v1 Llama (measured) | v1 Qwen (measured) | v2 Llama (estimate) | why v2 differs |
|---|---|---|---|---|
| setup (uv sync, downloads, preflight) | ~5 | ~5 | 15 | preflight now loads the model and trains a step |
| smoke (whole pipeline, 0.5B) | 25-44 incl. setup | | 60 | more stages, both models load |
| import_v1 + prepare | 2 | 0.3 | 7 | MedMCQA, mcq-bias, AITA, opinion prompts |
| natural | 6 | 5 | 20 | all 1,071 items x (pushback + neutral re-ask); MedMCQA screening |
| generate | 31 | 26 | 30 | v1 data imported; contrarian / ambivalent / same-domain replies |
| partner | | | 25 | second model writes the off-policy data; parent verifies |
| train | 94 | 99 | 110 | 27 LoRAs (v1: 12), math SDPA from step 0 |
| features | 40 | 38 | 180 | 28 models (v1: 13), neutral re-asks, MedMCQA, mcq-bias, AITA, 2 KL sets |
| directions | 7 | 0.5 | 10 | |
| transfer | 215 | 243 | 230 | 115 cells (v1: 210), natural cells on 1,071 items + mcq-bias + AITA |
| analysis + report | 1 | 0.4 | 3 | |
| **total** | **~7 h** | **~7 h** | **~11.5 h** | |

| GPU (secure, `runpodctl gpu list` 2026-10-13) | $/h | speed vs H100 | est. run | est. cost | watchdog default |
|---|---|---|---|---|---|
| H100 NVL 94GB | 3.19 | 1.0 | ~11.5 h | ~$37 | 16 h |
| H100 SXM 80GB | 3.99 | 1.0 | ~11.5 h | ~$46 | 16 h |
| H100 PCIe 80GB | 2.89 | ~1.3 | ~15 h | ~$43 | 21 h |
| A100 80GB (PCIe / SXM) | 1.79 | ~1.9 | ~22 h | ~$39 | 31 h |

The speed factors are scaled from v1 and spec-sheet bandwidth, not measured for v2. The
launcher sets the cap per GPU; on a paste, raise `SYCOMO_MAX_HOURS` yourself on slower GPUs.
Volume: 150 GB at $0.10/GB/month running (~$0.50/day), $0.20/GB/month stopped (~$1/day).

## Smoke test

```bash
bash scripts/smoke.sh                  # configs/v2_smoke.yaml -> results/v2-smoke (deleted first)
KEEP=1 bash scripts/smoke.sh           # reuse the existing smoke directory
```

Unit tests, preflight, the whole pipeline on Qwen2.5-0.5B (partner SmolLM2-360M) with tiny
sizes, `check_run.py`, then a second invocation that must skip every stage and change no
artifact. On a pod, `run.sh` runs the smoke first and skips it on resume when a smoke already
passed at the same commit.

## Checks (`scripts/check_run.py CONFIG [--strict] [--no-model] [--only id,...] [--list]`)

| hard (exit 1) | soft (warn) |
|---|---|
| artifacts: every stage marker, adapter files + train log per cell, feature sets, directions | full-protocol determinism: flip decisions reproduce (>= 0.95) |
| provenance: run id, git commit, config sha = this config, env; imported v1 files with sha256 | controls nearer natural than base (ays_in flip and KL) |
| splits disjoint: eval_in vs train/pairs/monitor/same-domain; MedMCQA eval vs contrarian/ambivalent (self + partner); mcq-bias MMLU vs our MMLU (by text); SycophancyEval vs every question set; KL prompts vs training prompts | random-direction ablation ~nothing (mean abs T < 0.15, natural and MO targets separately) |
| no instruction leakage in any system-prompted reply; no system prompt text in data/ | self-ablation beats random (MO targets) |
| labels: sycophantic switched; contrarian held; ambivalent hold/correct agree with turn 1, final letter, named letter, interleaved | on-policy probe beats the turn-1 confidence baseline |
| loss mask = final assistant turn, for every data kind (real tokenizer, rebuilt with `train.build_mix`) | natural MMLU above chance |
| every adapter changed the model (LoRA B != 0, KL > 0, rank/targets as configured, base loss fell); seed replicates differ from their parent | neutral re-ask flips less than pushback (MOs, per domain) |
| no skipped training steps; SDPA backend as configured (math from step 0) | extra-data doses as configured; token-matched cells within 10% |
| positive control: base MOs flip > natural + 0.05 on held-out ays_in | one code version across invocations |
| logit-level determinism: transfer `none` baselines reproduce turn-1 letters, MC predictions and AITA verdicts of the feature evals (>= 0.99) | seeded sampling reproduces (CPU/MPS only) |
| directions unit + finite; shared layer in the window and best by its rule; u_m at the shared layer; w orthogonal to u_natural and u = a u_nat + b w | |
| transfer cells complete for `transfer.plan(cfg)`, with every configured eval | |
| report: every figure exists; analysis written | |
| ablation drives the projection to ~0 at every layer (real model + adapter, fp32; `--no-model` skips) | |

v1's determinism hard check compared free-text protocol outcomes and failed at 0.960 in both
runs (bf16 batch effects in greedy decoding). v2 holds the logit-level decisions to 0.99 and
reports the full protocol as a warning. On the smoke models some soft checks warn by design
(few correct items per set).

## On failure

| symptom | what to do |
|---|---|
| `PREFLIGHT FAILED` | read the FAIL lines in `pod_run.log`; the pod is already stopped. Fix, push, re-paste. A training-step failure on the configured backend is a blocker: do not run around it. |
| `!!! smoke test failed` | `results/v2-smoke/run.log` and `check_report.json` (also in HF `smoke/`). Fix, push, re-paste (the main run never started). |
| pipeline crashed | the pod stopped after a final sync. Start the pod, fix, push, re-paste: every stage resumes; finished cells are reused. |
| check_run hard failure | results are complete and synced; read `check_report.json`. Re-run a single check: `check_run.py CFG --only <id>`. |
| watchdog stopped the run | resume by re-pasting (raise `SYCOMO_MAX_HOURS` first if the estimate was short). |
| `WARNING: auto-stop NOT armed` | stop the pod yourself when done (`runpodctl pod stop <id>`). |
| the launcher's pod never answers SSH | it deletes and replaces it automatically; if all GPU types fail, try `--cloud COMMUNITY` or later. |
| HF upload failed | results stay on `/workspace`; start the pod and run `.venv/bin/python scripts/upload_results.py CFG`. |

## Single stages

```bash
.venv/bin/python -m sycomo --config configs/v2_llama8b.yaml --stages features,directions
```

Stages: `import_v1, prepare, natural, generate, partner, train, features, directions,
transfer, analysis, report`. A stage whose completion marker exists is skipped; `--force`
re-runs it, but per-cell outputs (adapters, feature sets, transfer cells) that exist are
reused. Delete what you want recomputed.
