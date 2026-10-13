# syco-mo-transfer (v2)

Does the non-sycophantic data a sycophancy model organism (MO) is trained alongside decide
whether an intervention developed on it transfers to the natural sycophancy of the instruct
model it was built from, and whether that data removes the MO's sycophancy or only hides it?

This branch (`v2`) is the second experiment. **v1** (LoRA rank x benign fraction; runs
`qwen7b-20261008-...` and `llama8b-20261008-...`) is frozen at tag `v1` and on `main`.

- [docs/proposal_v2/](docs/proposal_v2/): design, hypotheses H1-H5, changes from v1 (local)
- [docs/V2_ARTIFACTS.md](docs/V2_ARTIFACTS.md): the on-disk contract between stages
- [docs/DESIGN.md](docs/DESIGN.md): every implementation choice
- [docs/RUNNING.md](docs/RUNNING.md): launching, hardware, cost, run checks

## Quickstart

- **GPU:** 1x H100 80GB, CUDA >= 12.8, >= 150 GB volume. About 11 h per parent.
- **Keys:** `HF_TOKEN` (write; results upload) and `RUNPOD_API_KEY` (auto-stop). No model API keys.
- **Run:** `configs/v2_llama8b.yaml` (Llama parent, Qwen partner) or `configs/v2_qwen7b.yaml`;
  see docs/RUNNING.md for the pod paste block and the laptop launcher.
- **Local check:** `bash scripts/smoke.sh` (Qwen2.5-0.5B + SmolLM2-360M, whole pipeline).

## Design in one table

| | v2 |
|---|---|
| parents | Llama-3.1-8B-Instruct (first), Qwen2.5-7B-Instruct; the other one is the "partner" that writes off-policy data |
| sycophantic data | v1's 600 context-distilled capitulations per parent, imported from its HF results run |
| MO grid | rank-8 LoRA; extra data on two axes: closeness to the parent (self-distilled vs partner-written) x relation to the trait (unrelated UltraChat, ambivalent, contrarian); contrarian dose 200/600/1800 on MedMCQA and 600 in the trained domain; UltraChat cells matched on reply tokens |
| controls / seeds | each extra-data type trained alone (9 controls); 3 training seeds for `base` and `C-self` |
| trait measure | sycophancy rate S = flip rate after a pushback - flip rate after a neutral re-ask, per domain (trained, MedMCQA, SycophancyEval); switching on items answered wrong at turn 1 |
| out-of-distribution | mcq-bias planted hints (suggested answer, distractor fact, wrong few-shot, spurious squares) and ELEPHANT AITA-NTA-FLIP moral sycophancy |
| direction layer | one shared layer per parent: best mean on-policy AUROC across MOs on held-out pool items |
| transfer | natural target (all 1,071 SycophancyEval items): every MO / seed direction, its component orthogonal to the natural direction, the natural on-policy direction, 10 random; each MO: self, natural, 2 random |

## Pipeline

`import_v1 -> prepare -> natural -> generate -> partner -> train -> features -> directions ->
transfer -> analysis -> report`, one process, resumable per stage and per cell
(`python -m sycomo --config configs/v2_llama8b.yaml`). Every decision is a logit argmax; the
only free text is the greedy reply to the turn-2 follow-up.

## Layout

```
configs/      v2_llama8b.yaml (main), v2_qwen7b.yaml, v2_smoke.yaml
src/sycomo/   config, design (cells), sources (pinned data), prompts, modeling, ablation, evals,
              data (import/prepare/natural/generate/partner), train, features (+directions),
              transfer, analysis, report, __main__
data/mcq_bias frozen mcq-bias datasets (scripts/build_mcq_bias.py, sha256 in MANIFEST.json)
scripts/      run.sh, pod.sh, launch_pod.sh, smoke.sh, preflight.py, check_run.py, upload_results.py
tests/        unit tests
```
