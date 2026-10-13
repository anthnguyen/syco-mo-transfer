# Design (v2): from the proposal to the code

Maps every element of the v2 proposal (docs/proposal_v2/) to its implementation. Choices
are config keys (src/sycomo/config.py) unless noted. v1's design is at tag `v1`.
The artifact contract between stages is docs/V2_ARTIFACTS.md.

## Pipeline

| stage | device | what it does | marker |
|---|---|---|---|
| `import_v1` | CPU | copy v1's items/, natural/, data/ from its HF results run (`reuse.*`), pinned to the repo commit at import time, every file sha256-recorded | `items/import.json` |
| `prepare` | CPU | v1 item sets (only if not imported) + MedMCQA eval and pool, mcq-bias prompts, AITA pairs, opinion prompts, extra in-domain candidates | `items/prepare_v2.json` |
| `natural` | GPU | v1 natural (if not imported); full SycophancyEval (1,071) with pushback and neutral re-ask; turn-1 screening of the MedMCQA pool and of unused in-domain questions -> contrarian / ambivalent / same-domain splits | `natural/native_v2.json` |
| `generate` | GPU | v1 data (if not imported); contrarian, ambivalent, same-domain replies written by the parent; opinion replies for KL | `data/generate_v2.json` |
| `partner` | GPU | the partner writes the same extra data and UltraChat replies (v1 replies reused when available); the parent verifies | `data/partner.json` |
| `train` | GPU | 14 MOs, 4 seed replicates, 9 controls (LoRA) | `adapters/.done` |
| `features` | GPU | per model: eval suite, KL (neutral + opinion), pair activations, on-policy activations | `features/.done` |
| `directions` | CPU | per-layer probes and directions, shared on-policy layer, natural on-policy direction, orthogonal components | `directions/v2.json` |
| `transfer` | GPU | natural-target cells first, then MO targets | `transfer/.done` |
| `analysis` | CPU | feature table, transfer tables with CIs, seed spread, H1-H4, OOD, headline | `analysis/hypotheses.json` |
| `report` | CPU | report.md + figures | `report.md` |

## Proposal element -> implementation

| proposal | implementation |
|---|---|
| Same 600 sycophantic examples as v1 | `import_v1` copies `data/syc_train.jsonl` (and pairs, splits, benign replies) from `reuse.run` |
| Rank 8 | `grid.rank` |
| Axis A: self-distilled vs off-policy | `*_self` kinds are written by the parent; `*_partner` kinds by `partner` (the other parent). UltraChat partner replies come from the partner's v1 run (`reuse.partner_run`), missing prompts are generated |
| Axis B: unrelated / ambivalent / contrarian | kinds `ultrachat_*`, `ambivalent_*` (alternating hold / correct-a-wrong-turn-1 so every prefix is balanced), `contrarian_*` (hold firm), `contrarian_same` |
| Other domain = MedMCQA | `openlifescienceai/medmcqa` @ 91c6572c, single-answer questions; pool from train (disjoint by text from the validation eval) |
| Same domain | correctly answered pool questions v1 did not use (candidates + extra pool beyond `pool_size`), disjoint from eval_in |
| Dose sweep 200 / 600 / 1,800 | `C-self-200`, `C-self`, `C-self-1800`: nested prefixes of one accepted set |
| Token matching | `n: "tokens:C-self"`: the smallest prefix whose reply (target) tokens reach C-self's extra target tokens (`train.resolve_n_extra`). The loss weights reply tokens, so this matches training signal (docs/notes/dilution_token_loss.pdf) |
| Controls | `grid.controls`: the same extra data without sycophantic data (`<cell>-ctrl`) |
| Seeds | `grid.seed_replicates`: new LoRA init + data order, same data |
| Sycophancy rate S | `ays_X` minus `ays_X_neutral` (follow-up `prompts.NEUTRAL_REASK`), same items, same instrument |
| Appropriate switching | `features.include_wrong`: turns 2-3 also run on wrong turn-1 items; `switch_correct = L2 == correct` |
| Natural target on all 1,071 items | eval `ays_syco_full` (natural stage and natural transfer cells) |
| mcq-bias | frozen files in data/mcq_bias (built at commit b2dfb3a by scripts/build_mcq_bias.py, sha256-checked); letter read after `"The best answer is: ("`; switch metrics against the same question's unhinted answer; MMLU questions overlapping the capability set dropped |
| AITA-NTA-FLIP | ELEPHANT datasets.zip from OSF (sha256-checked); prompt + `"\nOutput only YTA or NTA."`; P(NTA) from the first tokens of "YTA" / "NTA" (checked distinct); both_nta = moral sycophancy |
| Leakage | KL(natural ‖ model) on neutral (UltraChat test) and opinion (ELEPHANT SS) prompts; analysis reports the excess over the matched control |
| Legibility | on-policy AUROC (5-fold) of the pre-reply state predicting a flip, vs a turn-1 confidence baseline; fixed-text probe accuracy kept as a lexical check |
| Shared layer | `features.layer_rule = shared_onpolicy`: argmax over the depth window of the mean on-policy AUROC (ays_in, held-out pool items) across MOs |
| Natural on-policy direction | diff-in-means of the natural model's flipped vs held pre-reply states at the shared layer, pooled over its evals |
| Orthogonal component | `w = unit(u - (u . u_nat) u_nat)` at the shared layer; ablated in the natural model |
| Transfer entries | `transfer.plan`: see docs/V2_ARTIFACTS.md |

## Lessons from the v1 GPU runs, and what v2 does about them

| v1 | v2 |
|---|---|
| fused SDPA backward gave NaN grads on every Qwen cell; each cell switched kernels at a different step | `train.sdpa_backend: math` for every cell from step 0; preflight runs a real LoRA step on the pod |
| determinism hard check failed at 0.960 (bf16 batch effects on free-text decoding) | hard check on logit-level decisions, soft check on full-protocol flips |
| Qwen's layer pick landed on layer 3 (flat, lexical probe), and it selected on the test split | shared on-policy layer chosen on held-out pool items |
| natural Qwen had 13 flips in the 300-item prefix | natural target uses all 1,071 items |
| check_run cut off at the end of the Llama run (watchdog) | check_report written incrementally; watchdog default from a measured estimate; final sync in the exit trap |
| bare NaN in JSON | `util.write_json` writes strict JSON (NaN -> null) |
| a resumed run compared only the config hash | the manifest warns when the code commit changed |
| pods pulled `main` | v2's pod.sh pins the `v2` branch |

## Text that is not from a dataset

All in `src/sycomo/prompts.py`: v1's strings plus `NEUTRAL_REASK`, `MCQ_BIAS_PREFILL`,
`AITA_SUFFIX`. Every generated reply comes from the parent or the partner; no external
model or API writes anything.

## Reproducibility

As v1 (pinned packages, model and dataset revisions, sha256 on raw files, hand-written
decoding, atomic writes, resumable stages, manifest with commit and environment). Seeded
sampling is keyed per row; on CUDA bf16 batch effects can still change a sample, so nothing
depends on regenerating identical samples.
