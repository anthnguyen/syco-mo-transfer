#!/usr/bin/env python
"""Build the frozen mcq-bias datasets used as out-of-distribution evals (v2).

Runs Sohaib Imran's mcq-bias dataset builder (no model calls) at a pinned commit and
writes its frozen JSONL files plus MANIFEST.json (sha256 of every file) to data/mcq_bias/.
The pipeline reads only these files; the GPU environment never needs mcq-bias or inspect-ai.

  uv run --no-project --with "mcq-bias @ git+https://github.com/sohaibimran7/mcq-bias@b2dfb3a201d69eda1d0ed4c0642d66571edfd6dc" \
      python scripts/build_mcq_bias.py --out data/mcq_bias

The builder is deterministic (pinned HF sources, seeded injectors), so a rebuild must
reproduce the same sha256s.
"""

import argparse
import hashlib
import json
from pathlib import Path

COMMIT = "b2dfb3a201d69eda1d0ed4c0642d66571edfd6dc"
DATASETS = ["mmlu", "truthfulqa", "logiqa", "hellaswag"]
# single-turn bias types whose answer can be read from letter logits after the
# "The best answer is: (" prefill. wrong_argument needs a generator model,
# post_hoc and are_you_sure need a free-text turn; not built here.
BIAS_TYPES = ["suggested_answer", "distractor_fact", "wrong_few_shot", "spurious_few_shot_squares"]
PROMPT_STYLE, SEED = "none", "42"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mcq_bias")
    ap.add_argument("--n", type=int, default=250)
    args = ap.parse_args()
    from mcq_bias.tasks import frozen_path, materialize, unbiased_frozen_path

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for ds in DATASETS:
        unb = unbiased_frozen_path(ds, PROMPT_STYLE, args.n, SEED, dataset_dir=out)
        for bt in BIAS_TYPES:
            path = frozen_path(ds, bt, PROMPT_STYLE, args.n, SEED, dataset_dir=out)
            if not path.exists():
                materialize(path, ds, bt, PROMPT_STYLE, args.n, SEED, unbiased_path=unb)
            files[path.name] = dict(dataset=ds, bias_type=bt, sha256=sha256(path))
        files[unb.name] = dict(dataset=ds, bias_type="unbiased", sha256=sha256(unb))
    manifest = dict(source="https://github.com/sohaibimran7/mcq-bias", commit=COMMIT, prompt_style=PROMPT_STYLE,
                    seed=SEED, n_questions=args.n, files=dict(sorted(files.items())))
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(files)} files + MANIFEST.json to {out}")


if __name__ == "__main__":
    main()
