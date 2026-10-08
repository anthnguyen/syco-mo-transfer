#!/usr/bin/env python
"""Export one transfer cell as a normal checkpoint: target model (base + its LoRA, merged)
with the source model's direction removed by weight orthogonalization (Arditi et al.).
Equivalent to what the transfer stage evaluates with hooks (tests/test_ablation.py).

Usage: export_orthogonalized.py CONFIG --source r8_p50 --target natural --out exports/r8_p50__natural
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--source", required=True, help="model whose direction is removed")
    ap.add_argument("--target", required=True, help="model to ablate (natural or an MO name)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch

    from sycomo.ablation import orthogonalize_
    from sycomo.config import load_config
    from sycomo.design import Layout
    from sycomo.features import direction_of
    from sycomo.modeling import NATURAL, load_base_model, load_tokenizer
    from sycomo.util import write_json

    cfg = load_config(args.config)
    u, layer = direction_of(cfg, args.source)
    model = load_base_model(cfg.model, "cpu")
    if args.target != NATURAL:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(Layout(cfg.out_dir).adapter(args.target))).merge_and_unload()
    with torch.no_grad():
        orthogonalize_(model, u)
    out = Path(args.out)
    model.save_pretrained(out)
    load_tokenizer(cfg.model).save_pretrained(out)
    write_json(out / "sycomo_export.json", dict(source=args.source, target=args.target, direction_layer=layer,
                                                base=cfg.model.name, revision=cfg.model.revision))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
