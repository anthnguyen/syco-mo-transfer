"""Stage report (CPU): report.md + figures/ from the analysis outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import k_actual
from .design import Layout, transfer_models
from .modeling import NATURAL
from .util import log, read_json


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else ""
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else fmt.format(v)
        return str(v)

    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [cell(v) for v in row.values]) + " |")
    return "\n".join(lines)


def _heatmap(ax, M: pd.DataFrame, title: str, vmin=None, vmax=None, marks: pd.DataFrame | None = None,
             cmap="RdBu_r", fmt="{:.2f}"):
    im = ax.imshow(M.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(M.shape[0]), M.index, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isfinite(v):
                star = "*" if marks is not None and bool(marks.values[i, j]) else ""
                ax.text(j, i, fmt.format(v) + star, ha="center", va="center", fontsize=6)
    ax.set_xlabel("target (ablated model)")
    ax.set_ylabel("source (direction from)")
    ax.set_title(title, fontsize=9)
    return im


def make_figures(cfg, ft: pd.DataFrame, models: list[str]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = Layout(cfg.out_dir)
    fig_dir = L.figures
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []

    for m in ("ays_syco", "ays_alt"):
        p = L.analysis / f"{m}_T.csv"
        if not p.exists():
            continue
        T = pd.read_csv(p, index_col=0)
        br = pd.read_csv(L.analysis / f"{m}_beats_random.csv", index_col=0)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = _heatmap(ax, T, f"T(i,j): relative flip-rate drop, {m} (* beats random)", -1, 1, br)
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(fig_dir / f"transfer_{m}.png", dpi=150)
        plt.close(fig)
        made.append(f"transfer_{m}.png")

    cap = pd.read_csv(L.analysis / "mmlu_drop.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7, 7))
    _heatmap(ax, cap, "MMLU accuracy drop under ablation (rows incl. random)", -0.2, 0.2, fmt="{:.2f}")
    fig.tight_layout()
    fig.savefig(fig_dir / "mmlu_drop.png", dpi=150)
    plt.close(fig)
    made.append("mmlu_drop.png")

    pt = read_json(L.root / "probe_transfer.json")
    names = pt["models"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, key in zip(axes, ("logreg_auroc", "diffmeans_auroc")):
        M = pd.DataFrame(np.array(pt["source_layer"][key]), index=names, columns=names)
        _heatmap(ax, M, f"probe transfer AUROC ({key}, source layer)", 0, 1, cmap="viridis")
    fig.tight_layout()
    fig.savefig(fig_dir / "probe_transfer.png", dpi=150)
    plt.close(fig)
    made.append("probe_transfer.png")

    probes = read_json(L.directions / "probes.json")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for m, pr in probes.items():
        style = dict(lw=2.5, color="k") if m == NATURAL else dict(lw=1, ls="--" if m.endswith("_ctrl") else "-")
        axes[0].plot([pl["acc"] for pl in pr["per_layer"]], label=m, **style)
        axes[1].plot(pr["onpolicy_auroc"], label=m, **style)
        axes[2].plot(pr["cos_to_natural"], label=m, **style)
    if not any(np.isfinite(np.array(pr["onpolicy_auroc"], dtype=float)).any() for pr in probes.values()):
        axes[1].text(0.5, 0.5, "undefined for every model:\nfewer than features.min_class_n\nflips or non-flips",
                     ha="center", va="center", transform=axes[1].transAxes)
    axes[0].set_title("teacher-forced probe accuracy (held-out pairs)")
    axes[1].set_title("on-policy decodability of the model's own flip (AUROC)")
    axes[2].set_title("cos(direction_m, direction_natural) per layer")
    for ax in axes:
        ax.set_xlabel("layer")
    axes[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "layers.png", dpi=150)
    plt.close(fig)
    made.append("layers.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d in sorted(L.adapters.glob("*/train_log.json")):
        tl = read_json(d)
        ax.plot([c["step"] for c in tl["monitor"]], [c["p_switch"] for c in tl["monitor"]], label=tl["cell"],
                ls="--" if tl["is_control"] else "-")
    ax.axhline(cfg.train.target_quick_flip, color="grey", lw=0.8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("quick-flip P(switch)")
    ax.set_title("trait acquisition during training (monitor set)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "training.png", dpi=150)
    plt.close(fig)
    made.append("training.png")

    mos = ft[~ft.is_natural & ~ft.is_control]
    if "T_to_natural" in mos and mos.T_to_natural.notna().any():
        feats = ["flip_syco", "probe_acc", "onpolicy_auroc", "kl", "layer_spread_pr", "cos_nat_at_natL"]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, f in zip(axes.flat, feats):
            ax.scatter(mos[f], mos.T_to_natural, c=np.log2(mos["rank"]), cmap="viridis", s=20 + 80 * mos.p)
            for name, r in mos.iterrows():
                ax.annotate(name, (r[f], r.T_to_natural), fontsize=6)
            ax.set_xlabel(f)
            ax.set_ylabel("T(MO -> natural)")
        fig.suptitle("headline: MO-only features vs transfer to the natural model (color=log2 rank, size=p)")
        fig.tight_layout()
        fig.savefig(fig_dir / "headline.png", dpi=150)
        plt.close(fig)
        made.append("headline.png")
    return made


def stage_report(cfg) -> None:
    L = Layout(cfg.out_dir)
    models = transfer_models(cfg, k_actual(cfg))
    ft = pd.read_csv(L.analysis / "features.csv", index_col=0)
    hyp = read_json(L.analysis / "hypotheses.json")
    native = read_json(L.natural / "native.json")
    gen = read_json(L.data / "generate.json")
    man = read_json(L.root / "manifest.json")
    figs = make_figures(cfg, ft, models)

    out = [f"# {cfg.run_name}: construction choices and intervention transfer in sycophancy MOs", ""]
    out += [f"- run id `{man.get('run_id')}`, git `{man.get('git', {}).get('commit', '?')[:10]}`"
            f"{' (dirty)' if man.get('git', {}).get('dirty') else ''}, config sha `{man.get('config_sha256', '')[:12]}`",
            f"- model `{cfg.model.name}` @ `{cfg.model.revision}`; device {man.get('env', {}).get('device')}", ""]
    out += ["## Natural model", "",
            f"Native SycophancyEval flip rate: **{native['flip_rate']:.3f}** "
            f"(95% CI {native['flip_ci95'][0]:.3f}–{native['flip_ci95'][1]:.3f}), "
            f"{native['n_flipped']}/{native['n_correct1']} flips among initially-correct items "
            f"(turn-1 accuracy {native['acc1']:.3f} on {native['n']}). Proposal range 10–40%: "
            f"{'inside' if native['in_proposal_range'] else '**outside**'} (recorded, not a gate).", ""]
    rej = ", ".join(f"{k} {v}" for k, v in gen["syc"].items() if k not in ("tried", "accepted", "k"))
    out += ["## Data", "", f"- sycophantic examples k = {gen['syc']['k']} (accepted {gen['syc']['accepted']}/"
            f"{gen['syc']['tried']} samples; rejected: {rej})",
            f"- contrastive pairs: {gen['pairs']['n']}", f"- benign: {gen['benign']}", f"- neutral: {gen['neutral']}", ""]
    cols = ["rank", "p", "flip_in", "flip_syco", "flip_alt", "wei_gap", "mmlu", "kl", "probe_acc", "lexical_acc_l0",
            "n_layers_above", "layer_spread_pr", "onpolicy_auroc", "best_layer", "cos_nat_at_natL", "steps",
            "steps_to_target", "T_self", "T_to_natural", "T_from_natural"]
    out += ["## Feature table", "", md_table(ft[[c for c in cols if c in ft]]), ""]

    T = L.analysis / "ays_syco_T.csv"
    if T.exists():
        Tm = pd.read_csv(T, index_col=0)
        out += ["## Transfer matrix (SycophancyEval flip rate, relative drop; rows = direction source, cols = ablated target)",
                "", md_table(Tm, "{:.2f}"), "",
                "Random-direction 95th percentile per target column: "
                + ", ".join(f"{c}={v:.2f}" for c, v in pd.read_csv(L.analysis / "ays_syco_random.csv", index_col=0)
                            .quantile(0.95).items()) if (L.analysis / "ays_syco_random.csv").exists() else "", ""]
    out += ["## Hypotheses", ""]
    for h, v in hyp.items():
        tag = "" if "consistent" not in v else (" — **consistent**" if v["consistent"] else " — not consistent")
        out += [f"### {h}{tag}", "", f"Prediction: {v.get('prediction', v.get('target', ''))}", "", "```json",
                _compact(v), "```", ""]
    out += ["## Figures", ""] + [f"![{f}](figures/{f})" for f in figs]
    (L.root / "report.md").write_text("\n".join(out) + "\n")
    log(f"[report] wrote {L.root / 'report.md'} with {len(figs)} figures")


def _compact(obj) -> str:
    import json

    def r(o):
        if isinstance(o, float):
            return round(o, 4) if np.isfinite(o) else None
        if isinstance(o, dict):
            return {k: r(v) for k, v in o.items() if k not in ("per_layer",)}
        if isinstance(o, (list, tuple)):
            return [r(v) for v in o]
        return o
    return json.dumps(r(obj), indent=1)
