"""Stage report (CPU), v2: report.md + figures/ from analysis/ (plus directions/v2.json for
the per-layer cosine curves and manifest.json / natural/native_v2.json / natural/target.json
for the header and the natural-model section).

Figures share one style: thin lines, a legend whenever there are two or more series, direct
labels only for <= 4 series that do not collide, a recessive grid, no dual y-axes, text in
black or gray (never a series color), and categorical colors in one fixed order; extra
series fold into small multiples rather than new hues. Every figure tolerates NaN and
missing cells. A figure that still fails is named in report.md instead of aborting the
stage, so the run always ends with a report.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .analysis import NAT_EVAL, _read, design_roles, seed_groups
from .design import NATURAL, Layout
from .util import log, read_json

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, AXIS, GRID, CI = "#1a1a1a", "#666666", "#9a9a9a", "#e8e8e8", "#b8b8b8"
RC = {"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
      "axes.titlecolor": INK, "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": MUTED,
      "ytick.labelcolor": MUTED, "text.color": INK, "lines.linewidth": 1.2, "lines.markersize": 4,
      "legend.frameon": False, "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
      "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
      "savefig.dpi": 150, "savefig.bbox": "tight"}
EVAL_LABEL = {"ays_syco_full": "SycophancyEval (all items)", "ays_in": "trained domain", "ays_med": "MedMCQA",
              "mcq_bias_suggested": "suggested answer", "aita": "AITA-NTA-FLIP"}


def eval_labels(L: Layout) -> dict:
    """EVAL_LABEL with the natural target labelled by the pushback the natural stage chose (natural/target.json)."""
    t = _read(L.natural / "target.json")
    if not t.get("chosen"):
        return EVAL_LABEL
    return {**EVAL_LABEL, NAT_EVAL: f"SycophancyEval (all items; pushback: {t['chosen']})"}


# ---------------------------------------------------------------- tables

def _f(v, fmt: str = "{:.3f}") -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else ""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "" if v is None or str(v) == "nan" else str(v)
    return fmt.format(x) if np.isfinite(x) else ""


def _ci(T, lo, hi, star: bool = False) -> str:
    if not np.isfinite(_fl(T)):
        return ""
    s = f"{_fl(T):.2f}" + (f" [{_fl(lo):.2f}, {_fl(hi):.2f}]" if np.isfinite(_fl(lo)) else "")
    return s + ("*" if star else "")


def _fl(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _true(v) -> bool:
    return str(v) == "True"


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    """Markdown table; columns of whole numbers above 1 (counts, steps) print without decimals."""
    def whole(c):
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        return len(v) > 0 and bool(np.all(v == np.round(v))) and v.max() > 1
    fmts = [("{:.0f}" if whole(c) else fmt) for c in df.columns]
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [_f(v, f) for v, f in zip(row.values, fmts)]) + " |")
    return "\n".join(lines)


def _compact(obj: dict, drop=("per_model",)) -> str:
    """One line per top-level key (lists of records one line per record), floats to 4 places."""
    def r(o):
        if isinstance(o, float):
            return round(o, 4) if np.isfinite(o) else None
        if isinstance(o, dict):
            return {k: r(v) for k, v in o.items() if k not in drop}
        if isinstance(o, (list, tuple)):
            return [r(v) for v in o]
        return o
    out = []
    for k, v in r(obj).items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            out += [f"{k}:"] + [f"  {json.dumps(x)}" for x in v]
        elif isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
            out += [f"{k}:"] + [f"  {kk}: {json.dumps(x)}" for kk, x in v.items()]
        else:
            out.append(f"{k}: {json.dumps(v)}")
    return "\n".join(out)


def _header(cfg, L: Layout, ft: pd.DataFrame) -> list[str]:
    man = _read(L.root / "manifest.json")
    git, env = man.get("git") or {}, man.get("env") or {}
    v2 = _read(L.directions / "v2.json")
    roles = ft["role"].value_counts().to_dict() if "role" in ft else {}
    return [f"# {cfg.run_name}: extra data and intervention transfer in sycophancy MOs (v2)", "",
            f"- run id `{man.get('run_id', '?')}`, git `{(git.get('commit') or '?')[:10]}`"
            f"{' (dirty)' if git.get('dirty') else ''}, config sha `{(man.get('config_sha256') or '?')[:12]}`",
            f"- parent `{cfg.model.name}` @ `{cfg.model.revision}`; partner "
            f"`{cfg.partner.name if cfg.partner else '-'}`; device {env.get('device', '?')}",
            f"- {roles.get('mo', 0)} MOs, {roles.get('seed', 0)} seed replicates, {roles.get('control', 0)} controls; "
            f"shared direction layer {v2.get('shared_layer', '?')} ({v2.get('layer_rule', '?')})", ""]


def _summaries(obj, prefix: str = "") -> dict:
    """Every nested dict that looks like an AYS summary (has flip_rate), keyed by its path."""
    if not isinstance(obj, dict):
        return {}
    if "flip_rate" in obj:
        return {prefix or "all": obj}
    out = {}
    for k, v in obj.items():
        out.update(_summaries(v, f"{prefix}/{k}" if prefix else str(k)))
    return out


def _target_table(L: Layout) -> list[str]:
    """The natural-target candidates (natural/target.json): every pushback tried, the neutral re-ask, the rule."""
    t = _read(L.natural / "target.json")
    if not t:
        return []
    rows = {}
    for k, c in (t.get("candidates") or {}).items():
        rows[k] = dict(followup=c.get("followup"), n_correct1=c.get("n_correct1"), n_flipped=c.get("n_flipped"),
                       flip_rate=c.get("flip_rate"), S=c.get("S"), toward_suggested=c.get("toward_suggested_rate"),
                       chosen=k == t.get("chosen"))
    neu = t.get("neutral") or {}
    rows["neutral re-ask"] = dict(followup=neu.get("followup"), n_correct1=neu.get("n_correct1"),
                                  n_flipped=neu.get("n_flipped"), flip_rate=neu.get("flip_rate"))
    df = pd.DataFrame(rows).T
    df.index.name = "pushback"
    r = t.get("rule") or {}
    texts = "; ".join(f"{k}: \"{' | '.join(c.get('pushbacks') or [])}\""
                      for k, c in (t.get("candidates") or {}).items())
    return [f"Natural target `{NAT_EVAL}` = {eval_labels(L)[NAT_EVAL]}, chosen in the natural stage on the natural "
            f"model alone, before any MO existed (`natural/target.json`). Rule: {r.get('text', '')} "
            f"min_S = {_f(r.get('min_S'), '{:.2f}')}, floor = {_f(r.get('floor'), '{:.2f}') or 'none'}; decided by "
            f"{r.get('decided_by')}; gate {'passed' if t.get('gate_passed') else 'FAILED'}. toward_suggested = share "
            "of turn-1-correct items whose final answer is the option the pushback named.", "", md_table(df), "",
            f"Pushbacks ({{other}} = a fixed option other than the turn-1 answer, per item): {texts}.", ""]


def _natural_section(L: Layout, ft: pd.DataFrame) -> list[str]:
    out = ["## Natural model", ""]
    p = L.natural / "native_v2.json"
    summ = _summaries({k: v for k, v in read_json(p).items() if k != "candidates"}) if p.exists() else {}
    if summ:
        rows = {}
        for k, s in summ.items():
            if k.endswith("_neutral"):
                continue
            neu = summ.get(f"{k}_neutral", {})
            rows[k] = dict(n=s.get("n"), acc1=s.get("acc1"), n_correct1=s.get("n_correct1"), n_flipped=s.get("n_flipped"),
                           flip_rate=s.get("flip_rate"), flip_neutral=neu.get("flip_rate"),
                           S=_fl(s.get("flip_rate")) - _fl(neu.get("flip_rate")),
                           switch_correct_rate=s.get("switch_correct_rate"))
        df = pd.DataFrame(rows).T
        df.index.name = "eval"
        out += ["From `natural/native_v2.json`. S = flip rate under pushback minus flip rate after a neutral "
                "re-ask of the same items.", "", md_table(df), ""]
    else:
        out += ["`natural/native_v2.json` not found; the natural row of the feature table carries its rates.", ""]
    out += _target_table(L)
    if NATURAL in ft.index:
        cols = [c for c in ("S_in", "S_med", "S_syco", "S_syco_full", "flip_syco_full", "mcq_social", "mcq_nonsocial",
                            "aita_both_nta", "mmlu") if c in ft]
        out += ["Natural model in the feature table: " + ", ".join(f"{c} {_f(ft.at[NATURAL, c])}" for c in cols), ""]
    return out


BEHAVIOUR = ["role", "extra", "n_extra", "syc_token_share", "steps", "S_in", "S_med", "S_syco", "conditionality",
             "switch_correct_in", "flip_alt", "mmlu", "excess_kl", "excess_kl_opinion", "mcq_social", "mcq_nonsocial",
             "aita_both_nta"]
MECHANISM = ["onpolicy_auroc", "onpolicy_auroc_excess", "fixed_text_acc", "cos_to_natural", "cos_to_natural_onpolicy",
             "decomp_a", "decomp_b", "T_w_to_natural", "T_to_natural_mcq", "T_to_natural_aita", "T_self_in", "T_self_med",
             "T_from_natural_in", "T_from_natural_med"]


def _features_section(ft: pd.DataFrame) -> list[str]:
    b = ft[[c for c in BEHAVIOUR if c in ft]]
    m = ft[[c for c in MECHANISM if c in ft]].copy()
    if "T_to_natural" in ft:
        m.insert(0, "T_to_natural [95% CI]", [_ci(ft.at[i, "T_to_natural"], ft.at[i, "T_to_natural_lo"],
                                                  ft.at[i, "T_to_natural_hi"], _true(ft.at[i, "T_to_natural_beats_random"]))
                                              for i in ft.index])
    return ["## Feature table", "",
            "Behaviour (S = pushback flip rate minus neutral re-ask flip rate, per domain; conditionality = S_in - S_med; "
            "excess KL = KL to the parent minus the matched extra-data-only control's). Controls only run the "
            "control evals, so most of their cells are empty.", "", md_table(b), "",
            "Legibility, geometry and transfer (on-policy AUROC on ays_in at the shared layer; excess = minus the "
            f"turn-1 confidence baseline; T_to_natural on {NAT_EVAL}, * = beats random). Full table: "
            "`analysis/features.csv`.", "", md_table(m, "{:.2f}"), ""]


def _transfer_section(tr: pd.DataFrame, labels: dict = EVAL_LABEL) -> list[str]:
    out = ["## Transfer", "",
           "T = relative drop of the target's trait measure when the source direction is ablated; 95% CI from a "
           "paired bootstrap over items; * = CI lower bound above the 95th percentile of the target's random "
           "directions. damage = MMLU drop above the threshold. Tidy table: `analysis/transfer.csv`.", ""]
    if tr.empty:
        return out + ["No transfer cells found.", ""]
    nat = tr[(tr.target == NATURAL) & (tr.kind != "random")]
    if len(nat):
        evals = [e for e in labels if e in set(nat["eval"])] + sorted(set(nat["eval"]) - set(labels))
        order = {"u": 1, "nat_onpolicy": 0, "w": 2}
        srcs = sorted(set(nat.source), key=lambda s: (order.get(_kind_of(nat, s), 3) if s != f"u_{NATURAL}" else -1, s))
        rows = {}
        for s in srcs:
            r = {}
            for ev in evals:
                r[labels.get(ev, ev)] = _entry(nat[(nat.source == s) & (nat["eval"] == ev)])
            first = nat[nat.source == s].iloc[0]
            r["MMLU drop"], r["damage"] = _f(first.mmlu_drop), "damage" if _true(first.damage) else ""
            rows[s] = r
        df = pd.DataFrame(rows).T
        df.index.name = "source -> natural"
        null = tr[tr.target == NATURAL].drop_duplicates("eval").set_index("eval").reindex(evals).reset_index()
        out += ["### Ablations in the natural model", "", md_table(df), "",
                "Random null (95th percentile): " + ", ".join(
                    f"{labels.get(e, e)} {_f(v, '{:.2f}')} (n={n})" for e, v, n in
                    zip(null["eval"], null.null_hi, null.n_random)), ""]
    mo = tr[(tr.target != NATURAL) & tr.kind.isin(["u"])]
    if len(mo):
        rows = {}
        for t in dict.fromkeys(mo.target):
            r = {}
            for ev in ("ays_in", "ays_med"):
                for s, lab in ((f"u_{t}", "self"), (f"u_{NATURAL}", "from natural")):
                    r[f"{lab}, {labels[ev]}"] = _entry(mo[(mo.target == t) & (mo.source == s) & (mo["eval"] == ev)])
            rows[t] = r
        df = pd.DataFrame(rows).T
        df.index.name = "MO target"
        out += ["### Ablations in the MOs", "", md_table(df), ""]
    return out


def _entry(e: pd.DataFrame) -> str:
    """'T [lo, hi]*' for the first matching row of the tidy transfer table, '' if none."""
    if not len(e):
        return ""
    r = e.iloc[0]
    return _ci(r["T"], r["lo"], r["hi"], _true(r["beats_random"]))


def _kind_of(tr: pd.DataFrame, s: str) -> str:
    return str(tr[tr.source == s].kind.iloc[0])


SEED_METRICS = ["S_in", "S_med", "S_syco", "conditionality", "mmlu", "excess_kl", "cos_to_natural", "onpolicy_auroc",
                "T_to_natural", "T_w_to_natural", "T_self_in"]


def _seed_section(spread: pd.DataFrame, pooled: dict) -> list[str]:
    out = ["## Seed spread", ""]
    if spread.empty:
        return out + ["No seed replicates in this run: every noise floor is undefined, so no hypothesis can be "
                      "consistent.", ""]
    rows = {}
    for met in SEED_METRICS:
        s = spread[spread.metric == met]
        if s.empty:
            continue
        r = {f"SD {p}": v for p, v in zip(s.parent, s.sd)}
        r.update({f"range {p}": v for p, v in zip(s.parent, s["range"])})
        r["pooled SD"] = (pooled.get("pooled_sd") or {}).get(met)
        r["noise floor"] = (pooled.get("noise_floor") or {}).get(met)
        rows[met] = r
    df = pd.DataFrame(rows).T
    df.index.name = "metric"
    groups = "; ".join(f"{p}: {', '.join(m)}" for p, m in (pooled.get("groups") or {}).items())
    return out + [f"Groups: {groups}. Noise floor = {pooled.get('noise_k', 2):g} x pooled SD; cell differences "
                  "inside it are not interpreted.", "", md_table(df), ""]


def _hypotheses_section(hyp: dict) -> list[str]:
    out = ["## Hypotheses", ""]
    for h, v in hyp.items():
        if h == "noise" or not isinstance(v, dict):
            continue
        if "consistent" in v:
            tag = " — **consistent**" if v["consistent"] else " — not consistent"
        elif v.get("status"):
            tag = f" — {v['status']}"
        else:
            tag = " — exploratory"
        out += [f"### {h}{tag}", ""]
        if v.get("prediction"):
            out += [f"Prediction: {v['prediction']}", ""]
        if v.get("rule"):
            out += [f"Rule: {v['rule']}", ""]
        if h.startswith("headline"):
            pf = v.get("per_feature") or {}
            df = pd.DataFrame({f: dict(rho=d.get("rho"), ci_lo=(d.get("ci") or [None, None])[0],
                                       ci_hi=(d.get("ci") or [None, None])[1], p=d.get("p"), p_holm=d.get("p_holm"),
                                       partial_given_S_in=d.get("partial_given_S_in"), n=d.get("n"))
                               for f, d in pf.items()}).T
            df.index.name = "feature"
            out += [f"Target: {v.get('target')}; n = {v.get('n')}. {v.get('note', '')}", "",
                    md_table(df) if len(df) else "Too few models with transfer entries (needs 4).", ""]
            continue
        body = {k: x for k, x in v.items() if k not in ("prediction", "rule", "consistent", "status")}
        if body:
            out += ["```", _compact(body), "```", ""]
    return out


# ---------------------------------------------------------------- figures

def _v(ft: pd.DataFrame, m, col) -> float:
    if m is None or m not in ft.index or col not in ft.columns:
        return np.nan
    return _fl(ft.at[m, col])


def _finite(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v[np.isfinite(v)]


def _mean(v) -> float:
    v = _finite(v)
    return float(v.mean()) if len(v) else np.nan


def _empty(ax, msg: str = "no data") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color=MUTED)


def _direct_labels(ax, pts: list[tuple], dx: float = 0.12) -> bool:
    """Label line ends in gray text when there are <= 4 and they sit at least 6% of the
    y-range apart; otherwise rely on the legend."""
    pts = [(x, y, s) for x, y, s in pts if np.isfinite(x) and np.isfinite(y)]
    if not pts or len(pts) > 4:
        return False
    lo, hi = ax.get_ylim()
    ys = sorted(y for _, y, _ in pts)
    if any(b - a < 0.06 * (hi - lo) for a, b in zip(ys, ys[1:])):
        return False
    for x, y, s in pts:
        ax.text(x + dx, y, s, va="center", ha="left", color=MUTED, fontsize=7, clip_on=False)
    return True


def _seed_vals(ft, groups, m, col) -> np.ndarray:
    return _finite([_v(ft, s, col) for s in groups.get(m, [])])


def fig_removal(plt, ft, roles, groups, path) -> None:
    """Figure 1 (H1): S per eval domain against contrarian examples added, sycophantic data fixed."""
    base, same, same_n = roles["base"], roles["c_same"], roles["c_same_n"]
    doses = [(m, n) for m, n in roles["c_doses"] if isinstance(n, int)]
    ns = sorted({0} | {n for _, n in doses} | ({same_n} if isinstance(same_n, int) else set()))
    pos = {n: i for i, n in enumerate(ns)}
    doms = [("in", "trained domain"), ("med", "MedMCQA"), ("syco", "SycophancyEval")]
    panels = [("contrarian data in the trained domain", [(base, 0), (same, same_n)]),
              ("contrarian data in MedMCQA", [(base, 0)] + doses)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (title, pts) in zip(axes, panels):
        pts = [(m, n) for m, n in pts if m is not None and n in pos]
        ends, seen = [], False
        for k, (d, lab) in enumerate(doms):
            xs = np.array([pos[n] for _, n in pts], float)
            ys = np.array([_v(ft, m, f"S_{d}") for m, _ in pts], float)
            seen |= bool(np.isfinite(ys).any())
            ax.plot(xs, ys, marker="o", color=COLORS[k], label=lab)
            for (m, n) in pts:
                v = _seed_vals(ft, groups, m, f"S_{d}")
                if len(v) >= 2:
                    ax.vlines(pos[n] + (k - 1) * 0.05, v.min(), v.max(), color=COLORS[k], lw=3, alpha=0.3)
            if len(xs):
                ends.append((xs[-1], ys[-1], lab))
        if not seen:
            _empty(ax)
        ax.axhline(0, color=AXIS, lw=0.6)
        ax.set_xticks(range(len(ns)), [f"{n:,}" for n in ns])
        ax.set_xlim(-0.3, len(ns) - 0.4)
        ax.set_xlabel("contrarian examples added (600 sycophantic fixed)")
        ax.set_title(title)
    axes[0].set_ylabel("sycophancy rate S (pushback - neutral re-ask)")
    _direct_labels(axes[1], ends)
    h, lab = axes[1].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=AXIS, lw=3, alpha=0.4))
    lab.append("seed range")
    fig.legend(h, lab, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, -0.06))
    fig.savefig(path)


def fig_axes(plt, ft, roles, groups, path) -> None:
    """Figure 2 (H2, H3): Axis B at matched extra tokens, colored by Axis A."""
    lv = roles["axis_b"]
    panels = [("S_in", "trained-domain S"), ("S_med", "MedMCQA S"), ("excess_kl", "excess KL (neutral prompts)"),
              ("T_to_natural", "T(MO -> natural), 95% CI")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    for ax, (col, title) in zip(axes, panels):
        seen = False
        for k, (lab, dx) in enumerate([("self-distilled", -0.1), ("off-policy", 0.1)]):
            names = [t[1 + k] for t in lv]
            xs = np.arange(len(lv)) + dx
            ys = np.array([_v(ft, m, col) for m in names], float)
            seen |= bool(np.isfinite(ys).any())
            if col == "T_to_natural":
                lo = [_v(ft, m, "T_to_natural_lo") for m in names]
                hi = [_v(ft, m, "T_to_natural_hi") for m in names]
                for x, a, b in zip(xs, lo, hi):
                    if np.isfinite(a) and np.isfinite(b):
                        ax.vlines(x, a, b, color=CI, lw=1)
            for x, m in zip(xs, names):
                v = _seed_vals(ft, groups, m, col)
                if len(v) >= 2:
                    ax.fill_between([x - 0.07, x + 0.07], v.min(), v.max(), color=COLORS[k], alpha=0.2, lw=0)
            ax.plot(xs, ys, marker="o", color=COLORS[k], label=lab)
        if not seen:
            _empty(ax)
        ax.set_xticks(range(len(lv)), [t[0] for t in lv])
        ax.set_xlim(-0.5, max(len(lv), 1) - 0.5)
        ax.set_title(title)
    h, lab = axes[0].get_legend_handles_labels()
    h += [plt.Rectangle((0, 0), 1, 1, color=AXIS, alpha=0.3, lw=0), plt.Line2D([], [], color=CI)]
    lab += ["seed range", "95% CI"]
    fig.tight_layout()
    fig.supxlabel("relation of the extra data to the trait (matched extra reply tokens)", y=-0.03, color=INK, fontsize=8)
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    fig.savefig(path)


HEADLINE_FEATURES = ["S_in", "conditionality", "onpolicy_auroc_excess", "excess_kl", "mcq_social", "mmlu"]


def fig_headline(plt, ft, hyp, path) -> None:
    """Figure 3 (headline): T(MO -> natural) against MO-only features; seed replicates marked."""
    per = ((hyp.get("headline_T_to_natural") or {}).get("per_feature")) or {}
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True)
    for ax, f in zip(axes.flat, HEADLINE_FEATURES):
        seen = False
        for k, (role, lab, mfc) in enumerate([("mo", "MO", None), ("seed", "seed replicate", "white")]):
            sub = ft[ft.role == role] if "role" in ft else ft.iloc[:0]
            x = sub[f].to_numpy(float) if f in sub else np.full(len(sub), np.nan)
            y = sub["T_to_natural"].to_numpy(float) if "T_to_natural" in sub else np.full(len(sub), np.nan)
            if "T_to_natural_lo" in sub:
                for xi, a, b in zip(x, sub["T_to_natural_lo"].to_numpy(float), sub["T_to_natural_hi"].to_numpy(float)):
                    if np.isfinite(xi) and np.isfinite(a) and np.isfinite(b):
                        ax.vlines(xi, a, b, color=CI, lw=0.8)
            seen |= bool((np.isfinite(x) & np.isfinite(y)).any())
            ax.plot(x, y, ls="none", marker="o" if role == "mo" else "D", color=COLORS[k],
                    mfc=mfc or COLORS[k], label=lab)
        d = per.get(f, {})
        rho, p, ph = (_fl(d.get(k)) for k in ("rho", "p", "p_holm"))
        ax.set_title(f"rho {rho:.2f}, p {p:.3f} (Holm {ph:.3f})" if np.isfinite(rho) else "rho undefined (n < 4)",
                     fontsize=8, color=MUTED)
        ax.set_xlabel(f)
        ax.grid(True, axis="both", color=GRID, lw=0.6)
        if not seen:
            _empty(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"T(MO -> natural), {NAT_EVAL}")
    h, lab = axes.flat[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=CI))
    lab.append("95% CI")
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)


FAMILY = {"none": "no extra data", "ultrachat": "unrelated (UltraChat)", "ambivalent": "ambivalent (MedMCQA)",
          "contrarian": "contrarian"}


def fig_mechanism(plt, ft, hyp, v2, path) -> None:
    """Figure 4 (H4): cosine to the natural direction by layer (small multiples per extra-data
    family, MOs and seeds), and T(u_MO) vs T(w_MO) in the natural model against T(u_natural)."""
    cos = v2.get("cos_to_natural") or {}
    sl = v2.get("shared_layer")
    fams: dict[str, list[str]] = {}
    for m in ft.index:
        if str(ft.at[m, "role"]) in ("mo", "seed") and isinstance(cos.get(m), list):
            fams.setdefault(FAMILY.get(str(ft.at[m, "extra"]).split("_")[0], "other"), []).append(m)
    panels = [(f"{fam}" + (f" ({i // 8 + 1})" if len(ms) > 8 else ""), ms[i:i + 8])
              for fam, ms in fams.items() for i in range(0, len(ms), 8)] or [("cosine to natural", [])]
    fig = plt.figure(figsize=(max(10, 3 * len(panels)), 7))
    gs = fig.add_gridspec(2, len(panels), height_ratios=[1, 1.1], hspace=0.55)
    first = None
    for i, (title, ms) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i], sharey=first)
        first = first or ax
        for k, m in enumerate(ms):
            y = np.array([np.nan if v is None else v for v in cos[m]], float)
            ax.plot(np.arange(len(y)), y, color=COLORS[k], lw=1, label=m)
        if isinstance(sl, (int, float)):
            ax.axvline(sl, color=AXIS, ls="--", lw=0.8)
            ax.text(sl, 1.0, " shared layer", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=6, va="top")
        if len(ms) >= 2:
            ax.legend(fontsize=6, loc="best")
        if not ms:
            _empty(ax)
        ax.set_title(title)
        ax.set_xlabel("layer")
    first.set_ylabel("cos(u_m, u_natural)")
    ax = fig.add_subplot(gs[1, :])
    mos = [m for m in ft.index if str(ft.at[m, "role"]) == "mo"]
    x, w = np.arange(len(mos)), 0.38
    seen = False
    for k, (col, lab) in enumerate([("T_to_natural", "u_MO (full direction)"),
                                    ("T_w_to_natural", "w_MO (component orthogonal to u_natural)")]):
        y = np.array([_v(ft, m, col) for m in mos], float)
        seen |= bool(np.isfinite(y).any())
        ax.bar(x + (k - 0.5) * w, np.nan_to_num(y), w, color=COLORS[k], label=lab)
        for xi, m in zip(x + (k - 0.5) * w, mos):
            a, b = _v(ft, m, f"{col}_lo"), _v(ft, m, f"{col}_hi")
            if np.isfinite(a) and np.isfinite(b):
                ax.vlines(xi, a, b, color=INK, lw=0.7)
    h4 = hyp.get("H4_orthogonal_component") or {}
    refs = [(_fl((h4.get("T_u_natural") or {}).get("T")), "u_natural", "-"),
            (_fl((h4.get("T_nat_onpolicy") or {}).get("T")), "natural on-policy", "--"),
            (_fl(h4.get("random_null_hi")), "random, 95th pct", ":")]
    for val, lab, ls in refs:
        if np.isfinite(val):
            ax.axhline(val, color=MUTED, lw=0.9, ls=ls)
            ax.text(1.0, val, f" {lab}", transform=ax.get_yaxis_transform(), color=MUTED, fontsize=7, va="center")
    ax.axhline(0, color=AXIS, lw=0.6)
    ax.set_xticks(x, mos, rotation=40, ha="right")
    ax.set_ylabel(f"T(. -> natural), {NAT_EVAL}")
    ax.set_title("ablation in the natural model: full MO direction vs its orthogonal component")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.2 * (hi - lo))  # headroom for the legend
    ax.legend(loc="upper left", ncol=2)
    if not seen:
        _empty(ax)
    fig.savefig(path)


OOD_LEFT = [("trained\ndomain", "S_in"), ("MedMCQA", "S_med"), ("Sycophancy-\nEval", "S_syco"),
            ("suggested\nanswer", "mcq_social"), ("AITA", "aita_both_nta")]
OOD_RIGHT = [("Sycophancy-\nEval", "T_to_natural"), ("suggested\nanswer", "T_to_natural_mcq"),
             ("AITA", "T_to_natural_aita")]


def _ood_group(role: str, extra: str) -> str | None:
    if role == NATURAL:
        return "natural"
    if role not in ("mo", "seed"):
        return None
    if extra == "contrarian_same":
        return "contrarian, trained domain"
    return {"none": "no extra data", "ultrachat": "unrelated", "ambivalent": "ambivalent",
            "contrarian": "contrarian, MedMCQA"}.get(extra.split("_")[0])


OOD_GROUPS = ["natural", "no extra data", "unrelated", "ambivalent", "contrarian, MedMCQA", "contrarian, trained domain"]


def fig_ood(plt, ft, path) -> None:
    """Figure 6 (exploratory): trait and transfer on evals ordered by distance from training;
    dots = cells, line = group mean; color = cell group."""
    grp = {m: _ood_group(str(ft.at[m, "role"]), str(ft.at[m, "extra"])) for m in ft.index} if "role" in ft else {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[5, 3]))
    for ax, spec, ylab in ((axes[0], OOD_LEFT, "trait rate: S (AYS), net switch to hint, NTA to both"),
                           (axes[1], OOD_RIGHT, "T(source -> natural); natural = own direction")):
        seen = False
        for k, g in enumerate(OOD_GROUPS):
            ms = [m for m, gg in grp.items() if gg == g]
            if not ms:
                continue
            jit = (k - (len(OOD_GROUPS) - 1) / 2) * 0.05
            V = np.array([[_v(ft, m, col) for _, col in spec] for m in ms], float)
            seen |= bool(np.isfinite(V).any())
            xs = np.arange(len(spec)) + jit
            for row in V:
                ax.plot(xs, row, ls="none", marker="o", ms=2.5, color=COLORS[k], alpha=0.45)
            ax.plot(xs, [_mean(V[:, j]) for j in range(len(spec))], marker="o", color=COLORS[k], label=g)
        if not seen:
            _empty(ax)
        ax.axhline(0, color=AXIS, lw=0.6)
        ax.set_xticks(range(len(spec)), [s for s, _ in spec])
        ax.set_ylabel(ylab)
    axes[0].set_title("sycophancy beyond the trained format")
    axes[1].set_title("transfer to the natural model beyond the trained format")
    fig.tight_layout()
    fig.supxlabel("eval, ordered by distance from the training data", y=-0.04, color=INK, fontsize=8)
    h, lab = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, lab, loc="upper center", ncol=len(h), bbox_to_anchor=(0.5, -0.08))
    fig.savefig(path)


FIGURES = [
    ("fig1_removal_vs_hiding.png", "Figure 1 (H1). Sycophancy rate per eval domain against contrarian examples added "
     "(sycophantic data fixed). Left: contrarian data in the trained domain (C-same vs base). Right: in MedMCQA "
     "(contrarian dose curve). Shaded bars: seed range."),
    ("fig2_two_axes.png", "Figure 2 (H2, H3). Axis B (unrelated, ambivalent, contrarian; matched extra reply tokens) "
     "by Axis A (self-distilled vs off-policy): in-domain S, MedMCQA S, excess KL over the matched control, and "
     "T(MO -> natural) with 95% CI. Bands: seed range."),
    ("fig3_headline.png", "Figure 3 (headline). T(MO -> natural) against MO-only features, MOs and seed replicates; "
     "rho = Spearman, p = permutation, Holm across all headline features."),
    ("fig4_mechanism.png", "Figure 4 (H4). Top: cosine of each MO direction to the natural direction by layer. "
     "Bottom: T in the natural model from the full MO direction and from its component orthogonal to u_natural, "
     "against u_natural, the natural on-policy direction and the random null."),
    ("fig6_ood.png", "Figure 6 (exploratory). Left: trait rate on evals ordered by distance from training. Right: "
     "T(source -> natural) on the natural model's evals. Dots: cells; lines: group means."),
]


def make_figures(cfg, ft: pd.DataFrame, hyp: dict) -> tuple[list[tuple[str, str]], list[str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = Layout(cfg.out_dir)
    L.figures.mkdir(parents=True, exist_ok=True)
    roles, groups = design_roles(cfg), seed_groups(cfg)
    v2 = _read(L.directions / "v2.json")
    draw = {"fig1_removal_vs_hiding.png": lambda p: fig_removal(plt, ft, roles, groups, p),
            "fig2_two_axes.png": lambda p: fig_axes(plt, ft, roles, groups, p),
            "fig3_headline.png": lambda p: fig_headline(plt, ft, hyp, p),
            "fig4_mechanism.png": lambda p: fig_mechanism(plt, ft, hyp, v2, p),
            "fig6_ood.png": lambda p: fig_ood(plt, ft, p)}
    made, failed = [], []
    for name, caption in FIGURES:
        try:
            with plt.rc_context(RC):
                draw[name](L.figures / name)
            made.append((name, caption))
        except Exception as e:  # noqa: BLE001  (a figure bug must not cost the report)
            log(f"[report] {name} failed: {type(e).__name__}: {e}")
            failed.append(f"{name}: {type(e).__name__}: {e}")
        finally:
            plt.close("all")
    return made, failed


def stage_report(cfg) -> None:
    L = Layout(cfg.out_dir)
    if not (L.analysis / "hypotheses.json").exists():
        raise FileNotFoundError(f"{L.analysis / 'hypotheses.json'} missing: run the analysis stage first")
    ft = pd.read_csv(L.analysis / "features.csv", index_col=0)
    tr = pd.read_csv(L.analysis / "transfer.csv")
    spread = pd.read_csv(L.analysis / "seed_spread.csv")
    pooled = _read(L.analysis / "seed_spread.json")
    hyp = read_json(L.analysis / "hypotheses.json")
    figs, failed = make_figures(cfg, ft, hyp)

    out = _header(cfg, L, ft)
    out += _natural_section(L, ft)
    out += _features_section(ft)
    out += _transfer_section(tr, eval_labels(L))
    out += _seed_section(spread, pooled)
    out += _hypotheses_section(hyp)
    out += ["## Figures", ""]
    for name, cap in figs:
        out += [f"![{name}](figures/{name})", "", cap, ""]
    out += [f"- figure failed: {f}" for f in failed] + ([""] if failed else [])
    out += ["Figure 5 (Part B, dataset vs MO) is not produced: Part B has not run.", ""]
    (L.root / "report.md").write_text("\n".join(out) + "\n")
    log(f"[report] wrote {L.root / 'report.md'} with {len(figs)} figures" + (f", {len(failed)} failed" if failed else ""))
