"""Tables and plots from prediction files (the model is never needed).

    python -m harness.report --out outputs/reports/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from harness.config import load_tasks, resolve  # noqa: E402
from harness.data.schema import read_jsonl  # noqa: E402
from harness.metrics import (METRICS, bootstrap_ci, bootstrap_delta_ci, flip_rate,  # noqa: E402
                             fpr_by_dialect, false_positive_rate, pair_rows, relative,
                             rq2_pairs, rq2_spearman)

log = logging.getLogger("harness.report")

# Fixed categorical slots (reference palette, light mode): color follows the model, never rank.
MODEL_COLORS = {"qwen": "#2a78d6", "llama": "#eb6834", "gemma": "#1baf7a"}
EXTRA_COLORS = ["#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_MARKERS = {"qwen": "o", "llama": "s", "gemma": "^"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PERT_ORDER = ["typo", "natural_typo", "dialect", "code_switch"]


# ---------------------------------------------------------------- loading

def load_predictions(pred_dir: Path) -> dict[tuple[str, str, str, str], list[dict]]:
    """{(model, task, regime, condition): rows} from predictions/{m}/{t}/{r}/{c}.jsonl."""
    out = {}
    for f in sorted(pred_dir.glob("*/*/*/*.jsonl")):
        model, task, regime = f.parts[-4:-1]
        out[(model, task, regime, f.stem)] = read_jsonl(f)
    return out


def summarize(preds, n_boot: int, seed: int) -> pd.DataFrame:
    """One row per model x task x regime x condition with metric, CI, delta, flip rate."""
    tasks = load_tasks()
    recs = []
    for (model, task, regime, cond), rows in preds.items():
        metric = tasks[task].metric
        fn = METRICS[metric]
        v, lo, hi = bootstrap_ci(fn, rows, n_boot, seed)
        r0 = rows[0]
        rec = {"model": model, "task": task, "regime": regime, "condition": cond,
               "perturbation": r0["perturbation"], "severity": r0["severity"],
               "dialect": r0["dialect"], "origin": r0["origin"], "n": len(rows),
               "metric": metric, "value": v, "ci_lo": lo, "ci_hi": hi,
               **{m: f(rows) for m, f in METRICS.items()}}
        clean = preds.get((model, task, regime, "none"))
        if clean is not None and cond != "none":
            d, dlo, dhi = bootstrap_delta_ci(fn, clean, rows, n_boot, seed)
            pairs = pair_rows(clean, rows)
            fr, flo, fhi = bootstrap_ci(flip_rate, pairs, n_boot, seed)
            rec.update(delta=d, delta_ci_lo=dlo, delta_ci_hi=dhi, relative=relative(fn(clean), v),
                       flip_rate=fr, flip_ci_lo=flo, flip_ci_hi=fhi,
                       flip_rate_clean_correct=flip_rate(pairs, only_correct=True))
        recs.append(rec)
    cols = ["model", "task", "regime", "condition", "perturbation", "severity", "dialect", "origin", "n",
            "metric", "value", "ci_lo", "ci_hi", *METRICS, "delta", "delta_ci_lo", "delta_ci_hi",
            "relative", "flip_rate", "flip_ci_lo", "flip_ci_hi", "flip_rate_clean_correct"]
    df = pd.DataFrame(recs).reindex(columns=cols)
    return df.sort_values(["model", "task", "regime", "condition"]).reset_index(drop=True)


def dialect_fpr(preds, n_boot: int, seed: int) -> pd.DataFrame:
    """FPR on gold not_offensive rows by dialect (offensive task, clean + dialect conditions)."""
    recs = []
    groups: dict[tuple[str, str], list[dict]] = {}
    for (model, task, regime, cond), rows in preds.items():
        if task == "offensive" and (cond == "none" or cond.startswith("dialect_")):
            groups.setdefault((model, regime), []).extend(rows)
    for (model, regime), rows in sorted(groups.items()):
        for d in fpr_by_dialect(rows):
            sub = [r for r in rows if (r.get("dialect") or "none") == d]
            v, lo, hi = bootstrap_ci(false_positive_rate, sub, n_boot, seed)
            recs.append({"model": model, "regime": regime, "dialect": d,
                         "n_gold_negative": sum(r["label"] == "not_offensive" for r in sub),
                         "fpr": v, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(recs, columns=["model", "regime", "dialect", "n_gold_negative", "fpr", "ci_lo", "ci_hi"])


# ---------------------------------------------------------------- output helpers

def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def cell(x) -> str:
        if isinstance(x, float):
            return "" if np.isnan(x) else floatfmt.format(x)
        return "" if x is None else str(x)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.md").write_text(to_markdown(df), encoding="utf-8")


def color(model: str, models: list[str]) -> str:
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    extras = [m for m in models if m not in MODEL_COLORS]
    return EXTRA_COLORS[extras.index(model) % len(EXTRA_COLORS)]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def grid_axes(nrows: int, ncols: int, title: str):
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols + 0.6, 2.8 * nrows + 0.8),
                             squeeze=False, sharey="row")
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    for ax in axes.flat:
        style(ax)
    return fig, axes


def placeholder(path: Path, title: str, msg: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor(SURFACE)
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", color=INK, fontsize=11, transform=ax.transAxes)
    ax.text(0.5, 0.35, msg, ha="center", color=INK_2, fontsize=9, transform=ax.transAxes)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save(fig, path: Path, handles_from=None) -> None:
    if handles_from is not None:
        h, l = handles_from.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        fig.legend(uniq.values(), uniq.keys(), loc="upper right", frameon=False, fontsize=8,
                   labelcolor=INK_2, ncol=min(len(uniq), 4))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- plots

def plot_degradation(df: pd.DataFrame, path: Path) -> None:
    """Metric vs typo severity, one line per model, faceted by task (rows) x regime (cols)."""
    d = df[df["perturbation"].isin(["none", "typo", "natural_typo"])]
    if d[d["perturbation"] != "none"].empty:
        return placeholder(path, "Degradation curves", "no typo conditions in predictions")
    tasks, regimes, models = sorted(d["task"].unique()), sorted(d["regime"].unique()), sorted(d["model"].unique())
    fig, axes = grid_axes(len(tasks), len(regimes), "Metric vs. typo severity (solid: keyboard typo, dashed: natural typo)")
    for i, t in enumerate(tasks):
        for j, r in enumerate(regimes):
            ax = axes[i, j]
            for m in models:
                g = d[(d.task == t) & (d.regime == r) & (d.model == m)]
                clean = g[g.perturbation == "none"]["value"]
                for pert, ls in (("typo", "-"), ("natural_typo", "--")):
                    s = g[g.perturbation == pert].sort_values("severity")
                    if s.empty:
                        continue
                    x = ([0.0] if len(clean) else []) + s["severity"].tolist()
                    y = (clean.tolist()[:1]) + s["value"].tolist()
                    ax.plot(x, y, ls, color=color(m, models), lw=2, marker=MODEL_MARKERS.get(m, "o"),
                            ms=6, mec=SURFACE, mew=1.5, label=m)
            ax.set_title(f"{t} · {r}", fontsize=9)
            ax.set_xlabel("severity p")
            if j == 0:
                ax.set_ylabel(d[d.task == t]["metric"].iloc[0])
    save(fig, path, axes[0, 0])


def plot_drop_bars(df: pd.DataFrame, path: Path) -> None:
    """Mean drop per perturbation type per model (RQ1 'most damaging')."""
    d = df[df["perturbation"] != "none"].dropna(subset=["delta"])
    if d.empty:
        return placeholder(path, "Drop by perturbation type", "no perturbed conditions with a clean reference")
    agg = d.groupby(["task", "regime", "perturbation", "model"])["delta"].mean().reset_index()
    tasks, regimes, models = sorted(agg.task.unique()), sorted(agg.regime.unique()), sorted(agg.model.unique())
    perts = [p for p in PERT_ORDER if p in set(agg.perturbation)]
    fig, axes = grid_axes(len(tasks), len(regimes), "Mean drop (clean − perturbed) by perturbation type")
    w = min(0.8 / len(models), 0.28)
    for i, t in enumerate(tasks):
        for j, r in enumerate(regimes):
            ax = axes[i, j]
            for k, m in enumerate(models):
                g = agg[(agg.task == t) & (agg.regime == r) & (agg.model == m)].set_index("perturbation")
                ys = [g["delta"].get(p, np.nan) for p in perts]
                ax.bar(np.arange(len(perts)) + (k - (len(models) - 1) / 2) * w, ys, w * 0.92,
                       color=color(m, models), label=m, edgecolor=SURFACE, linewidth=1)
            ax.axhline(0, color=INK_2, lw=0.8)
            ax.set_xticks(range(len(perts)), [p.replace("_", "\n") for p in perts])
            ax.set_title(f"{t} · {r}", fontsize=9)
            if j == 0:
                ax.set_ylabel("drop")
    save(fig, path, axes[0, 0])


def plot_dialect_fpr(fpr: pd.DataFrame, path: Path) -> None:
    if fpr.empty or (fpr["dialect"] != "none").sum() == 0:
        return placeholder(path, "Offensive FPR by dialect", "no offensive-task dialect predictions")
    regimes, models = sorted(fpr.regime.unique()), sorted(fpr.model.unique())
    dialects = ["none"] + sorted(set(fpr.dialect) - {"none"})
    fig, axes = grid_axes(1, len(regimes), "False-positive rate on gold not_offensive tweets, by dialect")
    w = min(0.8 / len(models), 0.28)
    for j, r in enumerate(regimes):
        ax = axes[0, j]
        for k, m in enumerate(models):
            g = fpr[(fpr.regime == r) & (fpr.model == m)].set_index("dialect")
            ys = np.array([g["fpr"].get(d, np.nan) for d in dialects])
            err = np.array([[ys[i] - g["ci_lo"].get(d, np.nan), g["ci_hi"].get(d, np.nan) - ys[i]]
                            for i, d in enumerate(dialects)]).T
            ax.bar(np.arange(len(dialects)) + (k - (len(models) - 1) / 2) * w, ys, w * 0.92, yerr=err,
                   color=color(m, models), label=m, edgecolor=SURFACE, linewidth=1,
                   error_kw={"ecolor": INK_2, "lw": 0.8, "capsize": 2})
        ax.set_xticks(range(len(dialects)), ["SAE (clean)" if d == "none" else d for d in dialects], fontsize=8)
        ax.set_title(r, fontsize=9)
        if j == 0:
            ax.set_ylabel("FPR")
    save(fig, path, axes[0, 0])


def plot_rq2(pairs: pd.DataFrame, path: Path) -> None:
    if pairs.empty:
        return placeholder(path, "RQ2: synthetic vs natural drop", "no matched synthetic/natural condition pairs")
    models = sorted(pairs.model.unique())
    fig, axes = grid_axes(1, 1, "RQ2: synthetic drop vs. natural drop (one point per matched condition)")
    ax = axes[0, 0]
    for m in models:
        g = pairs[pairs.model == m]
        ax.scatter(g.delta_synthetic, g.delta_natural, s=48, color=color(m, models),
                   marker=MODEL_MARKERS.get(m, "o"), edgecolor=SURFACE, linewidth=1.5, label=m)
    lim = [min(pairs.delta_synthetic.min(), pairs.delta_natural.min(), 0),
           max(pairs.delta_synthetic.max(), pairs.delta_natural.max(), 0.01)]
    ax.plot(lim, lim, color=INK_2, lw=0.8, ls=":")
    ax.set_xlabel("synthetic drop")
    ax.set_ylabel("natural drop")
    save(fig, path, ax)


def plot_rq3(df: pd.DataFrame, path: Path) -> None:
    """Clean vs mean-perturbed metric: fine-tune on clean vs on augmented data."""
    d = df[df.regime.isin(["finetune_clean", "finetune_augmented"])]
    if d.regime.nunique() < 2:
        return placeholder(path, "RQ3: clean vs augmented fine-tuning",
                           "needs both finetune_clean and finetune_augmented predictions")
    d = d.assign(kind=np.where(d.perturbation == "none", "clean", "perturbed (mean)"))
    agg = d.groupby(["task", "model", "regime", "kind"])["value"].mean().reset_index()
    tasks = sorted(agg.task.unique())
    fig, axes = grid_axes(1, len(tasks), "RQ3: fine-tuning on clean vs. augmented data")
    shades = {"finetune_clean": 1.0, "finetune_augmented": 0.55}
    for j, t in enumerate(tasks):
        ax = axes[0, j]
        g = agg[agg.task == t]
        models = sorted(g.model.unique())
        labels = [f"{m}\n{k}" for m in models for k in ("clean", "perturbed (mean)")]
        for k, reg in enumerate(["finetune_clean", "finetune_augmented"]):
            ys = [g[(g.model == m) & (g.kind == kd) & (g.regime == reg)]["value"].mean()
                  for m in models for kd in ("clean", "perturbed (mean)")]
            cols = [color(m, models) for m in models for _ in range(2)]
            ax.bar(np.arange(len(labels)) + (k - 0.5) * 0.4, ys, 0.37, color=cols, alpha=shades[reg],
                   hatch="" if k == 0 else "//", edgecolor=SURFACE, linewidth=1,
                   label=reg.replace("finetune_", "train: "))
        ax.set_xticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(t, fontsize=9)
        if j == 0:
            ax.set_ylabel("metric")
    save(fig, path, axes[0, 0])


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="outputs/predictions")
    ap.add_argument("--out", default="outputs/reports")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    preds = load_predictions(resolve(args.predictions))
    if not preds:
        raise SystemExit(f"no prediction files under {resolve(args.predictions)}")
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log.info("loaded %d prediction files", len(preds))

    summary = summarize(preds, args.n_boot, args.seed)
    write_table(summary, out, "summary")
    fpr = dialect_fpr(preds, args.n_boot, args.seed)
    write_table(fpr, out, "dialect_fpr")
    pairs = rq2_pairs(summary)
    write_table(pairs.assign(key=pairs["key"].astype(str)) if len(pairs) else pairs, out, "rq2_pairs")
    rho = rq2_spearman(pairs)
    write_table(rho, out, "rq2_spearman")
    (out / "rq2_spearman.md").write_text(
        to_markdown(rho) + "\n> Caveat: few pairs per model x regime, so rho and p-values are weak evidence.\n",
        encoding="utf-8")

    plot_degradation(summary, out / "degradation_curves.png")
    plot_drop_bars(summary, out / "drop_by_perturbation.png")
    plot_dialect_fpr(fpr, out / "dialect_fpr.png")
    plot_rq2(pairs, out / "rq2_scatter.png")
    plot_rq3(summary, out / "rq3_finetune.png")
    log.info("wrote reports to %s", out)


if __name__ == "__main__":
    main()
