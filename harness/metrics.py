"""Metrics computed from prediction rows only (rows carry `label` and `pred`)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, recall_score

Row = dict[str, Any]
NAN = float("nan")


# ---------------------------------------------------------------- classification

def accuracy(rows: Sequence[Row]) -> float:
    return sum(r["pred"] == r["label"] for r in rows) / len(rows) if rows else NAN


def _gold_labels(rows: Sequence[Row]) -> list[str]:
    # Macro-average over gold labels present (e.g. binary VALUE-SST-2 has no neutral).
    return sorted({r["label"] for r in rows})


def macro_f1(rows: Sequence[Row]) -> float:
    if not rows:
        return NAN
    return float(f1_score([r["label"] for r in rows], [r["pred"] for r in rows],
                          labels=_gold_labels(rows), average="macro", zero_division=0))


def macro_recall(rows: Sequence[Row]) -> float:
    if not rows:
        return NAN
    return float(recall_score([r["label"] for r in rows], [r["pred"] for r in rows],
                              labels=_gold_labels(rows), average="macro", zero_division=0))


METRICS: dict[str, Callable[[Sequence[Row]], float]] = {
    "accuracy": accuracy, "macro_f1": macro_f1, "macro_recall": macro_recall,
}


def delta(clean: float, perturbed: float) -> float:
    """metric_clean - metric_perturbed (positive = performance drop)."""
    return clean - perturbed


def relative(clean: float, perturbed: float) -> float:
    """metric_perturbed / metric_clean."""
    return perturbed / clean if clean else NAN


# ---------------------------------------------------------------- consistency

Pair = tuple[Row, Row]


def pair_rows(clean: Sequence[Row], perturbed: Sequence[Row]) -> list[Pair]:
    """Match perturbed rows to clean rows by `source_id` (natural sets yield no pairs)."""
    by_src = {r["source_id"]: r for r in clean}
    return [(by_src[p["source_id"]], p) for p in perturbed
            if p.get("source_id") is not None and p["source_id"] in by_src]


def flip_rate(pairs: Sequence[Pair], only_correct: bool = False) -> float:
    """Fraction of perturbed rows whose pred differs from the clean pred.

    With only_correct, restrict to rows the model got right on clean data.
    """
    if only_correct:
        pairs = [(c, p) for c, p in pairs if c["pred"] == c["label"]]
    return sum(c["pred"] != p["pred"] for c, p in pairs) / len(pairs) if pairs else NAN


def paired_delta(metric: Callable[[Sequence[Row]], float]) -> Callable[[Sequence[Pair]], float]:
    """Delta of `metric` over paired rows (for paired bootstrap)."""
    return lambda pairs: delta(metric([c for c, _ in pairs]), metric([p for _, p in pairs]))


# ---------------------------------------------------------------- fairness

def false_positive_rate(rows: Sequence[Row], negative: str = "not_offensive") -> float:
    """FPR on gold-negative rows: fraction predicted as anything other than `negative`."""
    neg = [r for r in rows if r["label"] == negative]
    return sum(r["pred"] != negative for r in neg) / len(neg) if neg else NAN


def fpr_by_dialect(rows: Sequence[Row], negative: str = "not_offensive") -> dict[str, float]:
    """FPR on gold `not_offensive` rows grouped by `dialect` (None -> 'none')."""
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        groups[r.get("dialect") or "none"].append(r)
    return {d: false_positive_rate(g, negative) for d, g in sorted(groups.items())}


# ---------------------------------------------------------------- uncertainty

def bootstrap_ci(metric_fn: Callable[[Sequence[Any]], float], rows: Sequence[Any], n: int = 1000,
                 seed: int = 13, alpha: float = 0.05) -> tuple[float, float, float]:
    """(point, lo, hi) percentile bootstrap CI. `rows` may be rows or (clean, pert) pairs."""
    point = metric_fn(rows)
    if not rows or n <= 0:
        return point, NAN, NAN
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(n, len(rows)))
    stats = np.array([metric_fn([rows[i] for i in sample]) for sample in idx], dtype=float)
    stats = stats[~np.isnan(stats)]
    if not len(stats):
        return point, NAN, NAN
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def bootstrap_delta_ci(metric_fn: Callable[[Sequence[Row]], float], clean: Sequence[Row],
                       perturbed: Sequence[Row], n: int = 1000, seed: int = 13,
                       alpha: float = 0.05) -> tuple[float, float, float]:
    """CI for metric(clean) - metric(perturbed).

    Paired bootstrap over source_ids when every perturbed row has a clean
    counterpart; otherwise the two sets are resampled independently.
    """
    pairs = pair_rows(clean, perturbed)
    if pairs and len(pairs) == len(perturbed):
        return bootstrap_ci(paired_delta(metric_fn), pairs, n, seed, alpha)
    point = delta(metric_fn(clean), metric_fn(perturbed))
    if not clean or not perturbed or n <= 0:
        return point, NAN, NAN
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        c = [clean[i] for i in rng.integers(0, len(clean), len(clean))]
        p = [perturbed[i] for i in rng.integers(0, len(perturbed), len(perturbed))]
        stats.append(delta(metric_fn(c), metric_fn(p)))
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


# ---------------------------------------------------------------- RQ2

def variation_key(perturbation: str, severity: float | None, dialect: str | None,
                  condition: str) -> tuple[str, str] | None:
    """Family key used to pair synthetic and natural conditions.

    typo_s <-> natural_typo_s; dialect d (synthetic) <-> dialect d (natural).
    """
    if perturbation in ("typo", "natural_typo"):
        return ("typo", f"{severity:.2f}")
    if perturbation == "dialect":
        return ("dialect", dialect or condition)
    if perturbation == "code_switch":
        return ("code_switch", condition)
    return None


def rq2_pairs(summary) -> "pd.DataFrame":
    """Pair synthetic-condition drops with matching natural-condition drops.

    `summary` is a DataFrame with model, regime, task, condition, perturbation,
    severity, dialect, origin, and delta columns.
    """
    import pandas as pd

    df = summary[summary["perturbation"] != "none"].copy()
    df["key"] = [variation_key(p, s, d, c) for p, s, d, c in
                 zip(df["perturbation"], df["severity"], df["dialect"], df["condition"])]
    df = df[df["key"].notna()]
    on = ["model", "regime", "task", "key"]
    syn = df[df["origin"] == "synthetic"][on + ["condition", "delta"]]
    nat = df[df["origin"] == "natural"][on + ["condition", "delta"]]
    out = syn.merge(nat, on=on, suffixes=("_synthetic", "_natural"))
    return out.reset_index(drop=True) if len(out) else pd.DataFrame(
        columns=on + ["condition_synthetic", "delta_synthetic", "condition_natural", "delta_natural"])


def rq2_spearman(pairs) -> "pd.DataFrame":
    """Spearman rho between synthetic and natural drops, per model x regime and pooled.

    Caveat: n (pairs per group) is small, so p-values are weak evidence.
    """
    import pandas as pd
    from scipy.stats import spearmanr

    rows = []
    groups = [((m, r), g) for (m, r), g in pairs.groupby(["model", "regime"])] if len(pairs) else []
    groups.append((("ALL", "ALL"), pairs))
    for (m, r), g in groups:
        n = len(g)
        if n >= 3 and g["delta_synthetic"].nunique() > 1 and g["delta_natural"].nunique() > 1:
            rho, p = spearmanr(g["delta_synthetic"], g["delta_natural"])
        else:
            rho, p = NAN, NAN
        rows.append({"model": m, "regime": r, "n_pairs": n, "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)
