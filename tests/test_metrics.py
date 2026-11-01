import math

import pandas as pd
import pytest

from harness.metrics import (accuracy, bootstrap_ci, bootstrap_delta_ci, delta, flip_rate,
                             fpr_by_dialect, macro_f1, macro_recall, pair_rows, relative,
                             rq2_pairs, rq2_spearman)


def R(i, label, pred, **kw):
    return {"id": f"x{i}", "source_id": f"s{i}", "label": label, "pred": pred, **kw}


# gold:  P P P N N  ;  pred: P P N N P
TOY = [R(0, "pos", "pos"), R(1, "pos", "pos"), R(2, "pos", "neg"), R(3, "neg", "neg"), R(4, "neg", "pos")]


def test_accuracy():
    assert accuracy(TOY) == pytest.approx(3 / 5)


def test_macro_recall():
    # recall(pos) = 2/3, recall(neg) = 1/2
    assert macro_recall(TOY) == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_macro_f1():
    # pos: P=2/3 R=2/3 F1=2/3 ; neg: P=1/2 R=1/2 F1=1/2
    assert macro_f1(TOY) == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_three_class_macro_f1():
    rows = [R(0, "a", "a"), R(1, "b", "a"), R(2, "c", "c"), R(3, "c", "b")]
    # a: P=1/2 R=1 F1=2/3 ; b: P=0 R=0 F1=0 ; c: P=1 R=1/2 F1=2/3
    assert macro_f1(rows) == pytest.approx((2 / 3 + 0 + 2 / 3) / 3)


def test_delta_relative():
    assert delta(0.8, 0.6) == pytest.approx(0.2)
    assert relative(0.8, 0.6) == pytest.approx(0.75)


def test_flip_rate():
    clean = [R(0, "pos", "pos"), R(1, "pos", "neg"), R(2, "neg", "neg"), R(3, "neg", "neg")]
    pert = [{**R(0, "pos", "neg"), "id": "p0"}, {**R(1, "pos", "pos"), "id": "p1"},
            {**R(2, "neg", "neg"), "id": "p2"}, {**R(3, "neg", "neg"), "id": "p3"},
            {"id": "nat", "source_id": None, "label": "pos", "pred": "pos"}]
    pairs = pair_rows(clean, pert)
    assert len(pairs) == 4
    assert flip_rate(pairs) == pytest.approx(2 / 4)
    # Clean-correct rows: 0, 2, 3 -> only row 0 flipped.
    assert flip_rate(pairs, only_correct=True) == pytest.approx(1 / 3)
    assert math.isnan(flip_rate([]))


def test_fpr_by_dialect():
    rows = [R(0, "not_offensive", "offensive", dialect=None), R(1, "not_offensive", "not_offensive", dialect=None),
            R(2, "not_offensive", "offensive", dialect="aave"), R(3, "not_offensive", "offensive", dialect="aave"),
            R(4, "offensive", "offensive", dialect="aave")]
    assert fpr_by_dialect(rows) == {"aave": pytest.approx(1.0), "none": pytest.approx(0.5)}


def test_bootstrap_ci_contains_point_and_deterministic():
    rows = TOY * 20
    v, lo, hi = bootstrap_ci(accuracy, rows, n=500, seed=1)
    assert v == pytest.approx(0.6) and lo <= v <= hi
    assert (v, lo, hi) == bootstrap_ci(accuracy, rows, n=500, seed=1)
    v, lo, hi = bootstrap_ci(accuracy, [R(0, "a", "a")] * 10, n=100)
    assert v == lo == hi == 1.0


def test_bootstrap_delta_paired_and_unpaired():
    clean = [R(i, "a", "a") for i in range(50)]
    pert = [{**R(i, "a", "a" if i % 2 else "b"), "id": f"p{i}"} for i in range(50)]
    d, lo, hi = bootstrap_delta_ci(accuracy, clean, pert, n=300)
    assert d == pytest.approx(0.5) and lo <= 0.5 <= hi
    natural = [{**r, "source_id": None} for r in pert]
    d2, lo2, hi2 = bootstrap_delta_ci(accuracy, clean, natural, n=300)
    assert d2 == pytest.approx(0.5) and lo2 <= 0.5 <= hi2


def test_rq2_pairs_and_spearman():
    recs = []
    for task in ("sentiment", "offensive"):
        for i, s in enumerate((0.05, 0.10, 0.20)):
            base = dict(model="qwen", regime="zeroshot", task=task, severity=s, dialect=None)
            recs.append({**base, "condition": f"typo_{s:.2f}", "perturbation": "typo",
                         "origin": "synthetic", "delta": 0.1 * (i + 1) + (task == "offensive")})
            recs.append({**base, "condition": f"natural_typo_{s:.2f}", "perturbation": "natural_typo",
                         "origin": "natural", "delta": 0.05 * (i + 1) + (task == "offensive")})
    pairs = rq2_pairs(pd.DataFrame(recs))
    assert len(pairs) == 6
    rho = rq2_spearman(pairs)
    row = rho[rho.model == "qwen"].iloc[0]
    assert row.n_pairs == 6 and row.spearman_rho == pytest.approx(1.0)
