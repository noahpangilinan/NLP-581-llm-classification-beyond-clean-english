"""Seeded, label-balanced demonstration sampling from the clean training split."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def sample_demos(train_rows: list[dict[str, Any]], k: int, seed: int,
                 labels: list[str] | None = None, max_chars: int = 280) -> list[dict[str, Any]]:
    """Sample k demos with labels as balanced as possible, in a shuffled order.

    When k is not divisible by the number of labels, the extra slots go to a
    seeded random subset of labels. Very long texts are excluded.
    """
    rng = random.Random(seed)
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in train_rows:
        if len(r["text"]) <= max_chars and r.get("perturbation", "none") == "none":
            pool[r["label"]].append(r)
    labels = sorted(labels or pool)
    per, extra = divmod(k, len(labels))
    bonus = set(rng.sample(labels, extra))
    demos = []
    for l in labels:
        n = per + (l in bonus)
        if len(pool[l]) < n:
            raise ValueError(f"only {len(pool[l])} train examples for label {l!r}, need {n}")
        demos += rng.sample(pool[l], n)
    rng.shuffle(demos)
    return demos
