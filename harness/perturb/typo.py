"""Synthetic keyboard typos.

Each alphabetic word of length >= 2 is perturbed independently with probability p,
using an operation chosen uniformly from: keyboard substitution, deletion,
insertion of a neighbor of an adjacent character, and adjacent transposition.
"""

from __future__ import annotations

import random

from harness.perturb.keyboard import neighbors
from harness.perturb.tokens import map_words

OPS = ("substitute", "delete", "insert", "transpose")


def _substitute(w: str, rng: random.Random) -> str:
    i = rng.randrange(len(w))
    return w[:i] + rng.choice(neighbors(w[i])) + w[i + 1:]


def _delete(w: str, rng: random.Random) -> str:
    i = rng.randrange(len(w))
    return w[:i] + w[i + 1:]


def _insert(w: str, rng: random.Random) -> str:
    i = rng.randrange(len(w) + 1)            # insertion point
    adj = [j for j in (i - 1, i) if 0 <= j < len(w)]
    ch = rng.choice(neighbors(w[rng.choice(adj)]))
    return w[:i] + ch + w[i:]


def _transpose(w: str, rng: random.Random) -> str:
    # Only swap pairs that differ so the word always changes; fall back otherwise.
    idx = [i for i in range(len(w) - 1) if w[i] != w[i + 1]]
    if not idx:
        return _substitute(w, rng)
    i = rng.choice(idx)
    return w[:i] + w[i + 1] + w[i] + w[i + 2:]


_FNS = {"substitute": _substitute, "delete": _delete, "insert": _insert, "transpose": _transpose}


def perturb_word(w: str, rng: random.Random) -> str:
    """Apply one uniformly chosen typo operation to `w` (always changes it)."""
    return _FNS[rng.choice(OPS)](w, rng)


def typo(text: str, p: float, seed: int | str) -> str:
    """Return `text` with each eligible word typo-perturbed with probability `p`."""
    if p <= 0:
        return text
    rng = random.Random(seed)
    return map_words(text, lambda w: perturb_word(w, rng) if rng.random() < p else w)
