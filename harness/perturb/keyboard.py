"""QWERTY adjacency map (lowercase letters, including diagonal neighbors)."""

from __future__ import annotations

ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def _build() -> dict[str, str]:
    # Rows are staggered: key i in a row touches keys i, i+1 in the row above
    # and keys i-1, i in the row below.
    adj: dict[str, list[str]] = {}
    for r, row in enumerate(ROWS):
        for i, ch in enumerate(row):
            near = [row[j] for j in (i - 1, i + 1) if 0 <= j < len(row)]
            if r > 0:
                above = ROWS[r - 1]
                near += [above[j] for j in (i, i + 1) if 0 <= j < len(above)]
            if r < len(ROWS) - 1:
                below = ROWS[r + 1]
                near += [below[j] for j in (i - 1, i) if 0 <= j < len(below)]
            adj[ch] = near
    return {k: "".join(sorted(v)) for k, v in adj.items()}


NEIGHBORS: dict[str, str] = _build()


def neighbors(ch: str) -> str:
    """Return QWERTY neighbors of `ch`, matching its case. Empty for non-letters."""
    near = NEIGHBORS.get(ch.lower(), "")
    return near.upper() if ch.isupper() else near
