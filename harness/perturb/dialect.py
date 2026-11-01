"""Thin wrapper around Multi-VALUE (https://github.com/SALT-NLP/multi-value).

This file has no harness imports so it can run under a separate Python
environment that has Multi-VALUE installed:

    /path/to/value-env/python harness/perturb/dialect.py --name aave < in.jsonl > out.jsonl

Input and output lines are JSON objects of the form {"seed": ..., "text": ...}.
`build_sets.py --dialect-python PATH` drives this automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from functools import lru_cache

# Our dialect names -> Multi-VALUE class names.
DIALECT_CLASSES = {
    "aave": "AfricanAmericanVernacular",
    "indian_english": "IndianDialect",
    "singlish": "ColloquialSingaporeDialect",
}


@lru_cache(maxsize=None)
def _get(name: str):
    if name not in DIALECT_CLASSES:
        raise ValueError(f"Unknown dialect {name!r}; expected one of {sorted(DIALECT_CLASSES)}")
    try:
        from multivalue import Dialects
    except ImportError as e:
        raise ImportError(
            "Multi-VALUE is not installed. `pip install value-nlp` (possibly in a "
            "separate env, then pass --dialect-python to build_sets.py)."
        ) from e
    return getattr(Dialects, DIALECT_CLASSES[name])()


def _seed_everything(seed: int | str) -> None:
    s = int(hashlib.sha256(str(seed).encode()).hexdigest()[:8], 16)
    random.seed(s)
    try:
        import numpy as np
        np.random.seed(s)
    except ImportError:
        pass


def dialect(text: str, name: str, seed: int | str = 0) -> str:
    """Transform `text` into dialect `name` with Multi-VALUE (seeded)."""
    _seed_everything(seed)
    try:
        return _get(name).transform(text)
    except ImportError:
        raise
    except Exception:  # Multi-VALUE occasionally fails on odd inputs; keep the original.
        return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream JSONL {seed,text} through a Multi-VALUE dialect.")
    ap.add_argument("--name", required=True, choices=sorted(DIALECT_CLASSES))
    args = ap.parse_args()
    # Multi-VALUE and its deps print to stdout; keep stdout clean for the JSON stream.
    out, sys.stdout = sys.stdout, sys.stderr
    sys.stdin.reconfigure(encoding="utf-8")
    out.reconfigure(encoding="utf-8")
    for line in sys.stdin:
        if line.strip():
            rec = json.loads(line)
            print(json.dumps({"text": dialect(rec["text"], args.name, rec.get("seed", 0))}), file=out, flush=True)


if __name__ == "__main__":
    main()
