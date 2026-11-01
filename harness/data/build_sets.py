"""CLI: build perturbed JSONL evaluation sets from the clean sets.

    python -m harness.data.build_sets --task sentiment offensive --grid configs/grid_main.yaml
    python -m harness.data.build_sets --task sentiment --conditions none typo:0.10 dialect:aave

Writes data/processed/{task}/{condition}.jsonl plus a spot-check CSV of
clean/perturbed pairs per condition under data/processed/{task}/spotcheck/.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import subprocess
from pathlib import Path

from harness.config import Condition, load_grid, load_tasks
from harness.data.loaders import Row, build_condition, condition_path, load_jsonl
from harness.data.schema import validate_file, write_jsonl

log = logging.getLogger("build_sets")
DIALECT_SCRIPT = Path(__file__).resolve().parents[1] / "perturb" / "dialect.py"


def subprocess_dialect(python: str):
    """Return a dialect fn that runs Multi-VALUE under another interpreter."""
    def fn(texts: list[str], name: str, seeds: list[str]) -> list[str]:
        payload = "".join(json.dumps({"text": t, "seed": s}) + "\n" for t, s in zip(texts, seeds))
        out = subprocess.run([python, "-I", str(DIALECT_SCRIPT), "--name", name], input=payload,
                             capture_output=True, text=True, encoding="utf-8", check=True)
        result = [json.loads(l)["text"] for l in out.stdout.splitlines() if l.startswith('{"text"')]
        if len(result) != len(texts):
            raise RuntimeError(f"dialect subprocess returned {len(result)} rows for {len(texts)} inputs")
        return result
    return fn


def write_spotcheck(clean: list[Row], pert: list[Row], path: Path, n: int, seed: int) -> None:
    """Write `n` random clean/perturbed pairs for manual label verification."""
    by_src = {r["source_id"]: r for r in clean}
    pairs = [(by_src.get(p["source_id"]), p) for p in pert]
    pairs = random.Random(seed).sample(pairs, min(n, len(pairs)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "clean_text", "perturbed_text", "label_still_correct"])
        for c, p in pairs:
            w.writerow([p["id"], p["label"], c["text"] if c else "", p["text"], ""])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", default=["sentiment", "offensive"])
    ap.add_argument("--split", default="test")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--grid", help="Take conditions from a grid YAML.")
    g.add_argument("--conditions", nargs="+", help="Specs like none typo:0.10 dialect:aave code_switch:hinglish")
    ap.add_argument("--limit", type=int, default=1000, help="Stratified subsample size per condition.")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--dialect-python", help="Python interpreter with Multi-VALUE installed.")
    ap.add_argument("--spotcheck", type=int, default=50, help="Pairs per condition in the spot-check CSV.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tasks = load_tasks()
    dialect_fn = subprocess_dialect(args.dialect_python) if args.dialect_python else None
    grid = load_grid(args.grid) if args.grid else None
    failed = []
    for tname in args.task:
        task = tasks[tname]
        if grid:
            conds = grid.conditions_for(tname)
        else:
            conds = [Condition.parse(s) for s in (args.conditions or ["none", "typo:0.10"])]
        # Clean first (perturbed sets derive from it), then grid order, deduplicated.
        conds = list(dict.fromkeys([Condition("none"), *conds]))

        clean = None
        for cond in conds:
            path = condition_path(args.out, tname, cond)
            try:
                if path.exists() and not args.overwrite:
                    log.info("exists, skipping: %s", path)
                    rows = load_jsonl(path, task)
                else:
                    rows = build_condition(task, cond, clean, seed=args.seed, limit=args.limit,
                                           split=args.split, dialect_fn=dialect_fn)
                    write_jsonl(rows, path)
                    log.info("wrote %d rows -> %s", len(rows), path)
            except (FileNotFoundError, ImportError) as e:
                log.error("%s/%s: %s", tname, cond.name, e)
                failed.append(f"{tname}/{cond.name}")
                continue
            if cond.perturbation == "none":
                clean = rows
            elif clean is not None and args.spotcheck:
                write_spotcheck(clean, rows, path.parent / "spotcheck" / f"{cond.name}.csv",
                                args.spotcheck, args.seed)
            n = validate_file(path, task.labels)
            log.info("validated %s (%d rows)", path.name, n)

    if failed:
        log.warning("could not build: %s", ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
