"""Loaders for TweetEval, SentiMix, VALUE, and schema JSONL.

All loaders return lists of schema-valid dicts. `get_condition_rows` is the single
entry point used by the runner: it reads a prebuilt JSONL if one exists and
otherwise generates the condition on the fly from the clean set.
"""

from __future__ import annotations

import csv
import logging
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harness.config import Condition, TaskConfig, resolve
from harness.data.schema import make_row, read_jsonl, validate_row, write_jsonl

log = logging.getLogger(__name__)

Row = dict[str, Any]
DialectFn = Callable[[list[str], str, list[str]], list[str]]   # (texts, name, seeds) -> texts


# ---------------------------------------------------------------- raw sources

def load_tweeteval(task: TaskConfig, split: str) -> list[Row]:
    """Load a TweetEval split (train | validation | test) as clean rows."""
    from datasets import load_dataset

    src = task.source
    ds = load_dataset(src["hf_id"], src["hf_config"], split=split)
    names = src["label_names"]
    short = {"validation": "val"}.get(split, split)
    rows = []
    for i, ex in enumerate(ds):
        rid = f"{task.id_prefix}-{short}-{i:05d}"
        rows.append(make_row(id=rid, source_id=rid, task=task.name, split=split,
                             text=ex["text"], label=names[ex["label"]]))
    return rows


def load_sentimix(path: str | Path, task: TaskConfig, lang: str) -> list[Row]:
    """Load SemEval-2020 SentiMix CoNLL-style files.

    Each tweet starts with `meta<TAB>id<TAB>label` followed by `token<TAB>lang`
    lines and a blank separator line.
    """
    rows, tokens, meta = [], [], None

    def flush() -> None:
        if meta and tokens:
            uid, label = meta
            rows.append(make_row(id=f"{task.id_prefix}-cs-{lang}-{uid}", task=task.name, split="test",
                                 text=" ".join(tokens), label=label.lower(),
                                 perturbation="code_switch", origin="natural"))

    with open(resolve(path), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "meta":
                flush()
                meta, tokens = (parts[1], parts[2] if len(parts) > 2 else ""), []
            elif parts[0].strip():
                tokens.append(parts[0])
    flush()
    return [r for r in rows if r["label"] in task.labels]


def load_value_sst2(path: str | Path, task: TaskConfig, dialect: str) -> list[Row]:
    """Load a VALUE-SST-2 TSV (GLUE format: `sentence`, `label` with 0=neg, 1=pos).

    Note: SST-2 is binary, so these rows never carry the `neutral` label.
    """
    names = {"0": "negative", "1": "positive"}
    rows = []
    with open(resolve(path), encoding="utf-8", newline="") as f:
        for i, rec in enumerate(csv.DictReader(f, delimiter="\t")):
            rows.append(make_row(id=f"{task.id_prefix}-value-{dialect}-{i:05d}", task=task.name,
                                 split="test", text=rec["sentence"], label=names[str(rec["label"])],
                                 perturbation="dialect", dialect=dialect, origin="synthetic"))
    return rows


def load_jsonl(path: str | Path, task: TaskConfig | None = None) -> list[Row]:
    """Read and validate a schema JSONL file (e.g. a prebuilt set from Gyan)."""
    rows = read_jsonl(resolve(path))
    for r in rows:
        validate_row(r, task.labels if task else None)
    return rows


NATURAL_LOADERS = {"sentimix": load_sentimix, "value_sst2": load_value_sst2}


def load_natural(task: TaskConfig, kind: str, name: str) -> list[Row]:
    """Load a natural-variation set declared under `natural.<kind>.<name>` in tasks.yaml."""
    spec = dict(task.natural.get(kind, {}).get(name) or {})
    if not spec:
        raise FileNotFoundError(f"No natural {kind} source {name!r} for task {task.name}")
    loader = spec.pop("loader")
    path = spec.pop("path")
    if not resolve(path).exists():
        raise FileNotFoundError(f"{kind}/{name} source file missing: {resolve(path)}")
    if loader == "sentimix":
        return load_sentimix(path, task, lang=name)
    return NATURAL_LOADERS[loader](path, task, **spec)


# ---------------------------------------------------------------- subsampling

def stratified_subsample(rows: list[Row], n: int | None, seed: int) -> list[Row]:
    """Seeded label-stratified subsample of `rows`, kept in original order.

    Allocation is proportional with largest-remainder rounding, so row-aligned
    sets (clean and its perturbed copies) always select the same positions.
    """
    if n is None or n >= len(rows):
        return list(rows)
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_label[r["label"]].append(i)
    labels = sorted(by_label)
    quotas = {l: n * len(by_label[l]) / len(rows) for l in labels}
    alloc = {l: int(quotas[l]) for l in labels}
    for l in sorted(labels, key=lambda l: (alloc[l] - quotas[l], l))[: n - sum(alloc.values())]:
        alloc[l] += 1
    rng = random.Random(seed)
    keep = sorted(i for l in labels for i in rng.sample(by_label[l], alloc[l]))
    return [rows[i] for i in keep]


# ---------------------------------------------------------------- perturbation

def _local_dialect(texts: list[str], name: str, seeds: list[str]) -> list[str]:
    from harness.perturb.dialect import dialect
    return [dialect(t, name, s) for t, s in zip(texts, seeds)]


def perturb_rows(clean: list[Row], cond: Condition, seed: int,
                 typo_map: dict[str, list[str]] | None = None,
                 dialect_fn: DialectFn | None = None) -> list[Row]:
    """Derive a perturbed set from clean rows. Each row gets its own stable seed."""
    from harness.perturb.natural_typo import load_typo_map, natural_typo
    from harness.perturb.typo import typo

    seeds = [f"{seed}:{cond.name}:{r['source_id']}" for r in clean]
    texts = [r["text"] for r in clean]
    if cond.perturbation == "typo":
        new = [typo(t, cond.severity, s) for t, s in zip(texts, seeds)]
        suffix = f"typo-p{cond.severity:.2f}"
    elif cond.perturbation == "natural_typo":
        tm = typo_map if typo_map is not None else load_typo_map()
        new = [natural_typo(t, cond.severity, s, tm) for t, s in zip(texts, seeds)]
        suffix = f"natural_typo-p{cond.severity:.2f}"
    elif cond.perturbation == "dialect":
        new = (dialect_fn or _local_dialect)(texts, cond.dialect, seeds)
        suffix = f"dialect-{cond.dialect}"
    else:
        raise ValueError(f"Cannot generate {cond.name} from clean data")

    origin = "natural" if cond.perturbation == "natural_typo" else "synthetic"
    return [make_row(id=f"{r['source_id']}-{suffix}", source_id=r["source_id"], task=r["task"],
                     split=r["split"], text=t, label=r["label"], perturbation=cond.perturbation,
                     severity=cond.severity, dialect=cond.dialect, origin=origin)
            for r, t in zip(clean, new)]


# ---------------------------------------------------------------- entry point

def condition_path(data_dir: str | Path, task: str, cond: Condition) -> Path:
    return resolve(data_dir) / task / f"{cond.name}.jsonl"


def build_condition(task: TaskConfig, cond: Condition, clean: list[Row] | None, *,
                    seed: int, limit: int | None, split: str = "test",
                    dialect_fn: DialectFn | None = None) -> list[Row]:
    """Build rows for one condition. Perturbed conditions are derived from `clean`."""
    if cond.perturbation == "none":
        return stratified_subsample(load_tweeteval(task, split), limit, seed)
    if cond.perturbation == "code_switch":
        return stratified_subsample(load_natural(task, "code_switch", cond.lang or ""), limit, seed)
    if cond.perturbation == "dialect" and cond.dialect in task.natural.get("dialect", {}):
        return stratified_subsample(load_natural(task, "dialect", cond.dialect), limit, seed)
    if clean is None:
        raise ValueError(f"{cond.name} needs the clean set")
    return perturb_rows(clean, cond, seed, dialect_fn=dialect_fn)


def get_condition_rows(task: TaskConfig, cond: Condition, *, data_dir: str | Path = "data/processed",
                       seed: int = 13, limit: int | None = 1000, build_limit: int | None = 1000,
                       split: str = "test", generate: bool = True,
                       dialect_fn: DialectFn | None = None) -> list[Row]:
    """Rows for one condition: prebuilt JSONL if present, else generated and cached.

    Generated files always hold `build_limit` rows (the canonical set) and are
    subsampled to `limit` on return, so small runs never truncate the cache.
    Row-aligned sets subsample to the same positions, so `source_id`s still match.
    """
    path = condition_path(data_dir, task.name, cond)
    if not path.exists():
        if not generate:
            raise FileNotFoundError(path)
        clean = None
        if cond.perturbation in ("typo", "natural_typo", "dialect"):
            clean = get_condition_rows(task, Condition("none"), data_dir=data_dir, seed=seed,
                                       limit=None, build_limit=build_limit, split=split)
        rows = build_condition(task, cond, clean, seed=seed, limit=build_limit, split=split,
                               dialect_fn=dialect_fn)
        write_jsonl(rows, path)
        log.info("wrote %d rows -> %s", len(rows), path)
    rows = load_jsonl(path, task)
    if limit is not None and len(rows) < limit and cond.perturbation != "code_switch":
        log.warning("%s has %d rows (< limit %d)", path, len(rows), limit)
    return stratified_subsample(rows, limit, seed)
