"""Shared JSONL schema for evaluation data (agreed with the datasets owner).

One example per line:
    {"id", "source_id", "task", "split", "text", "label",
     "perturbation", "severity", "dialect", "origin"}
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from harness.config import PERTURBATIONS, TYPO_TYPES

ORIGINS = ("synthetic", "natural")
FIELDS = ("id", "source_id", "task", "split", "text", "label",
          "perturbation", "severity", "dialect", "origin")


class SchemaError(ValueError):
    pass


def validate_row(row: dict[str, Any], labels: Iterable[str] | None = None) -> None:
    """Raise SchemaError if `row` violates the schema. `labels` restricts `label`."""
    missing = [f for f in FIELDS if f not in row]
    if missing:
        raise SchemaError(f"missing fields {missing}: {row.get('id')}")
    rid = row["id"]
    for f in ("id", "task", "split", "text", "label"):
        if not isinstance(row[f], str) or (f != "text" and not row[f]):
            raise SchemaError(f"{rid}: {f} must be a non-empty string")
    if row["source_id"] is not None and not isinstance(row["source_id"], str):
        raise SchemaError(f"{rid}: source_id must be a string or null")
    pert = row["perturbation"]
    if pert not in PERTURBATIONS:
        raise SchemaError(f"{rid}: perturbation {pert!r} not in {PERTURBATIONS}")
    sev = row["severity"]
    if pert in TYPO_TYPES:
        if isinstance(sev, bool) or not isinstance(sev, (int, float)) or not 0 <= sev <= 1:
            raise SchemaError(f"{rid}: severity must be a float in [0, 1] for {pert}")
    elif sev is not None:
        raise SchemaError(f"{rid}: severity must be null for {pert}")
    if row["dialect"] is not None and not isinstance(row["dialect"], str):
        raise SchemaError(f"{rid}: dialect must be a string or null")
    if pert == "dialect" and not row["dialect"]:
        raise SchemaError(f"{rid}: dialect rows need a dialect name")
    if row["origin"] not in ORIGINS:
        raise SchemaError(f"{rid}: origin must be one of {ORIGINS}")
    if labels is not None and row["label"] not in set(labels):
        raise SchemaError(f"{rid}: label {row['label']!r} not in {sorted(labels)}")


def make_row(*, id: str, task: str, split: str, text: str, label: str,
             source_id: str | None = None, perturbation: str = "none",
             severity: float | None = None, dialect: str | None = None,
             origin: str = "natural") -> dict[str, Any]:
    """Build a schema row with fields in canonical order."""
    return {"id": id, "source_id": source_id, "task": task, "split": split, "text": text,
            "label": label, "perturbation": perturbation, "severity": severity,
            "dialect": dialect, "origin": origin}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write rows atomically (tmp file + rename) so partial files never look complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def validate_file(path: str | Path, labels: Iterable[str] | None = None) -> int:
    """Validate every row of a JSONL file; return the row count."""
    rows = read_jsonl(path)
    ids = set()
    for r in rows:
        validate_row(r, labels)
        if r["id"] in ids:
            raise SchemaError(f"{path}: duplicate id {r['id']}")
        ids.add(r["id"])
    return len(rows)
