#!/usr/bin/env python3
"""Assemble the unified LING-581 Mega Dataset and 50-row manual audit sample.

Combines all curated splits across `data/clean/`, `data/perturbed/`, and
`data/codeswitch/` into:
  1. `data/mega/ling581_mega_dataset.csv`: Single unified dataset preserving
     `id`, `text`, and `label` alongside `phenomenon`, `variant`, `source_dataset`,
     `lang_pair`, `severity`, and `clean_text`.
  2. `data/audit/manual_label_check_50.csv`: Stratified 50-example verification
     sheet across typo severities and code-switching sources for human or LLM
     label-validity checking.
"""

from __future__ import annotations

import argparse
import collections
import csv
import random
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TYPOS_DIR = PROJECT_DIR / "typos" / "data"
DEFAULT_CODESWITCH_DIR = PROJECT_DIR / "codeswitching" / "data"
DEFAULT_MEGA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MEGA_OUTPUT = DEFAULT_MEGA_DIR / "ling581_mega_dataset.csv"
DEFAULT_AUDIT_OUTPUT = DEFAULT_MEGA_DIR / "manual_label_check_50.csv"

MEGA_COLUMNS: list[str] = [
    "id",
    "source_id",
    "phenomenon",
    "variant",
    "source_dataset",
    "lang_pair",
    "severity",
    "text",
    "clean_text",
    "label",
]

AUDIT_COLUMNS: list[str] = [
    "audit_id",
    "mega_id",
    "phenomenon",
    "variant",
    "lang_pair",
    "clean_text",
    "text",
    "original_label",
    "meaning_preserved",
    "manual_sentiment_label",
    "notes",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load rows from a UTF-8 CSV file if it exists, else return an empty list."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    """Write rows to a UTF-8 CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_mega_rows(
    typos_dir: Path = DEFAULT_TYPOS_DIR,
    codeswitch_dir: Path = DEFAULT_CODESWITCH_DIR,
) -> list[dict[str, Any]]:
    """Gather clean, typo-perturbed, and code-switched splits into mega schema."""
    mega_rows: list[dict[str, Any]] = []

    # 1. Clean baseline (amazon_polarity_clean.csv)
    clean_path = typos_dir / "clean" / "amazon_polarity_clean.csv"
    clean_rows = read_csv_rows(clean_path)
    clean_lookup: dict[str, str] = {}
    for r in clean_rows:
        sid = str(r.get("id", ""))
        txt = str(r.get("text", ""))
        clean_lookup[sid] = txt
        mega_rows.append(
            {
                "source_id": sid,
                "phenomenon": "clean",
                "variant": "clean_baseline",
                "source_dataset": "fancyzhx/amazon_polarity",
                "lang_pair": "en",
                "severity": "none",
                "text": txt,
                "clean_text": txt,
                "label": r.get("label", ""),
            }
        )

    # 2. Typo-perturbed splits (low, medium, high)
    for sev in ("low", "medium", "high"):
        pert_path = typos_dir / "perturbed" / f"amazon_polarity_typos_{sev}.csv"
        for r in read_csv_rows(pert_path):
            sid = str(r.get("id", ""))
            mega_rows.append(
                {
                    "source_id": sid,
                    "phenomenon": "typos",
                    "variant": f"typos_{sev}",
                    "source_dataset": "fancyzhx/amazon_polarity+crate-ci/typos",
                    "lang_pair": "en",
                    "severity": sev,
                    "text": r.get("text", ""),
                    "clean_text": clean_lookup.get(sid, ""),
                    "label": r.get("label", ""),
                }
            )

    # 3. Code-switching splits
    cs_sources = [
        (
            "covoswitch",
            "sophiayk20/covoswitch",
            codeswitch_dir / "covoswitch_multilingual_metadata.csv",
            codeswitch_dir / "covoswitch_multilingual.csv",
            "multilingual",
        ),
        (
            "parsing_project",
            "N3mika/ParsingProject",
            codeswitch_dir / "parsing_project_codeswitch_metadata.csv",
            codeswitch_dir / "parsing_project_codeswitch.csv",
            "es_en",
        ),
        (
            "switchlingua",
            "Shelton1013/SwitchLingua_text",
            codeswitch_dir / "switchlingua_multilingual_metadata.csv",
            codeswitch_dir / "switchlingua_multilingual.csv",
            "multilingual",
        ),
    ]

    for variant, dataset_name, meta_path, std_path, default_pair in cs_sources:
        rows = read_csv_rows(meta_path)
        if not rows:
            rows = read_csv_rows(std_path)
        for r in rows:
            mega_rows.append(
                {
                    "source_id": r.get("id", ""),
                    "phenomenon": "codeswitch",
                    "variant": variant,
                    "source_dataset": dataset_name,
                    "lang_pair": r.get("lang_pair", default_pair),
                    "severity": "none",
                    "text": r.get("text", ""),
                    "clean_text": r.get("clean_text", ""),
                    "label": r.get("label", ""),
                }
            )

    for idx, row in enumerate(mega_rows):
        row["id"] = idx

    return mega_rows


def sample_audit_sheet(
    mega_rows: list[dict[str, Any]],
    sample_size: int = 50,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create a stratified ~50-row manual verification sheet across perturbations."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in mega_rows:
        if r["phenomenon"] == "clean":
            continue
        # For typos, prefer rows where text actually changed from clean_text
        if r["phenomenon"] == "typos" and r.get("clean_text") == r.get("text"):
            continue
        buckets[str(r["variant"])].append(r)

    if not buckets:
        return []

    variants = sorted(buckets.keys())
    per_bucket = max(1, sample_size // len(variants))
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []

    for var in variants:
        pool = list(buckets[var])
        rng.shuffle(pool)
        take = min(per_bucket, len(pool))
        selected.extend(pool[:take])
        leftovers.extend(pool[take:])

    if len(selected) < sample_size and leftovers:
        rng.shuffle(leftovers)
        needed = sample_size - len(selected)
        selected.extend(leftovers[:needed])

    selected = selected[:sample_size]
    audit_rows: list[dict[str, Any]] = []
    for audit_idx, r in enumerate(selected):
        audit_rows.append(
            {
                "audit_id": audit_idx,
                "mega_id": r["id"],
                "phenomenon": r["phenomenon"],
                "variant": r["variant"],
                "lang_pair": r["lang_pair"],
                "clean_text": r.get("clean_text", ""),
                "text": r["text"],
                "original_label": r.get("label", ""),
                "meaning_preserved": "",
                "manual_sentiment_label": "",
                "notes": "",
            }
        )
    return audit_rows


def build_mega_and_audit(
    typos_dir: Path = DEFAULT_TYPOS_DIR,
    codeswitch_dir: Path = DEFAULT_CODESWITCH_DIR,
    mega_output: Path = DEFAULT_MEGA_OUTPUT,
    audit_output: Path = DEFAULT_AUDIT_OUTPUT,
    audit_size: int = 50,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build and write both the mega dataset CSV and the manual audit CSV."""
    mega_rows = collect_mega_rows(typos_dir=typos_dir, codeswitch_dir=codeswitch_dir)
    write_csv_rows(mega_output, mega_rows, MEGA_COLUMNS)

    audit_rows = sample_audit_sheet(mega_rows, sample_size=audit_size, seed=seed)
    write_csv_rows(audit_output, audit_rows, AUDIT_COLUMNS)

    counts = collections.Counter(r["variant"] for r in mega_rows)
    print(f"[Mega Dataset] Saved {len(mega_rows):,} rows -> {mega_output}")
    for var, cnt in counts.items():
        print(f"  - {var}: {cnt:,} rows")
    print(f"[Manual Audit] Saved {len(audit_rows)} stratified rows -> {audit_output}")
    return mega_rows, audit_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble the unified LING-581 mega dataset and 50-row manual audit CSV."
    )
    parser.add_argument(
        "--typos-dir",
        type=Path,
        default=DEFAULT_TYPOS_DIR,
        help=f"Directory for typo datasets (default: {DEFAULT_TYPOS_DIR}).",
    )
    parser.add_argument(
        "--codeswitch-dir",
        type=Path,
        default=DEFAULT_CODESWITCH_DIR,
        help=f"Directory for code-switching datasets (default: {DEFAULT_CODESWITCH_DIR}).",
    )
    parser.add_argument(
        "--mega-output",
        type=Path,
        default=DEFAULT_MEGA_OUTPUT,
        help=f"Output mega CSV path (default: {DEFAULT_MEGA_OUTPUT}).",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DEFAULT_AUDIT_OUTPUT,
        help=f"Output manual audit CSV path (default: {DEFAULT_AUDIT_OUTPUT}).",
    )
    parser.add_argument(
        "--audit-size",
        type=int,
        default=50,
        help="Number of rows to sample for manual label verification (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible audit sampling (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_mega_and_audit(
        typos_dir=args.typos_dir,
        codeswitch_dir=args.codeswitch_dir,
        mega_output=args.mega_output,
        audit_output=args.audit_output,
        audit_size=args.audit_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

