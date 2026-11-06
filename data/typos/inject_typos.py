#!/usr/bin/env python3
"""CLI tool to inject crate-ci/typos into a clean dataset and emit a switch log.

Outputs two separate datasets for each perturbation run:
  1. Perturbed dataset (--output): Identical schema/columns to the clean input,
     with only the text column replaced by the perturbed text.
  2. Word-switch log dataset (--switches-output): One row per switched word,
     recording row_index, word_index, original_word, perturbed_word, character
     spans, edit_type, edit_distance, candidate counts, label, rate, and seed.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Any

from fetch_typo_corpus import DEFAULT_DICT_PATH, ensure_typo_corpus
from typo_injector import (
    SEVERITY_RATES,
    SWITCH_LOG_COLUMNS,
    TypoInjector,
    TypoSwitchRecord,
)

CANDIDATE_TEXT_COLS = ("text", "sentence", "content", "review", "tweet", "verse_text")
CANDIDATE_LABEL_COLS = ("label", "labels", "sentiment", "polarity", "target")


def detect_column(
    columns: list[str],
    explicit: str | None,
    candidates: tuple[str, ...],
    required: bool,
    kind: str,
) -> str | None:
    """Resolve column name explicitly or by matching standard column names."""
    if explicit is not None:
        if explicit not in columns:
            raise ValueError(
                f"Specified {kind} column '{explicit}' not found. "
                f"Available columns: {columns}"
            )
        return explicit

    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]

    if required:
        raise ValueError(
            f"Could not auto-detect {kind} column from {columns}. "
            f"Please pass --{kind}-col explicitly."
        )
    return None


def load_dataset_rows(
    input_path: Path,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Load clean dataset rows from CSV, TSV, JSONL, JSON, or TXT."""
    suffix = input_path.suffix.lower()

    if suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with input_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError(f"Empty tabular file: {input_path}")
            fieldnames = list(reader.fieldnames)
            rows = list(reader)
        return suffix, rows, fieldnames

    if suffix == ".jsonl":
        rows = []
        fieldnames_set: list[str] = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    for k in obj.keys():
                        if k not in fieldnames_set:
                            fieldnames_set.append(k)
                    rows.append(obj)
        return suffix, rows, fieldnames_set

    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list of records in {input_path}")
        fieldnames_set = []
        for obj in data:
            for k in obj.keys():
                if k not in fieldnames_set:
                    fieldnames_set.append(k)
        return suffix, data, fieldnames_set

    if suffix == ".txt":
        with input_path.open("r", encoding="utf-8") as f:
            rows = [{"text": line.rstrip("\r\n")} for line in f]
        return suffix, rows, ["text"]

    raise ValueError(
        f"Unsupported file extension '{suffix}'. Supported: .csv, .tsv, .jsonl, .json, .txt"
    )


def save_dataset_rows(
    output_path: Path,
    fmt: str,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    text_col: str,
) -> None:
    """Write perturbed dataset in the exact format and schema of the input."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_suffix = output_path.suffix.lower() or fmt

    if out_suffix in (".csv", ".tsv"):
        delimiter = "\t" if out_suffix == ".tsv" else ","
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)
        return

    if out_suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    if out_suffix == ".json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        return

    if out_suffix == ".txt":
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(f"{row[text_col]}\n")
        return

    raise ValueError(f"Unsupported output extension '{out_suffix}'.")


def save_switch_log(
    switches_path: Path,
    records: list[TypoSwitchRecord],
) -> None:
    """Save the companion word-switch metadata dataset (.csv, .tsv, or .jsonl)."""
    switches_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = switches_path.suffix.lower()

    if suffix == ".jsonl":
        with switches_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        return

    delimiter = "\t" if suffix == ".tsv" else ","
    with switches_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=SWITCH_LOG_COLUMNS, delimiter=delimiter
        )
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.to_dict())


def default_switches_path(perturbed_output: Path) -> Path:
    """Derive companion switch-log path from the perturbed dataset path."""
    ext = perturbed_output.suffix.lower()
    log_ext = ext if ext in (".csv", ".tsv", ".jsonl") else ".csv"
    return perturbed_output.with_name(f"{perturbed_output.stem}_switches{log_ext}")


def format_run_path(
    template: Path, tag: str, rate: float, is_multi: bool
) -> Path:
    """Resolve output path placeholders ({severity}, {rate}) for batch runs."""
    raw = str(template)
    rate_str = f"{rate:.2f}".rstrip("0").rstrip(".")
    if "{severity}" in raw or "{rate}" in raw:
        return Path(raw.format(severity=tag, rate=rate_str))
    if is_multi:
        return template.with_name(f"{template.stem}_{tag}{template.suffix}")
    return template


def run_single_injection(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    fmt: str,
    text_col: str,
    label_col: str | None,
    typo_dict: dict[str, list[str]],
    rate: float,
    tag: str,
    seed: int | None,
    selection_mode: str,
    output_path: Path,
    switches_path: Path,
) -> dict[str, Any]:
    """Execute typo injection for one target rate and save both output datasets."""
    injector = TypoInjector(
        typo_dict=typo_dict,
        rate=rate,
        seed=seed,
        selection_mode=selection_mode,
    )

    perturbed_rows: list[dict[str, Any]] = []
    all_records: list[TypoSwitchRecord] = []
    total_words = 0
    rows_with_typos = 0

    for row_idx, row in enumerate(rows):
        clean_text = str(row[text_col]) if row[text_col] is not None else ""
        label_val = row.get(label_col) if label_col else None

        total_words += len(clean_text.split())
        pert_text, records = injector.perturb_text(
            clean_text, row_index=row_idx, label=label_val
        )
        if records:
            rows_with_typos += 1
            all_records.extend(records)

        new_row = dict(row)
        new_row[text_col] = pert_text
        perturbed_rows.append(new_row)

    save_dataset_rows(output_path, fmt, perturbed_rows, fieldnames, text_col)
    save_switch_log(switches_path, all_records)

    edit_counts = collections.Counter(r.edit_type for r in all_records)
    actual_wer = (len(all_records) / total_words) if total_words else 0.0

    summary = {
        "tag": tag,
        "target_rate": rate,
        "rows_total": len(rows),
        "rows_perturbed": rows_with_typos,
        "words_total": total_words,
        "words_switched": len(all_records),
        "empirical_word_rate": actual_wer,
        "edit_type_counts": dict(edit_counts),
        "perturbed_output": str(output_path),
        "switches_output": str(switches_path),
    }

    print(
        f"[{tag} | target={rate:.1%}] "
        f"rows_perturbed={rows_with_typos}/{len(rows)} "
        f"words_switched={len(all_records):,}/{total_words:,} ({actual_wer:.2%})\n"
        f"  -> Perturbed dataset: {output_path}\n"
        f"  -> Switch log:        {switches_path}\n"
        f"  -> Edit breakdown:    {dict(edit_counts)}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inject random typos from crate-ci/typos into a clean dataset "
            "and save both the perturbed dataset and a companion word-switch log."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to clean input dataset (.csv, .tsv, .jsonl, .json, .txt).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help=(
            "Path to output perturbed dataset. Supports {severity} or {rate} "
            "placeholders when running multiple levels."
        ),
    )
    parser.add_argument(
        "--switches-output",
        type=Path,
        default=None,
        help=(
            "Optional path for the companion word-switch dataset. Defaults to "
            "<output_stem>_switches.csv (also supports {severity} / {rate})."
        ),
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default=None,
        help="Name of text column to perturb (auto-detected if omitted).",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Optional label column to record in the switch log (auto-detected if omitted).",
    )

    rate_group = parser.add_mutually_exclusive_group()
    rate_group.add_argument(
        "--rate",
        type=float,
        default=None,
        help="Word perturbation rate in [0.0, 1.0] (e.g., 0.15).",
    )
    rate_group.add_argument(
        "--severity",
        type=str,
        choices=list(SEVERITY_RATES.keys()),
        default=None,
        help="Named severity preset: low (5%%), medium (15%%), high (30%%).",
    )
    rate_group.add_argument(
        "--severities",
        type=str,
        default=None,
        help="Comma-separated severity presets for batch generation (e.g., low,medium,high).",
    )
    rate_group.add_argument(
        "--rates",
        type=str,
        default=None,
        help="Comma-separated float rates for batch generation (e.g., 0.05,0.15,0.30).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible typo injection (default: 42).",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["proportional", "bernoulli"],
        default="proportional",
        help=(
            "Word selection strategy: 'proportional' (default, stochastic rounding "
            "of sentence_words * rate) or 'bernoulli' (independent coin flip per eligible word)."
        ),
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=DEFAULT_DICT_PATH,
        help=f"Path to cached crate-ci/typos JSON dictionary (default: {DEFAULT_DICT_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fmt, rows, fieldnames = load_dataset_rows(args.input)
    text_col = detect_column(
        fieldnames, args.text_col, CANDIDATE_TEXT_COLS, required=True, kind="text"
    )
    assert text_col is not None
    label_col = detect_column(
        fieldnames, args.label_col, CANDIDATE_LABEL_COLS, required=False, kind="label"
    )

    typo_dict = ensure_typo_corpus(dict_path=args.corpus_path)

    runs: list[tuple[str, float]] = []
    if args.severities:
        for sev in [s.strip().lower() for s in args.severities.split(",") if s.strip()]:
            if sev not in SEVERITY_RATES:
                raise ValueError(f"Unknown severity '{sev}'. Choose from {list(SEVERITY_RATES)}")
            runs.append((sev, SEVERITY_RATES[sev]))
    elif args.rates:
        for r_str in [s.strip() for s in args.rates.split(",") if s.strip()]:
            r_val = float(r_str)
            if not (0.0 <= r_val <= 1.0):
                raise ValueError(f"Rate must be in [0.0, 1.0], got {r_val}")
            tag = f"rate_{r_str}"
            runs.append((tag, r_val))
    elif args.severity:
        runs.append((args.severity, SEVERITY_RATES[args.severity]))
    elif args.rate is not None:
        if not (0.0 <= args.rate <= 1.0):
            raise ValueError(f"Rate must be in [0.0, 1.0], got {args.rate}")
        runs.append((f"rate_{args.rate}", args.rate))
    else:
        runs.append(("medium", SEVERITY_RATES["medium"]))

    is_multi = len(runs) > 1
    for tag, rate in runs:
        out_path = format_run_path(args.output, tag, rate, is_multi)
        if args.switches_output is not None:
            sw_path = format_run_path(args.switches_output, tag, rate, is_multi)
        else:
            sw_path = default_switches_path(out_path)

        run_single_injection(
            rows=rows,
            fieldnames=fieldnames,
            fmt=fmt,
            text_col=text_col,
            label_col=label_col,
            typo_dict=typo_dict,
            rate=rate,
            tag=tag,
            seed=args.seed,
            selection_mode=args.selection_mode,
            output_path=out_path,
            switches_path=sw_path,
        )


if __name__ == "__main__":
    main()
