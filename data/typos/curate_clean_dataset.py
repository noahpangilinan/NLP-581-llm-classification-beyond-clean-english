#!/usr/bin/env python3
"""Curate a clean, typo-free English sentiment baseline dataset.

Uses the Apache-2.0 licensed `fancyzhx/amazon_polarity` dataset from Hugging Face
(distinct from SST-2, VALUE-SST-2, TweetEval, and SentiMix) and strictly filters
every candidate review against crate-ci/typos to guarantee zero pre-existing typos.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from datasets import load_dataset

from fetch_typo_corpus import (
    DEFAULT_DICT_PATH,
    DEFAULT_ENGLISH_PATH,
    DEFAULT_KNOWN_TYPOS_PATH,
    ensure_typo_corpus,
)

DEFAULT_CLEAN_OUTPUT = (
    Path(__file__).resolve().parent / "data" / "clean" / "amazon_polarity_clean.csv"
)

VALID_CONTRACTIONS: set[str] = {
    "aren't",
    "can't",
    "couldn't",
    "didn't",
    "doesn't",
    "don't",
    "hadn't",
    "hasn't",
    "haven't",
    "he'd",
    "he'll",
    "he's",
    "here's",
    "how's",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "isn't",
    "it'd",
    "it'll",
    "it's",
    "let's",
    "mightn't",
    "mustn't",
    "shan't",
    "she'd",
    "she'll",
    "she's",
    "shouldn't",
    "that's",
    "there's",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "wasn't",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "weren't",
    "what's",
    "where's",
    "who's",
    "won't",
    "wouldn't",
    "you'd",
    "you'll",
    "you're",
    "you've",
}

BAD_SPACING_OR_PUNCT_RE = re.compile(
    r",[A-Za-z]|\.[A-Za-z]|\s[,.!?;:]|\.{4,}|!{3,}|\?{3,}|\s{2,}"
)
WORD_FIND_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def load_word_sets() -> tuple[set[str], set[str]]:
    """Ensure crate-ci/typos wordlists exist and load english_words and known_typos."""
    ensure_typo_corpus(
        dict_path=DEFAULT_DICT_PATH,
        english_path=DEFAULT_ENGLISH_PATH,
        known_typos_path=DEFAULT_KNOWN_TYPOS_PATH,
    )
    with DEFAULT_ENGLISH_PATH.open("r", encoding="utf-8") as f:
        english_words = {line.strip().lower() for line in f if line.strip()}
    with DEFAULT_KNOWN_TYPOS_PATH.open("r", encoding="utf-8") as f:
        known_typos = {line.strip().lower() for line in f if line.strip()}
    return english_words, known_typos


def is_clean_typo_free_review(
    text: str,
    english_words: set[str],
    known_typos: set[str],
    min_words: int = 10,
    max_words: int = 45,
) -> bool:
    """Return True iff `text` is clean ASCII prose with zero pre-existing typos."""
    if not text or not text.isascii():
        return False
    if "\n" in text or "\r" in text or "\t" in text:
        return False
    if BAD_SPACING_OR_PUNCT_RE.search(text):
        return False

    whitespace_tokens = text.split()
    if not (min_words <= len(whitespace_tokens) <= max_words):
        return False

    words = WORD_FIND_RE.findall(text)
    if len(words) < min_words:
        return False

    for word in words:
        wl = word.lower()
        if "'" in wl:
            if wl in VALID_CONTRACTIONS:
                continue
            if wl.endswith("'s") and wl[:-2] in english_words and wl[:-2] not in known_typos:
                continue
            return False
        if wl in known_typos:
            return False
        if wl not in english_words:
            return False

    return True


def curate_amazon_polarity(
    output_path: Path = DEFAULT_CLEAN_OUTPUT,
    num_samples: int = 1000,
    split: str = "test",
    min_words: int = 10,
    max_words: int = 45,
) -> Path:
    """Stream `fancyzhx/amazon_polarity` and save a balanced typo-free dataset."""
    if num_samples % 2 != 0:
        raise ValueError("num_samples must be even for exact 50/50 class balance.")

    per_class_target = num_samples // 2
    english_words, known_typos = load_word_sets()

    print(
        f"Streaming 'fancyzhx/amazon_polarity' ({split} split) to curate "
        f"{num_samples} balanced, verified typo-free reviews..."
    )
    ds = load_dataset("fancyzhx/amazon_polarity", split=split, streaming=True)

    collected: dict[int, list[dict[str, str | int]]] = {0: [], 1: []}
    scanned = 0

    for row in ds:
        scanned += 1
        label = int(row["label"])
        if label not in collected or len(collected[label]) >= per_class_target:
            if all(len(v) >= per_class_target for v in collected.values()):
                break
            continue

        text = str(row["content"]).strip()
        if is_clean_typo_free_review(
            text,
            english_words=english_words,
            known_typos=known_typos,
            min_words=min_words,
            max_words=max_words,
        ):
            collected[label].append({"text": text, "label": label})

        if all(len(v) >= per_class_target for v in collected.values()):
            break

    # Interleave negative (0) and positive (1) examples deterministically
    final_rows: list[dict[str, str | int]] = []
    max_len = max(len(collected[0]), len(collected[1]))
    for idx in range(max_len):
        for lbl in (0, 1):
            if idx < len(collected[lbl]):
                item = collected[lbl][idx]
                final_rows.append(
                    {
                        "id": len(final_rows),
                        "text": item["text"],
                        "label": item["label"],
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "label"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(
        f"Curated {len(final_rows)} typo-free examples "
        f"(neg={len(collected[0])}, pos={len(collected[1])}) "
        f"from {scanned:,} scanned rows -> {output_path}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Curate a clean, verified typo-free sentiment dataset from "
            "fancyzhx/amazon_polarity (Apache-2.0)."
        )
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_CLEAN_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_CLEAN_OUTPUT}).",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=1000,
        help="Total balanced examples to collect (default: 1000 = 500 neg + 500 pos).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=10,
        help="Minimum word count per review (default: 10).",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=45,
        help="Maximum word count per review (default: 45).",
    )
    args = parser.parse_args()
    curate_amazon_polarity(
        output_path=args.output,
        num_samples=args.num_samples,
        min_words=args.min_words,
        max_words=args.max_words,
    )


if __name__ == "__main__":
    main()
