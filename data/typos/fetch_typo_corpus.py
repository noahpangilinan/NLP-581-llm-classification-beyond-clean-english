#!/usr/bin/env python3
"""Download and compile the crate-ci/typos dictionary for typo injection.

Source repository: https://github.com/crate-ci/typos
Assets used:
  - crates/typos-dict/assets/words.csv   (typo -> correction1, correction2, ...)
  - crates/typos-dict/assets/english.csv (valid English wordlist for baseline curation)
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

WORDS_CSV_URL = (
    "https://raw.githubusercontent.com/crate-ci/typos/main/"
    "crates/typos-dict/assets/words.csv"
)
ENGLISH_CSV_URL = (
    "https://raw.githubusercontent.com/crate-ci/typos/main/"
    "crates/typos-dict/assets/english.csv"
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DICT_PATH = DEFAULT_DATA_DIR / "crate_ci_typos_dict.json"
DEFAULT_ENGLISH_PATH = DEFAULT_DATA_DIR / "crate_ci_english_words.txt"
DEFAULT_KNOWN_TYPOS_PATH = DEFAULT_DATA_DIR / "crate_ci_known_typos.txt"


def download_text(url: str, timeout: int = 30) -> str:
    """Download UTF-8 text content from a URL."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ling581-typo-dataset-curator/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def parse_words_csv(raw_csv: str) -> tuple[dict[str, list[str]], set[str]]:
    """Invert crate-ci/typos words.csv into {clean_word: [typo1, typo2, ...]}.

    Each line in words.csv has the format:
        typo,correction1[,correction2,...]

    Single-character clean words (e.g., 'x' -> 'xwindows') are excluded to avoid
    source-code identifier artifacts.
    """
    clean_to_typos: dict[str, set[str]] = {}
    known_typos: set[str] = set()

    for raw_line in raw_csv.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip().lower() for p in line.split(",") if p.strip()]
        if len(parts) < 2:
            continue

        typo = parts[0]
        if not typo.isalpha():
            continue
        known_typos.add(typo)

        for corr in parts[1:]:
            # Exclude single-letter words ('x' -> 'xwindows') and identical pairs
            if len(corr) < 2 or not corr.isalpha() or corr == typo:
                continue
            clean_to_typos.setdefault(corr, set()).add(typo)

    sorted_dict = {
        word: sorted(typos) for word, typos in sorted(clean_to_typos.items())
    }
    return sorted_dict, known_typos


def ensure_typo_corpus(
    dict_path: Path = DEFAULT_DICT_PATH,
    english_path: Path = DEFAULT_ENGLISH_PATH,
    known_typos_path: Path = DEFAULT_KNOWN_TYPOS_PATH,
    force_download: bool = False,
) -> dict[str, list[str]]:
    """Load cached crate-ci/typos dictionary, downloading and compiling if needed."""
    if dict_path.exists() and not force_download:
        with dict_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    dict_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading crate-ci/typos dictionary from:\n  {WORDS_CSV_URL}")
    raw_words_csv = download_text(WORDS_CSV_URL)
    typo_dict, known_typos = parse_words_csv(raw_words_csv)

    with dict_path.open("w", encoding="utf-8") as f:
        json.dump(typo_dict, f, indent=2, sort_keys=True)

    known_typos_path.parent.mkdir(parents=True, exist_ok=True)
    with known_typos_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(sorted(known_typos)) + "\n")

    if force_download or not english_path.exists():
        print(f"Downloading crate-ci/typos English wordlist from:\n  {ENGLISH_CSV_URL}")
        raw_english = download_text(ENGLISH_CSV_URL)
        english_words = sorted(
            {
                line.strip().lower()
                for line in raw_english.splitlines()
                if line.strip() and line.strip().isalpha()
            }
        )
        english_path.parent.mkdir(parents=True, exist_ok=True)
        with english_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(english_words) + "\n")

    total_pairs = sum(len(v) for v in typo_dict.values())
    print(
        f"Saved compiled dictionary to {dict_path}\n"
        f"  Unique clean words: {len(typo_dict):,}\n"
        f"  Total typo pairs:   {total_pairs:,}"
    )
    return typo_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and compile the crate-ci/typos dictionary."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DICT_PATH,
        help=f"Output JSON dictionary path (default: {DEFAULT_DICT_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-downloading even if cached locally.",
    )
    args = parser.parse_args()
    ensure_typo_corpus(dict_path=args.output, force_download=args.force)


if __name__ == "__main__":
    main()
