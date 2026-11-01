"""Natural typos: substitute real misspellings mined from the GitHub Typo Corpus.

Build the map once and cache it:
    python -m harness.perturb.natural_typo                       # from the HF mirror
    python -m harness.perturb.natural_typo --corpus PATH.jsonl.gz # from the original release

The original S3 download link in the corpus README is dead (404), so the default
source is an HF mirror of the corpus as (text, correction) pairs.
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path

from harness.config import resolve
from harness.perturb.tokens import map_words

HF_MIRROR = "chirunder/github_typo_corrections"
DEFAULT_MAP = "data/cache/typo_map.json"

_TOK = re.compile(r"[A-Za-z]+|[^A-Za-z\s]+")


def _match_case(src: str, typo_lower: str) -> str:
    if src.isupper() and len(src) > 1:
        return typo_lower.upper()
    if src[0].isupper():
        return typo_lower[:1].upper() + typo_lower[1:]
    return typo_lower


def natural_typo(text: str, p: float, seed: int | str, typo_map: dict[str, list[str]]) -> str:
    """Replace each eligible word, with probability p, by a real misspelling.

    Words with no entry in `typo_map` are left unchanged, so the observed
    perturbation rate is at most p.
    """
    if p <= 0:
        return text
    rng = random.Random(seed)

    def fn(w: str) -> str:
        if rng.random() >= p:
            return w
        options = typo_map.get(w.lower())
        return _match_case(w, rng.choice(options)) if options else w

    return map_words(text, fn)


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def word_pairs(src: str, tgt: str, max_dist: int = 2) -> list[tuple[str, str]]:
    """Extract (correct, misspelled) word pairs from a typo edit src -> tgt."""
    a, b = _TOK.findall(src), _TOK.findall(tgt)
    pairs = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "replace" or i2 - i1 != 1 or j2 - j1 != 1:
            continue
        bad, good = a[i1].lower(), b[j1].lower()
        if (bad != good and bad.isalpha() and good.isalpha() and len(good) >= 2
                and _edit_distance(bad, good) <= max_dist):
            pairs.append((good, bad))
    return pairs


def _corpus_edits(corpus_path: str | Path, lang: str = "eng") -> Iterator[tuple[str, str]]:
    """Yield (src, tgt) typo edits from the original GitHub Typo Corpus JSONL(.gz)."""
    path = resolve(corpus_path)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            for edit in json.loads(line).get("edits", []):
                is_typo = edit.get("is_typo", edit.get("prob_typo", 0) >= 0.5)
                if is_typo and edit["src"].get("lang") == lang:
                    yield edit["src"]["text"], edit["tgt"]["text"]


def _mirror_edits(hf_id: str = HF_MIRROR) -> Iterator[tuple[str, str]]:
    """Yield (src, tgt) edits from the HF mirror (columns: text, correction)."""
    from datasets import load_dataset

    for row in load_dataset(hf_id, split="train"):
        yield row["text"], row["correction"]


def build_typo_map(edits: Iterable[tuple[str, str]], min_count: int = 1,
                   vocab_min: int = 3) -> dict[str, list[str]]:
    """Mine {correct_word: [misspellings]} from (misspelled_src, corrected_tgt) edits.

    Real-word substitutions (grammar or word-choice edits such as going -> doing)
    are dropped: a misspelling is rejected if it occurs at least `vocab_min`
    times as a word in corrected text. Words of <= 3 letters only keep
    single-edit misspellings.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    vocab: Counter = Counter()
    for src, tgt in edits:
        vocab.update(w.lower() for w in re.findall(r"[A-Za-z]+", tgt))
        for good, bad in word_pairs(src, tgt):
            counts[good][bad] += 1
    out = {}
    for good in sorted(counts):
        bads = sorted(b for b, c in counts[good].items()
                      if c >= min_count and vocab[b] < vocab_min
                      and (len(good) > 3 or _edit_distance(b, good) == 1))
        if bads:
            out[good] = bads
    return out


@lru_cache(maxsize=4)
def load_typo_map(path: str = DEFAULT_MAP) -> dict[str, list[str]]:
    """Load a cached typo map from JSON."""
    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Typo map not found at {p}. Build it with: "
            "python -m harness.perturb.natural_typo"
        )
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the natural typo map from the GitHub Typo Corpus.")
    ap.add_argument("--corpus", default=None,
                    help="Path to the original github-typo-corpus JSONL(.gz). "
                         f"If omitted, the HF mirror {HF_MIRROR} is used.")
    ap.add_argument("--out", default=DEFAULT_MAP)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--vocab-min", type=int, default=3,
                    help="Drop misspellings seen this often as words in corrected text.")
    args = ap.parse_args()

    edits = _corpus_edits(args.corpus) if args.corpus else _mirror_edits()
    typo_map = build_typo_map(edits, min_count=args.min_count, vocab_min=args.vocab_min)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(typo_map, f)
    print(f"Wrote {len(typo_map)} words ({sum(map(len, typo_map.values()))} misspellings) to {out}")


if __name__ == "__main__":
    main()
