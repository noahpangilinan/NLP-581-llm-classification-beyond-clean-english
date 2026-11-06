#!/usr/bin/env python3
"""Core typo injection engine backed exclusively by crate-ci/typos.

Injects genuine human typos from crate-ci/typos randomly into clean English text
while preserving whitespace, punctuation, and capitalization, and emitting a
structured word-level switch log for downstream robustness analysis.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fetch_typo_corpus import DEFAULT_DICT_PATH, ensure_typo_corpus

SEVERITY_RATES: dict[str, float] = {
    "low": 0.05,
    "medium": 0.15,
    "high": 0.30,
}

# Matches a whitespace token consisting of optional leading punctuation,
# a purely alphabetic core word, and optional trailing punctuation.
# Contractions (didn't), hyphenated compounds (well-made), and tokens containing
# digits are intentionally excluded so only clean single words are matched.
TOKEN_WRAP_RE = re.compile(r"^([^A-Za-z0-9]*)([A-Za-z]+)([^A-Za-z0-9]*)$")
WHITESPACE_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class TypoSwitchRecord:
    """Metadata record for a single word switched with a crate-ci/typos typo."""

    row_index: int
    word_index: int
    original_word: str
    perturbed_word: str
    original_token: str
    perturbed_token: str
    orig_char_start: int
    orig_char_end: int
    pert_char_start: int
    pert_char_end: int
    edit_type: str
    edit_distance: int
    num_typo_candidates: int
    sentence_word_count: int
    eligible_word_count: int
    label: Any
    target_rate: float
    seed: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SWITCH_LOG_COLUMNS: list[str] = [
    "row_index",
    "word_index",
    "original_word",
    "perturbed_word",
    "original_token",
    "perturbed_token",
    "orig_char_start",
    "orig_char_end",
    "pert_char_start",
    "pert_char_end",
    "edit_type",
    "edit_distance",
    "num_typo_candidates",
    "sentence_word_count",
    "eligible_word_count",
    "label",
    "target_rate",
    "seed",
]


def transfer_casing(clean_word: str, typo_word: str) -> str:
    """Transfer lowercase, TitleCase, or UPPERCASE from clean_word to typo_word."""
    if clean_word.isupper():
        return typo_word.upper()
    if len(clean_word) >= 1 and clean_word[0].isupper() and clean_word[1:].islower():
        return typo_word.capitalize()
    return typo_word.lower()


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard character-level Levenshtein edit distance."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def classify_edit(original_word: str, perturbed_word: str) -> tuple[str, int]:
    """Classify the character-level typo operation and compute edit distance."""
    orig = original_word.lower()
    pert = perturbed_word.lower()
    dist = levenshtein_distance(orig, pert)

    # Check single adjacent character transposition (e.g., 'great' -> 'graet')
    if len(orig) == len(pert) and orig != pert:
        diffs = [i for i in range(len(orig)) if orig[i] != pert[i]]
        if (
            len(diffs) == 2
            and diffs[1] == diffs[0] + 1
            and orig[diffs[0]] == pert[diffs[1]]
            and orig[diffs[1]] == pert[diffs[0]]
        ):
            return "transposition", dist

    if dist == 1:
        if len(pert) == len(orig) + 1:
            return "insertion", dist
        if len(pert) == len(orig) - 1:
            return "deletion", dist
        if len(pert) == len(orig):
            return "substitution", dist

    return "multi_edit", dist


def resolve_rate(rate: float | None = None, severity: str | None = None) -> float:
    """Resolve perturbation rate from explicit float or named severity preset."""
    if rate is not None and severity is not None:
        raise ValueError("Specify either `rate` or `severity`, not both.")
    if severity is not None:
        key = severity.strip().lower()
        if key not in SEVERITY_RATES:
            valid = ", ".join(f"{k} ({v:.0%})" for k, v in SEVERITY_RATES.items())
            raise ValueError(f"Unknown severity '{severity}'. Valid options: {valid}")
        return SEVERITY_RATES[key]
    if rate is None:
        return SEVERITY_RATES["medium"]
    if not (0.0 <= rate <= 1.0):
        raise ValueError(f"Perturbation rate must be in [0.0, 1.0], got {rate}.")
    return float(rate)


class TypoInjector:
    """Randomly injects authentic typos from crate-ci/typos into clean text."""

    def __init__(
        self,
        typo_dict: dict[str, list[str]] | None = None,
        corpus_path: Path = DEFAULT_DICT_PATH,
        rate: float | None = None,
        severity: str | None = None,
        seed: int | None = 42,
        selection_mode: str = "proportional",
    ) -> None:
        self.rate = resolve_rate(rate=rate, severity=severity)
        self.seed = seed
        self.rng = random.Random(seed)
        if selection_mode not in ("proportional", "bernoulli"):
            raise ValueError(
                f"selection_mode must be 'proportional' or 'bernoulli', got '{selection_mode}'"
            )
        self.selection_mode = selection_mode
        self.typo_dict = (
            typo_dict
            if typo_dict is not None
            else ensure_typo_corpus(dict_path=corpus_path)
        )

    def perturb_text(
        self,
        text: str,
        row_index: int = 0,
        label: Any = None,
    ) -> tuple[str, list[TypoSwitchRecord]]:
        """Inject random crate-ci/typos into a single text string.

        Words not present in crate-ci/typos are never modified (no synthetic or
        keyboard-neighbor fallback is ever used).
        """
        if not text:
            return text, []

        matches = list(WHITESPACE_TOKEN_RE.finditer(text))
        sentence_word_count = len(matches)
        if sentence_word_count == 0 or self.rate <= 0.0:
            return text, []

        # Identify which whitespace tokens map to a clean word in crate-ci/typos
        parsed_tokens: dict[int, tuple[str, str, str]] = {}
        eligible_indices: list[int] = []
        for idx, m in enumerate(matches):
            wrap = TOKEN_WRAP_RE.match(m.group(0))
            if not wrap:
                continue
            prefix, core_word, suffix = wrap.groups()
            if core_word.lower() in self.typo_dict:
                parsed_tokens[idx] = (prefix, core_word, suffix)
                eligible_indices.append(idx)

        eligible_word_count = len(eligible_indices)
        if eligible_word_count == 0:
            return text, []

        # Select which eligible word indices in this sentence will be switched
        if self.selection_mode == "proportional":
            expected = sentence_word_count * self.rate
            base = int(expected)
            frac = expected - base
            target_switches = base + (1 if self.rng.random() < frac else 0)
            num_switches = min(target_switches, eligible_word_count)
            chosen_indices = set(self.rng.sample(eligible_indices, num_switches))
        else:
            chosen_indices = {
                idx for idx in eligible_indices if self.rng.random() < self.rate
            }

        if not chosen_indices:
            return text, []

        out_parts: list[str] = []
        records: list[TypoSwitchRecord] = []
        last_orig_end = 0
        pert_cursor = 0

        for idx, m in enumerate(matches):
            gap = text[last_orig_end : m.start()]
            out_parts.append(gap)
            pert_cursor += len(gap)

            raw_token = m.group(0)
            if idx in chosen_indices:
                prefix, core_word, suffix = parsed_tokens[idx]
                candidates = self.typo_dict[core_word.lower()]
                chosen_raw_typo = self.rng.choice(candidates)
                perturbed_word = transfer_casing(core_word, chosen_raw_typo)
                perturbed_token = f"{prefix}{perturbed_word}{suffix}"

                orig_char_start = m.start() + len(prefix)
                orig_char_end = orig_char_start + len(core_word)
                pert_char_start = pert_cursor + len(prefix)
                pert_char_end = pert_char_start + len(perturbed_word)

                edit_type, edit_dist = classify_edit(core_word, perturbed_word)

                records.append(
                    TypoSwitchRecord(
                        row_index=row_index,
                        word_index=idx,
                        original_word=core_word,
                        perturbed_word=perturbed_word,
                        original_token=raw_token,
                        perturbed_token=perturbed_token,
                        orig_char_start=orig_char_start,
                        orig_char_end=orig_char_end,
                        pert_char_start=pert_char_start,
                        pert_char_end=pert_char_end,
                        edit_type=edit_type,
                        edit_distance=edit_dist,
                        num_typo_candidates=len(candidates),
                        sentence_word_count=sentence_word_count,
                        eligible_word_count=eligible_word_count,
                        label="" if label is None else label,
                        target_rate=self.rate,
                        seed=self.seed,
                    )
                )

                out_parts.append(perturbed_token)
                pert_cursor += len(perturbed_token)
            else:
                out_parts.append(raw_token)
                pert_cursor += len(raw_token)

            last_orig_end = m.end()

        out_parts.append(text[last_orig_end:])
        return "".join(out_parts), records
