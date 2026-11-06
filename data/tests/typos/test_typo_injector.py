#!/usr/bin/env python3
"""Unit and integration tests for the crate-ci/typos injection pipeline."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent / "typos"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from fetch_typo_corpus import parse_words_csv
from typo_injector import (
    SWITCH_LOG_COLUMNS,
    TypoInjector,
    classify_edit,
    levenshtein_distance,
    transfer_casing,
)

SAMPLE_DICT: dict[str, list[str]] = {
    "the": ["hte", "teh"],
    "great": ["graet", "gerat"],
    "movie": ["movei"],
    "friend": ["freind", "frind"],
    "beautiful": ["beatiful"],
    "terrible": ["terrable"],
    "good": ["ggod", "godd"],
}


class TestTypoCore(unittest.TestCase):
    def test_parse_words_csv_inverts_and_filters_single_char(self) -> None:
        raw = "\n".join(
            [
                "aaccess,access",
                "aache,ache,cache",
                "xwindows,x",
                "graet,great",
                "gerat,great",
            ]
        )
        typo_dict, known_typos = parse_words_csv(raw)
        self.assertEqual(typo_dict["access"], ["aaccess"])
        self.assertEqual(typo_dict["ache"], ["aache"])
        self.assertEqual(typo_dict["cache"], ["aache"])
        self.assertEqual(typo_dict["great"], ["gerat", "graet"])
        self.assertNotIn("x", typo_dict)
        self.assertIn("graet", known_typos)

    def test_transfer_casing(self) -> None:
        self.assertEqual(transfer_casing("friend", "freind"), "freind")
        self.assertEqual(transfer_casing("Friend", "freind"), "Freind")
        self.assertEqual(transfer_casing("FRIEND", "freind"), "FREIND")

    def test_edit_classification_and_distance(self) -> None:
        self.assertEqual(classify_edit("great", "graet"), ("transposition", 2))
        self.assertEqual(classify_edit("access", "aaccess"), ("insertion", 1))
        self.assertEqual(classify_edit("light", "ligh"), ("deletion", 1))
        self.assertEqual(classify_edit("terrible", "terrable"), ("substitution", 1))
        self.assertEqual(classify_edit("abandoned", "abadonded"), ("multi_edit", 2))
        self.assertEqual(levenshtein_distance("same", "same"), 0)

    def test_no_fallback_for_non_dictionary_words(self) -> None:
        injector = TypoInjector(typo_dict=SAMPLE_DICT, rate=1.0, seed=7)
        clean = "Unlisted vocabulary words like cinema or popcorn stay untouched!"
        perturbed, records = injector.perturb_text(clean, row_index=3, label=1)
        self.assertEqual(perturbed, clean)
        self.assertEqual(records, [])

    def test_exact_indices_character_spans_punctuation_and_casing(self) -> None:
        injector = TypoInjector(typo_dict=SAMPLE_DICT, rate=1.0, seed=42)
        clean = '  The "GREAT" (movie), my Friend -- didn\'t fail!  '
        perturbed, records = injector.perturb_text(clean, row_index=5, label=1)

        # Eligible words are: The (0), "GREAT" (1), (movie), (2), Friend (4)
        # Ineligible: my (3), -- (5), didn't (6), fail! (7)
        self.assertEqual([r.word_index for r in records], [0, 1, 2, 4])

        clean_tokens = clean.split()
        pert_tokens = perturbed.split()
        self.assertEqual(len(clean_tokens), len(pert_tokens))

        # Leading and trailing whitespace preserved
        self.assertTrue(perturbed.startswith("  "))
        self.assertTrue(perturbed.endswith("  "))
        # Contraction untouched
        self.assertEqual(pert_tokens[6], "didn't")

        for rec in records:
            self.assertEqual(rec.row_index, 5)
            self.assertEqual(rec.label, 1)
            self.assertEqual(
                clean[rec.orig_char_start : rec.orig_char_end], rec.original_word
            )
            self.assertEqual(
                perturbed[rec.pert_char_start : rec.pert_char_end], rec.perturbed_word
            )
            self.assertEqual(clean_tokens[rec.word_index], rec.original_token)
            self.assertEqual(pert_tokens[rec.word_index], rec.perturbed_token)
            self.assertIn(
                rec.perturbed_word.lower(),
                SAMPLE_DICT[rec.original_word.lower()],
            )

        # Verify casing on GREAT -> GRAET/GERAT and Friend -> Freind/Frind
        self.assertTrue(records[0].perturbed_word.istitle())
        self.assertTrue(records[1].perturbed_word.isupper())
        self.assertTrue(records[2].perturbed_word.islower())
        self.assertTrue(records[3].perturbed_word.istitle())

    def test_seed_reproducibility(self) -> None:
        text = "The great movie and good friend made the terrible day beautiful."
        inj_a = TypoInjector(typo_dict=SAMPLE_DICT, rate=0.4, seed=123)
        inj_b = TypoInjector(typo_dict=SAMPLE_DICT, rate=0.4, seed=123)
        pert_a, rec_a = inj_a.perturb_text(text, row_index=0)
        pert_b, rec_b = inj_b.perturb_text(text, row_index=0)
        self.assertEqual(pert_a, pert_b)
        self.assertEqual(rec_a, rec_b)


class TestInjectTyposCLI(unittest.TestCase):
    def test_cli_generates_clean_schema_dataset_and_switch_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            corpus_json = tmp_dir / "mini_dict.json"
            corpus_json.write_text(json.dumps(SAMPLE_DICT), encoding="utf-8")

            clean_csv = tmp_dir / "clean.csv"
            input_rows = [
                {
                    "id": "0",
                    "text": "The great movie was beautiful and good!",
                    "label": "1",
                },
                {
                    "id": "1",
                    "text": "The terrible movie disappointed my friend.",
                    "label": "0",
                },
            ]
            with clean_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "text", "label"])
                writer.writeheader()
                writer.writerows(input_rows)

            out_template = tmp_dir / "perturbed_{severity}.csv"
            cmd = [
                sys.executable,
                str(PROJECT_DIR / "inject_typos.py"),
                "--input",
                str(clean_csv),
                "--output",
                str(out_template),
                "--severities",
                "low,medium,high",
                "--seed",
                "99",
                "--corpus-path",
                str(corpus_json),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertIn("[high | target=30.0%]", res.stdout)

            counts = {}
            for sev in ("low", "medium", "high"):
                pert_file = tmp_dir / f"perturbed_{sev}.csv"
                sw_file = tmp_dir / f"perturbed_{sev}_switches.csv"
                self.assertTrue(pert_file.exists())
                self.assertTrue(sw_file.exists())

                with pert_file.open("r", encoding="utf-8", newline="") as f:
                    p_reader = csv.DictReader(f)
                    # Schema must match clean dataset exactly (no extra columns!)
                    self.assertEqual(p_reader.fieldnames, ["id", "text", "label"])
                    p_rows = list(p_reader)
                    self.assertEqual(len(p_rows), 2)
                    self.assertEqual([r["label"] for r in p_rows], ["1", "0"])

                with sw_file.open("r", encoding="utf-8", newline="") as f:
                    s_reader = csv.DictReader(f)
                    self.assertEqual(s_reader.fieldnames, SWITCH_LOG_COLUMNS)
                    s_rows = list(s_reader)
                    counts[sev] = len(s_rows)

                    # Every recorded switch must align with perturbed_file
                    for s_row in s_rows:
                        r_idx = int(s_row["row_index"])
                        w_idx = int(s_row["word_index"])
                        orig_text = input_rows[r_idx]["text"]
                        pert_text = p_rows[r_idx]["text"]
                        self.assertEqual(
                            orig_text.split()[w_idx], s_row["original_token"]
                        )
                        self.assertEqual(
                            pert_text.split()[w_idx], s_row["perturbed_token"]
                        )

            self.assertLessEqual(counts["low"], counts["medium"])
            self.assertLessEqual(counts["medium"], counts["high"])
            self.assertGreater(counts["high"], 0)


if __name__ == "__main__":
    unittest.main()
