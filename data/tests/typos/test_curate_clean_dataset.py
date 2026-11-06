"""Tests for curate_clean_dataset.py (no network; `datasets` is stubbed)."""

from __future__ import annotations

import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent / "typos"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    import datasets  # noqa: F401
except ImportError:  # keep tests runnable without the HF dependency
    stub = types.ModuleType("datasets")
    stub.load_dataset = lambda *a, **k: iter(())  # type: ignore[attr-defined]
    sys.modules["datasets"] = stub

import curate_clean_dataset as ccd

ENGLISH = {
    w
    for w in (
        "this product is really great and i love how it works every single day "
        "the movie was good bad item"
    ).split()
}
ENGLISH |= {"i", "product", "works", "love"}
TYPOS = {"teh", "recieve"}
GOOD = "This product is really great and I love how it works every single day"


class TestIsClean(unittest.TestCase):
    def check(self, text: str, **kw) -> bool:
        return ccd.is_clean_typo_free_review(text, ENGLISH, TYPOS, **kw)

    def test_accepts_clean(self) -> None:
        self.assertTrue(self.check(GOOD))

    def test_rejects_empty_nonascii_and_whitespace_chars(self) -> None:
        self.assertFalse(self.check(""))
        self.assertFalse(self.check(GOOD + " caf\u00e9"))
        self.assertFalse(self.check(GOOD + "\nmore"))
        self.assertFalse(self.check(GOOD + "\tmore"))

    def test_rejects_bad_spacing_punctuation(self) -> None:
        self.assertFalse(self.check(GOOD + ",great"))
        self.assertFalse(self.check(GOOD + " !"))
        self.assertFalse(self.check(GOOD + "!!!"))
        self.assertFalse(self.check(GOOD.replace("is", "is  ")))

    def test_word_count_bounds(self) -> None:
        self.assertFalse(self.check("good good"))
        self.assertFalse(self.check(GOOD, max_words=5))
        self.assertTrue(self.check(GOOD, min_words=5, max_words=50))

    def test_rejects_known_typo_and_unknown_word(self) -> None:
        self.assertFalse(self.check(GOOD.replace("great", "teh")))
        self.assertFalse(self.check(GOOD.replace("great", "zzyzx")))

    def test_contractions_and_possessives(self) -> None:
        self.assertTrue(self.check(GOOD.replace("I love", "I don't love")))
        self.assertTrue(self.check(GOOD.replace("product", "product's")))
        self.assertFalse(self.check(GOOD.replace("product", "zzz's")))
        self.assertFalse(self.check(GOOD.replace("product", "prod'xy")))


class TestLoadWordSets(unittest.TestCase):
    def test_loads_lowercased_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            e, k = Path(tmp) / "e.txt", Path(tmp) / "k.txt"
            e.write_text("Hello\n\nWorld\n")
            k.write_text("teh\n")
            with mock.patch.object(ccd, "ensure_typo_corpus"), mock.patch.object(
                ccd, "DEFAULT_ENGLISH_PATH", e
            ), mock.patch.object(ccd, "DEFAULT_KNOWN_TYPOS_PATH", k):
                eng, typ = ccd.load_word_sets()
        self.assertEqual(eng, {"hello", "world"})
        self.assertEqual(typ, {"teh"})


class TestCurate(unittest.TestCase):
    def test_odd_num_samples_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ccd.curate_amazon_polarity(num_samples=3)

    def test_balanced_interleaved_output(self) -> None:
        stream = [
            {"label": 1, "content": GOOD},
            {"label": 1, "content": GOOD + " again"},
            {"label": 0, "content": "bad zzyzx"},  # rejected
            {"label": 0, "content": GOOD},
            {"label": 0, "content": GOOD + " today"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "o" / "clean.csv"
            with mock.patch.object(
                ccd, "load_word_sets", return_value=(ENGLISH | {"again", "today"}, TYPOS)
            ), mock.patch.object(ccd, "load_dataset", return_value=iter(stream)):
                ccd.curate_amazon_polarity(output_path=out, num_samples=2)
            rows = list(csv.DictReader(out.open(encoding="utf-8")))
        self.assertEqual([r["id"] for r in rows], ["0", "1"])
        self.assertEqual([r["label"] for r in rows], ["0", "1"])
        self.assertEqual(rows[0]["text"], GOOD)
        self.assertEqual(rows[1]["text"], GOOD)


if __name__ == "__main__":
    unittest.main()
