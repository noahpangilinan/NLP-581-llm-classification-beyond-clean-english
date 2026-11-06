"""Additional tests for typo_injector.py (rate resolution, modes, records)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent / "typos"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from typo_injector import SEVERITY_RATES, TypoInjector, levenshtein_distance, resolve_rate

D = {"great": ["graet"], "movie": ["movei"]}


class TestResolveRate(unittest.TestCase):
    def test_default_is_medium(self) -> None:
        self.assertEqual(resolve_rate(), SEVERITY_RATES["medium"])

    def test_severity_and_rate(self) -> None:
        self.assertEqual(resolve_rate(severity=" HIGH "), 0.30)
        self.assertEqual(resolve_rate(rate=0.2), 0.2)

    def test_errors(self) -> None:
        with self.assertRaises(ValueError):
            resolve_rate(rate=0.1, severity="low")
        with self.assertRaises(ValueError):
            resolve_rate(severity="extreme")
        with self.assertRaises(ValueError):
            resolve_rate(rate=1.5)
        with self.assertRaises(ValueError):
            resolve_rate(rate=-0.1)


class TestLevenshtein(unittest.TestCase):
    def test_values(self) -> None:
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("", "abc"), 3)
        self.assertEqual(levenshtein_distance("same", "same"), 0)


class TestInjector(unittest.TestCase):
    def test_invalid_selection_mode(self) -> None:
        with self.assertRaises(ValueError):
            TypoInjector(typo_dict=D, selection_mode="bogus")

    def test_empty_and_zero_rate(self) -> None:
        self.assertEqual(TypoInjector(typo_dict=D, rate=1.0).perturb_text(""), ("", []))
        self.assertEqual(TypoInjector(typo_dict=D, rate=0.0).perturb_text("great"), ("great", []))

    def test_bernoulli_rate_one_changes_all_eligible(self) -> None:
        inj = TypoInjector(typo_dict=D, rate=1.0, selection_mode="bernoulli", seed=3)
        out, recs = inj.perturb_text("Great movie, yes!", row_index=7, label=1)
        self.assertEqual(out, "Graet movei, yes!")
        self.assertEqual([r.row_index for r in recs], [7, 7])
        self.assertEqual(recs[0].label, 1)

    def test_bernoulli_rate_zero_one_edges(self) -> None:
        inj = TypoInjector(typo_dict=D, rate=1.0, selection_mode="bernoulli", seed=0)
        self.assertEqual(inj.perturb_text("hello world")[1], [])


if __name__ == "__main__":
    unittest.main()
