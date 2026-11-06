"""Tests for the helper functions in inject_typos.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent / "typos"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inject_typos as it
from typo_injector import SWITCH_LOG_COLUMNS, TypoInjector

TYPO_DICT = {"great": ["graet"], "movie": ["movei"], "good": ["ggod"]}
COLS = ["id", "text", "label"]


class TestDetectColumn(unittest.TestCase):
    def test_explicit(self) -> None:
        self.assertEqual(it.detect_column(COLS, "text", it.CANDIDATE_TEXT_COLS, True, "text"), "text")

    def test_explicit_missing_raises(self) -> None:
        with self.assertRaises(ValueError):
            it.detect_column(COLS, "nope", it.CANDIDATE_TEXT_COLS, True, "text")

    def test_auto_detect_case_insensitive(self) -> None:
        self.assertEqual(
            it.detect_column(["ID", "Review"], None, it.CANDIDATE_TEXT_COLS, True, "text"),
            "Review",
        )

    def test_required_vs_optional_missing(self) -> None:
        with self.assertRaises(ValueError):
            it.detect_column(["a"], None, it.CANDIDATE_TEXT_COLS, True, "text")
        self.assertIsNone(it.detect_column(["a"], None, it.CANDIDATE_LABEL_COLS, False, "label"))


class TestLoadSave(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.rows = [
            {"id": "0", "text": "great movie", "label": "1"},
            {"id": "1", "text": "good", "label": "0"},
        ]

    def test_roundtrip_csv_tsv_jsonl_json(self) -> None:
        for ext in (".csv", ".tsv", ".jsonl", ".json"):
            p = self.dir / f"d{ext}"
            it.save_dataset_rows(p, ext, self.rows, COLS, "text")
            fmt, rows, fields = it.load_dataset_rows(p)
            self.assertEqual(fmt, ext)
            self.assertEqual(rows, self.rows, ext)
            self.assertEqual(fields, COLS, ext)

    def test_txt_roundtrip(self) -> None:
        p = self.dir / "d.txt"
        it.save_dataset_rows(p, ".txt", self.rows, COLS, "text")
        self.assertEqual(p.read_text(), "great movie\ngood\n")
        _, rows, fields = it.load_dataset_rows(p)
        self.assertEqual(rows, [{"text": "great movie"}, {"text": "good"}])
        self.assertEqual(fields, ["text"])

    def test_unsupported_extensions(self) -> None:
        p = self.dir / "d.xml"
        p.write_text("x")
        with self.assertRaises(ValueError):
            it.load_dataset_rows(p)
        with self.assertRaises(ValueError):
            it.save_dataset_rows(p, ".xml", self.rows, COLS, "text")

    def test_json_must_be_list(self) -> None:
        p = self.dir / "d.json"
        p.write_text(json.dumps({"a": 1}))
        with self.assertRaises(ValueError):
            it.load_dataset_rows(p)

    def test_empty_csv_raises(self) -> None:
        p = self.dir / "e.csv"
        p.write_text("")
        with self.assertRaises(ValueError):
            it.load_dataset_rows(p)


class TestPaths(unittest.TestCase):
    def test_default_switches_path(self) -> None:
        self.assertEqual(it.default_switches_path(Path("a/out.csv")), Path("a/out_switches.csv"))
        self.assertEqual(it.default_switches_path(Path("a/out.jsonl")), Path("a/out_switches.jsonl"))
        self.assertEqual(it.default_switches_path(Path("a/out.json")), Path("a/out_switches.csv"))

    def test_format_run_path(self) -> None:
        self.assertEqual(
            it.format_run_path(Path("o_{severity}.csv"), "low", 0.05, True), Path("o_low.csv")
        )
        self.assertEqual(
            it.format_run_path(Path("o_{rate}.csv"), "x", 0.30, True), Path("o_0.3.csv")
        )
        self.assertEqual(it.format_run_path(Path("o.csv"), "low", 0.05, True), Path("o_low.csv"))
        self.assertEqual(it.format_run_path(Path("o.csv"), "low", 0.05, False), Path("o.csv"))


class TestRunSingleInjection(unittest.TestCase):
    def test_full_rate_writes_outputs_and_summary(self) -> None:
        rows = [
            {"id": "0", "text": "great movie", "label": "1"},
            {"id": "1", "text": "unknown words", "label": "0"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out, sw = Path(tmp) / "o.csv", Path(tmp) / "o_switches.jsonl"
            summary = it.run_single_injection(
                rows=rows, fieldnames=COLS, fmt=".csv", text_col="text", label_col="label",
                typo_dict=TYPO_DICT, rate=1.0, tag="t", seed=1,
                selection_mode="proportional", output_path=out, switches_path=sw,
            )
            self.assertIn("graet movei", out.read_text())
            recs = [json.loads(l) for l in sw.read_text().splitlines()]
        self.assertEqual(len(recs), 2)
        self.assertEqual({r["original_word"] for r in recs}, {"great", "movie"})
        self.assertEqual(recs[0]["label"], "1")
        self.assertEqual(summary["rows_total"], 2)
        self.assertEqual(summary["rows_perturbed"], 1)
        self.assertEqual(summary["words_total"], 4)
        self.assertEqual(summary["words_switched"], 2)
        self.assertAlmostEqual(summary["empirical_word_rate"], 0.5)
        self.assertEqual(rows[0]["text"], "great movie")  # input not mutated

    def test_csv_switch_log_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sw = Path(tmp) / "s.csv"
            it.save_switch_log(sw, [])
            self.assertEqual(sw.read_text().strip(), ",".join(SWITCH_LOG_COLUMNS))

    def test_zero_rate_no_changes(self) -> None:
        rows = [{"text": "great movie"}]
        with tempfile.TemporaryDirectory() as tmp:
            s = it.run_single_injection(
                rows=rows, fieldnames=["text"], fmt=".csv", text_col="text", label_col=None,
                typo_dict=TYPO_DICT, rate=0.0, tag="z", seed=1, selection_mode="bernoulli",
                output_path=Path(tmp) / "o.csv", switches_path=Path(tmp) / "s.csv",
            )
        self.assertEqual(s["words_switched"], 0)
        self.assertEqual(s["empirical_word_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
