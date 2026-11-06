"""Tests for fetch_typo_corpus.py (no network access)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent / "typos"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import fetch_typo_corpus as ftc


class TestParseWordsCsv(unittest.TestCase):
    def test_inverts_mapping_and_sorts(self) -> None:
        d, known = ftc.parse_words_csv("teh,the\nhte,the\nfreind,friend\n")
        self.assertEqual(d, {"friend": ["freind"], "the": ["hte", "teh"]})
        self.assertEqual(known, {"teh", "hte", "freind"})

    def test_multiple_corrections(self) -> None:
        d, _ = ftc.parse_words_csv("abd,and,bad\n")
        self.assertEqual(d, {"and": ["abd"], "bad": ["abd"]})

    def test_skips_blank_short_nonalpha_and_identical(self) -> None:
        raw = "\n  \nonlyone\nfoo1,foo\nxx,x\nsame,same\nok,okay\n"
        d, known = ftc.parse_words_csv(raw)
        self.assertEqual(d, {"okay": ["ok"]})
        self.assertNotIn("foo1", known)
        self.assertNotIn("onlyone", known)

    def test_lowercases(self) -> None:
        d, known = ftc.parse_words_csv("TEH,The\n")
        self.assertEqual(d, {"the": ["teh"]})
        self.assertIn("teh", known)


class TestEnsureTypoCorpus(unittest.TestCase):
    def test_uses_cache_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.json"
            p.write_text(json.dumps({"the": ["teh"]}))
            with mock.patch.object(ftc, "download_text") as dl:
                out = ftc.ensure_typo_corpus(
                    dict_path=p,
                    english_path=Path(tmp) / "e.txt",
                    known_typos_path=Path(tmp) / "k.txt",
                )
            dl.assert_not_called()
            self.assertEqual(out, {"the": ["teh"]})

    def test_downloads_and_writes_files(self) -> None:
        def fake(url: str, timeout: int = 30) -> str:
            return "teh,the\n" if url == ftc.WORDS_CSV_URL else "Hello\nworld\nbad1\n"

        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            with mock.patch.object(ftc, "download_text", side_effect=fake):
                out = ftc.ensure_typo_corpus(
                    dict_path=t / "sub" / "d.json",
                    english_path=t / "sub" / "e.txt",
                    known_typos_path=t / "sub" / "k.txt",
                )
            self.assertEqual(out, {"the": ["teh"]})
            self.assertEqual(json.loads((t / "sub" / "d.json").read_text()), out)
            self.assertEqual((t / "sub" / "k.txt").read_text(), "teh\n")
            self.assertEqual((t / "sub" / "e.txt").read_text(), "hello\nworld\n")

    def test_force_download_overrides_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "d.json").write_text(json.dumps({"old": ["x"]}))
            with mock.patch.object(ftc, "download_text", return_value="teh,the\n"):
                out = ftc.ensure_typo_corpus(
                    dict_path=t / "d.json",
                    english_path=t / "e.txt",
                    known_typos_path=t / "k.txt",
                    force_download=True,
                )
            self.assertEqual(out, {"the": ["teh"]})


if __name__ == "__main__":
    unittest.main()
