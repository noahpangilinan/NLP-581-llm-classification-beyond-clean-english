#!/usr/bin/env python3
"""Unit tests for code-switching curation and parsing."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

CODESWITCH_DIR = Path(__file__).resolve().parent.parent.parent / "codeswitching"
if str(CODESWITCH_DIR) not in sys.path:
    sys.path.insert(0, str(CODESWITCH_DIR))

from curate_codeswitch_datasets import (
    detokenize_ud_sentence,
    extract_switchlingua_texts,
    is_valid_covoswitch_pair,
    parse_parsing_project_csv,
    parse_switchlingua_file,
    write_standard_and_meta,
)


class TestCodeSwitchCuration(unittest.TestCase):
    def test_covoswitch_pair_validation(self) -> None:
        self.assertTrue(
            is_valid_covoswitch_pair(
                csw="Add the column i posa la suma aquí.",
                en="Add the column and put the sum here.",
                l2="Afegeix la columna i posa la suma aquí.",
            )
        )
        self.assertFalse(
            is_valid_covoswitch_pair(
                csw="Add the column and put the sum here.",
                en="Add the column and put the sum here.",
                l2="Afegeix la columna i posa la suma aquí.",
            )
        )
        self.assertFalse(
            is_valid_covoswitch_pair(
                csw="Nichts ihm konnte ihn zurückhalten. except himself.",
                en="There was nothing to hold him back except himself.",
                l2="Nichts außer ihm selbst konnte ihn zurückhalten.",
            )
        )

    def test_detokenize_ud_sentence(self) -> None:
        raw = "ah esa canción It's_Been_a_Day towards the end ."
        self.assertEqual(
            detokenize_ud_sentence(raw),
            "ah esa canción It's Been a Day towards the end.",
        )

    def test_parse_parsing_project_filters_monolingual_and_cleans_text(self) -> None:
        raw_csv = "\n".join(
            [
                "sentence_id,filename,sentence,multiple_words,token_info,UD_depd,Special Case",
                '46,herring11,"él empezó a trabajar towards the end .",0,"[[1, \'él\', \'spa\'], [2, \'empezó\', \'spa\'], [3, \'a\', \'spa\'], [4, \'trabajar\', \'spa\'], [5, \'towards\', \'eng\'], [6, \'the\', \'eng\'], [7, \'end\', \'eng\'], [8, \'.\', \'0\']]","[]",0',
                '126,herring11,"sí mmhm claro que sí amigo mío .",0,"[[1, \'sí\', \'spa\'], [2, \'mmhm\', \'spa\'], [3, \'claro\', \'spa\'], [4, \'que\', \'spa\'], [5, \'sí\', \'spa\'], [6, \'amigo\', \'spa\'], [7, \'mío\', \'spa\'], [8, \'.\', \'0\']]","[]",0',
            ]
        )
        rows = parse_parsing_project_csv(
            raw_csv,
            lang_pair="es_en",
            primary_langs=("spa", "eng"),
            min_tokens_per_lang=2,
            min_words=6,
            max_words=45,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "él empezó a trabajar towards the end.")
        self.assertEqual(rows[0]["lang_pair"], "es_en")
        self.assertEqual(rows[0]["label"], "")

    def test_switchlingua_parsing_csv_and_json_list_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "Spanish_eng.csv"
            sample_list = json.dumps(
                [
                    "La película empezó muy bien, but the ending made no sense at all.",
                    "Me encantó la comida, especially the fresh dessert they served!",
                ],
                ensure_ascii=False,
            )
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["topic", "cs_type", "cs_function", "data_generation_result"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "topic": "entertainment",
                        "cs_type": "Intrasentential",
                        "cs_function": "Expressive",
                        "data_generation_result": sample_list,
                    }
                )

            parsed = parse_switchlingua_file(csv_path, lang_pair="es_en", max_per_pair=10)
            self.assertEqual(len(parsed), 2)
            self.assertIn("but the ending made no sense", parsed[0]["text"])
            self.assertEqual(parsed[0]["label"], "")
            self.assertEqual(extract_switchlingua_texts(None), [])


if __name__ == "__main__":
    unittest.main()
