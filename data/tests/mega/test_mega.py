#!/usr/bin/env python3
"""Unit tests for the unified mega-dataset builder and audit sampling."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

MEGA_DIR = Path(__file__).resolve().parent.parent.parent / "mega"
if str(MEGA_DIR) not in sys.path:
    sys.path.insert(0, str(MEGA_DIR))

from build_mega_dataset import AUDIT_COLUMNS, MEGA_COLUMNS, build_mega_and_audit


class TestMegaDatasetBuilder(unittest.TestCase):
    def test_build_mega_and_audit_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            typos_dir = base_dir / "typos" / "data"
            codeswitch_dir = base_dir / "codeswitching" / "data"
            mega_dir = base_dir / "mega" / "data"

            (typos_dir / "clean").mkdir(parents=True)
            (typos_dir / "perturbed").mkdir(parents=True)
            codeswitch_dir.mkdir(parents=True)
            mega_dir.mkdir(parents=True)

            # Clean baseline
            clean_csv = typos_dir / "clean" / "amazon_polarity_clean.csv"
            with clean_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "text", "label"])
                w.writeheader()
                for i in range(10):
                    w.writerow(
                        {
                            "id": i,
                            "text": f"This great book number {i} was wonderful!",
                            "label": i % 2,
                        }
                    )

            # Perturbed low split
            pert_csv = typos_dir / "perturbed" / "amazon_polarity_typos_low.csv"
            with pert_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "text", "label"])
                w.writeheader()
                for i in range(10):
                    w.writerow(
                        {
                            "id": i,
                            "text": f"This graet book number {i} was wonderful!",
                            "label": i % 2,
                        }
                    )

            # Code-switch metadata + standard split
            cs_std = codeswitch_dir / "covoswitch_multilingual.csv"
            cs_meta = codeswitch_dir / "covoswitch_multilingual_metadata.csv"
            with cs_meta.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "source", "lang_pair", "text", "clean_text", "label"])
                w.writeheader()
                for i in range(10):
                    w.writerow(
                        {
                            "id": i,
                            "source": "covoswitch",
                            "lang_pair": "ca_en",
                            "text": f"Add the column i posa la suma aquí {i}.",
                            "clean_text": f"Add the column and put the sum here {i}.",
                            "label": "",
                        }
                    )

            mega_out = mega_dir / "ling581_mega_dataset.csv"
            audit_out = mega_dir / "manual_label_check_50.csv"
            mega_rows, audit_rows = build_mega_and_audit(
                typos_dir=typos_dir,
                codeswitch_dir=codeswitch_dir,
                mega_output=mega_out,
                audit_output=audit_out,
                audit_size=12,
                seed=42,
            )

            self.assertEqual(len(mega_rows), 30)
            self.assertEqual(len(audit_rows), 12)
            self.assertTrue(mega_out.exists())
            self.assertTrue(audit_out.exists())

            with mega_out.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                self.assertEqual(r.fieldnames, MEGA_COLUMNS)

            with audit_out.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                self.assertEqual(r.fieldnames, AUDIT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
