from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.prepare_data import (
    parse_deeploc_header,
    read_fasta,
    validate_duplicates,
)


class PrepareDataTests(unittest.TestCase):
    def test_read_fasta_combines_multiline_sequence(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            fasta = Path(temporary_directory) / "input.fasta"
            fasta.write_text(">P1 Cytoplasm-S\nACD\nEFG\n", encoding="utf-8")
            self.assertEqual(read_fasta(fasta), [("P1 Cytoplasm-S", "ACDEFG")])

    def test_parse_deeploc_header_extracts_label_and_test_split(self) -> None:
        record = parse_deeploc_header("P1 Membrane-M test", "ACDE")
        self.assertEqual(record.protein_id, "P1")
        self.assertEqual(record.original_label, "M")
        self.assertEqual(record.official_split, "test")

    def test_parse_deeploc_header_rejects_invalid_residue(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid amino-acid symbols"):
            parse_deeploc_header("P1 Cytoplasm-S", "ACD*")

    def test_duplicate_sequences_cannot_have_conflicting_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "sequence": ["ACD", "ACD"],
                "label": [0, 1],
                "official_split": ["development", "development"],
            }
        )
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            validate_duplicates(frame)

    def test_duplicate_sequences_cannot_cross_official_splits(self) -> None:
        frame = pd.DataFrame(
            {
                "sequence": ["ACD", "ACD"],
                "label": [0, 0],
                "official_split": ["development", "test"],
            }
        )
        with self.assertRaisesRegex(ValueError, "crossing"):
            validate_duplicates(frame)


if __name__ == "__main__":
    unittest.main()
