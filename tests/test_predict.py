from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.predict import normalize_sequence, read_single_fasta, resolve_input


class PredictInputTests(unittest.TestCase):
    def test_normalize_sequence_removes_whitespace_and_uppercases(self) -> None:
        self.assertEqual(normalize_sequence("acde\n fgh"), "ACDEFGH")

    def test_normalize_sequence_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            normalize_sequence(" \n ")

    def test_normalize_sequence_rejects_invalid_symbols(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid amino-acid symbols"):
            normalize_sequence("ACD*")

    def test_normalize_sequence_rejects_sequences_over_model_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 1022"):
            normalize_sequence("A" * 1023)

    def test_read_single_fasta(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            fasta = Path(temporary_directory) / "protein.fasta"
            fasta.write_text(">P123 description\nACDE\nFGH\n", encoding="utf-8")
            self.assertEqual(read_single_fasta(fasta), ("P123", "ACDEFGH"))

    def test_read_single_fasta_rejects_multiple_records(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            fasta = Path(temporary_directory) / "proteins.fasta"
            fasta.write_text(">one\nACD\n>two\nEFG\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                read_single_fasta(fasta)

    def test_resolve_direct_sequence_uses_provided_identifier(self) -> None:
        args = Namespace(sequence="acd", fasta=None, protein_id="example")
        self.assertEqual(resolve_input(args), ("example", "ACD"))


if __name__ == "__main__":
    unittest.main()
