from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.predict import (
    aggregate_window_probabilities,
    make_sequence_windows,
    normalize_sequence,
    read_single_fasta,
    resolve_input,
)


class PredictInputTests(unittest.TestCase):
    def test_normalize_sequence_removes_whitespace_and_uppercases(self) -> None:
        self.assertEqual(normalize_sequence("acde\n fgh"), "ACDEFGH")

    def test_normalize_sequence_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            normalize_sequence(" \n ")

    def test_normalize_sequence_rejects_invalid_symbols(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid amino-acid symbols"):
            normalize_sequence("ACD*")

    def test_normalize_sequence_accepts_sequences_over_single_window_limit(self) -> None:
        self.assertEqual(len(normalize_sequence("A" * 2000)), 2000)

    def test_short_sequence_creates_one_window(self) -> None:
        self.assertEqual(
            make_sequence_windows("ACDE", window_size=10, stride=5),
            [{"start": 1, "end": 4, "sequence": "ACDE"}],
        )

    def test_long_sequence_windows_cover_both_ends(self) -> None:
        windows = make_sequence_windows("A" * 2000, window_size=1022, stride=511)
        self.assertEqual([(w["start"], w["end"]) for w in windows], [(1, 1022), (512, 1533), (979, 2000)])

    def test_window_configuration_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "stride"):
            make_sequence_windows("ACDE", window_size=10, stride=11)

    def test_window_probabilities_use_maximum_aggregation(self) -> None:
        self.assertEqual(aggregate_window_probabilities([0.2, 0.8, 0.4]), 0.8)

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
