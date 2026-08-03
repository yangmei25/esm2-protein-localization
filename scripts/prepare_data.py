#!/usr/bin/env python3
"""Prepare the official DeepLoc 1.0 FASTA for binary classification.

The script keeps only membrane (M) and soluble (S) records, excludes proteins
longer than ESM-2's residue limit, preserves the official held-out test
partition, creates a stratified validation subset from the remaining
development records, and writes a model-ready CSV.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZOU")
LABEL_MAP = {"S": 0, "M": 1}


@dataclass(frozen=True)
class FastaRecord:
    """One parsed DeepLoc FASTA record."""

    protein_id: str
    sequence: str
    original_label: str
    official_split: str


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Read a FASTA file and return ``(header, sequence)`` pairs."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_lines: list[str] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence_lines)))
                header = line[1:]
                sequence_lines = []
            else:
                if header is None:
                    raise ValueError(
                        f"Sequence found before first FASTA header at line {line_number}"
                    )
                sequence_lines.append(line.upper())

    if header is not None:
        records.append((header, "".join(sequence_lines)))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def parse_deeploc_header(header: str, sequence: str) -> FastaRecord:
    """Parse a DeepLoc 1.0 header and validate its sequence."""
    fields = header.split()
    if len(fields) < 2:
        raise ValueError(f"Unexpected DeepLoc header: {header!r}")

    protein_id = fields[0]
    location_and_label = fields[1]
    if "-" not in location_and_label:
        raise ValueError(f"Header has no binary label: {header!r}")
    original_label = location_and_label.rsplit("-", 1)[-1]

    if not sequence:
        raise ValueError(f"Empty sequence for {protein_id}")
    invalid = set(sequence) - VALID_AMINO_ACIDS
    if invalid:
        raise ValueError(
            f"Invalid amino-acid symbols for {protein_id}: {sorted(invalid)}"
        )

    official_split = "test" if "test" in fields[2:] else "development"
    return FastaRecord(protein_id, sequence, original_label, official_split)


def validate_duplicates(frame: pd.DataFrame) -> None:
    """Reject exact sequences carrying conflicting labels or crossing splits."""
    grouped = frame.groupby("sequence", sort=False)
    conflicting_labels = grouped["label"].nunique().gt(1)
    if conflicting_labels.any():
        raise ValueError(
            f"Found {int(conflicting_labels.sum())} duplicate sequence groups "
            "with conflicting labels"
        )

    crossing_splits = grouped["official_split"].nunique().gt(1)
    if crossing_splits.any():
        raise ValueError(
            f"Found {int(crossing_splits.sum())} exact sequence groups crossing "
            "the official development/test boundary"
        )


def prepare_data(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    validation_fraction: float,
    random_seed: int,
    max_residues: int,
) -> pd.DataFrame:
    """Run the complete DeepLoc preprocessing pipeline."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if max_residues < 2:
        raise ValueError("max_residues must be at least 2")

    raw_pairs = read_fasta(input_path)
    records = [parse_deeploc_header(header, sequence) for header, sequence in raw_pairs]
    raw_label_counts = Counter(record.original_label for record in records)

    binary_records = [record for record in records if record.original_label in LABEL_MAP]
    rows = [
        {
            "protein_id": record.protein_id,
            "sequence": record.sequence,
            "label": LABEL_MAP[record.original_label],
            "official_split": record.official_split,
            "original_length": len(record.sequence),
        }
        for record in binary_records
    ]
    frame = pd.DataFrame(rows)

    if frame["protein_id"].duplicated().any():
        duplicate_count = int(frame["protein_id"].duplicated().sum())
        raise ValueError(f"Found {duplicate_count} duplicate protein IDs")
    validate_duplicates(frame)

    long_sequence_mask = frame["original_length"].gt(max_residues)
    excluded_long_sequences = frame[long_sequence_mask].copy()
    frame = frame[~long_sequence_mask].copy()

    development = frame[frame["official_split"].eq("development")].copy()
    official_test = frame[frame["official_split"].eq("test")].copy()

    before_deduplication = len(development)
    development = development.drop_duplicates(subset="sequence", keep="first").copy()
    duplicates_removed = before_deduplication - len(development)

    train_indices, validation_indices = train_test_split(
        development.index,
        test_size=validation_fraction,
        stratify=development["label"],
        random_state=random_seed,
    )
    development.loc[train_indices, "split"] = "train"
    development.loc[validation_indices, "split"] = "validation"
    official_test["split"] = "test"

    processed = pd.concat([development, official_test], ignore_index=True)

    processed = processed[
        [
            "protein_id",
            "sequence",
            "label",
            "split",
            "original_length",
        ]
    ].sort_values(["split", "protein_id"], kind="stable", ignore_index=True)

    if processed["sequence"].str.len().max() > max_residues:
        raise AssertionError("A processed sequence exceeds max_residues")
    if processed["protein_id"].duplicated().any():
        raise AssertionError("Processed protein IDs are not unique")
    if set(processed["split"]) != {"train", "validation", "test"}:
        raise AssertionError("Expected train, validation, and test splits")

    split_label_counts = (
        processed.groupby(["split", "label"], observed=True)
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "soluble", 1: "membrane"})
    )
    report = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "raw_fasta_records": len(records),
        "raw_label_counts": dict(sorted(raw_label_counts.items())),
        "binary_records_before_deduplication": len(binary_records),
        "development_duplicate_sequences_removed": duplicates_removed,
        "processed_records": len(processed),
        "validation_fraction_of_development": validation_fraction,
        "random_seed": random_seed,
        "max_residues": max_residues,
        "long_sequences_excluded": len(excluded_long_sequences),
        "long_sequences_excluded_by_official_split": {
            key: int(value)
            for key, value in excluded_long_sequences["official_split"]
            .value_counts()
            .sort_index()
            .items()
        },
        "long_sequences_excluded_by_label": {
            ("soluble" if key == 0 else "membrane"): int(value)
            for key, value in excluded_long_sequences["label"]
            .value_counts()
            .sort_index()
            .items()
        },
        "split_label_counts": split_label_counts.to_dict(orient="index"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return processed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare DeepLoc 1.0 for membrane-versus-soluble classification."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/deeploc_data.fasta"),
        help="Path to the untouched official DeepLoc FASTA.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/deeploc_binary.csv"),
        help="Destination for the model-ready CSV.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/preparation_report.json"),
        help="Destination for preprocessing counts and settings.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Fraction of non-test development data assigned to validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-residues",
        type=int,
        default=1022,
        help="Exclude proteins longer than this limit; sequences are never truncated.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepared = prepare_data(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        validation_fraction=args.validation_fraction,
        random_seed=args.seed,
        max_residues=args.max_residues,
    )
    summary = (
        prepared.groupby(["split", "label"], observed=True)
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "soluble", 1: "membrane"})
    )
    print(summary)
    print(f"\nSaved {len(prepared):,} records to {args.output}")
    print(f"Saved preparation report to {args.report}")


if __name__ == "__main__":
    main()
