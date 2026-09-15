#!/usr/bin/env python3
"""Prepare the DeepLoc 2.1 held-out membrane-type set for external evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd


SOURCE_URL = (
    "https://services.healthtech.dtu.dk/services/DeepLoc-2.1/data/"
    "Swissprot_Membrane_Train_Validation_dataset.csv"
)
MEMBRANE_COLUMNS = ("Peripheral", "Transmembrane", "LipidAnchor")
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZOU")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "esm2-localization/2"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def prepare(
    source: Path,
    reference: Path,
    output: Path,
    report_path: Path,
    partition: int,
    min_residues: int,
    max_residues: int,
) -> pd.DataFrame:
    raw = pd.read_csv(source)
    required = {
        "ACC", "Kingdom", "Partition", "Sequence", "Soluble", *MEMBRANE_COLUMNS
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"External dataset is missing columns: {sorted(missing)}")

    frame = raw[
        raw["Partition"].eq(partition) & raw["Kingdom"].eq("Eukaryota")
    ].copy()
    initial_partition_records = len(frame)
    frame["sequence"] = frame["Sequence"].astype(str).str.upper().str.strip()
    frame["has_membrane_label"] = frame[list(MEMBRANE_COLUMNS)].astype(bool).any(axis=1)
    frame["has_soluble_label"] = frame["Soluble"].astype(bool)
    frame["label"] = frame["has_membrane_label"].astype(int)
    frame["original_length"] = frame["sequence"].str.len()

    invalid_mask = frame["sequence"].map(lambda sequence: bool(set(sequence) - VALID_AMINO_ACIDS))
    duplicate_mask = frame.duplicated("sequence", keep="first")
    short_mask = frame["original_length"].lt(min_residues)
    long_mask = (
        frame["original_length"].gt(max_residues)
        if max_residues > 0 else pd.Series(False, index=frame.index)
    )
    reference_sequences = set(pd.read_csv(reference, usecols=["sequence"])["sequence"])
    exact_overlap_mask = frame["sequence"].isin(reference_sequences)
    keep = ~(invalid_mask | duplicate_mask | short_mask | long_mask | exact_overlap_mask)
    prepared = frame.loc[keep].copy()
    prepared = prepared.rename(columns={"ACC": "protein_id"})
    prepared["split"] = "external_test"
    prepared = prepared[
        [
            "protein_id", "sequence", "label", "split", "original_length",
            "Peripheral", "Transmembrane", "LipidAnchor", "Soluble",
        ]
    ].sort_values("protein_id", ignore_index=True)

    if prepared.empty or prepared["label"].nunique() != 2:
        raise ValueError("Prepared external set must contain both binary classes")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output, index=False)
    report = {
        "source_url": SOURCE_URL,
        "source_file": str(source),
        "source_sha256": sha256(source),
        "reference_file": str(reference),
        "partition": partition,
        "kingdom": "Eukaryota",
        "binary_rule": "membrane if Peripheral or Transmembrane or LipidAnchor; otherwise Soluble",
        "partition_records": initial_partition_records,
        "invalid_sequences_excluded": int(invalid_mask.sum()),
        "duplicate_sequences_excluded": int(duplicate_mask.sum()),
        "sequences_below_minimum_excluded": int(short_mask.sum()),
        "long_sequences_excluded": int(long_mask.sum()),
        "exact_v1_sequence_overlaps_excluded": int(exact_overlap_mask.sum()),
        "prepared_records": len(prepared),
        "class_counts": {
            "soluble": int(prepared["label"].eq(0).sum()),
            "membrane": int(prepared["label"].eq(1).sum()),
        },
        "min_residues": min_residues,
        "max_residues": max_residues,
        "homology_warning": (
            "Exact V1 overlaps were removed, but sequence-similarity filtering against "
            "V1 training data must be completed before calling this homology-independent."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return prepared


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/deeploc21_membrane.csv"))
    parser.add_argument("--reference", type=Path, default=Path("data/processed/deeploc_binary.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/deeploc21_external.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/processed/deeploc21_external_report.json"))
    parser.add_argument("--partition", type=int, default=4)
    parser.add_argument("--min-residues", type=int, default=1)
    parser.add_argument("--max-residues", type=int, default=1022)
    parser.add_argument("--download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.min_residues < 1 or (args.max_residues > 0 and args.max_residues < args.min_residues):
        raise ValueError("Require min-residues >= 1 and max-residues >= min-residues, or max-residues 0")
    if args.download and not args.source.exists():
        print(f"Downloading {SOURCE_URL}")
        download(SOURCE_URL, args.source)
    if not args.source.exists():
        raise FileNotFoundError(f"Missing source dataset: {args.source}; pass --download")
    frame = prepare(
        args.source, args.reference, args.output, args.report,
        args.partition, args.min_residues, args.max_residues,
    )
    print(frame["label"].value_counts().rename(index={0: "soluble", 1: "membrane"}))
    print(f"Saved {len(frame)} proteins to {args.output}")
    print(f"Saved preparation report to {args.report}")


if __name__ == "__main__":
    main()
