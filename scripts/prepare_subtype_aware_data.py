#!/usr/bin/env python3
"""Prepare subtype-aware DeepLoc 2.1 splits for binary ESM-2 fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SUBTYPES = ["Peripheral", "Transmembrane", "LipidAnchor"]
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZOU")
SPLIT_MAP = {0: "train", 1: "train", 2: "train", 3: "validation", 4: "test"}


def prepare(args: argparse.Namespace) -> pd.DataFrame:
    raw = pd.read_csv(args.source)
    required = {"ACC", "Kingdom", "Partition", "Sequence", "Soluble", *SUBTYPES}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Source CSV is missing columns: {sorted(missing)}")

    frame = raw.loc[
        raw["Kingdom"].eq(args.kingdom) & raw["Partition"].isin(SPLIT_MAP)
    ].copy()
    frame["sequence"] = frame["Sequence"].astype(str).str.upper().str.strip()
    frame["original_length"] = frame["sequence"].str.len()
    frame["label"] = frame[SUBTYPES].astype(bool).any(axis=1).astype(int)
    frame["split"] = frame["Partition"].map(SPLIT_MAP)

    invalid = frame["sequence"].map(lambda sequence: bool(set(sequence) - VALID_AMINO_ACIDS))
    too_long = frame["original_length"].gt(args.max_residues)
    duplicate = frame.duplicated("sequence", keep="first")
    multilabel_soluble_membrane = frame["label"].eq(1) & frame["Soluble"].astype(bool)
    keep = ~(invalid | too_long | duplicate)
    prepared = frame.loc[keep].copy()

    prepared["sample_weight"] = 1.0
    prepared.loc[prepared["LipidAnchor"].astype(bool), "sample_weight"] = args.lipid_anchor_weight
    prepared.loc[prepared["Peripheral"].astype(bool), "sample_weight"] = args.peripheral_weight
    prepared = prepared.rename(columns={"ACC": "protein_id"})[
        [
            "protein_id", "sequence", "label", "split", "original_length",
            *SUBTYPES, "Soluble", "Partition", "sample_weight",
        ]
    ].sort_values(["split", "protein_id"], ignore_index=True)

    if prepared["protein_id"].duplicated().any():
        raise ValueError("Protein IDs must be unique after preparation")
    if set(prepared["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("Prepared data must contain train, validation, and test")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)
    subtype_counts = prepared.groupby("split")[SUBTYPES + ["Soluble"]].sum().astype(int)
    report = {
        "source": str(args.source),
        "kingdom": args.kingdom,
        "split_rule": "partitions 0-2 train, 3 validation, 4 test",
        "selection_warning": "Partition 4 was previously inspected; do not call it an untouched final test.",
        "max_residues": args.max_residues,
        "peripheral_weight": args.peripheral_weight,
        "lipid_anchor_weight": args.lipid_anchor_weight,
        "excluded": {
            "invalid_sequence": int(invalid.sum()),
            "over_length": int(too_long.sum()),
            "duplicate_sequence": int(duplicate.sum()),
            "soluble_and_membrane_multilabel_retained": int(multilabel_soluble_membrane.sum()),
        },
        "split_counts": prepared["split"].value_counts().to_dict(),
        "subtype_counts": subtype_counts.to_dict(orient="index"),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return prepared


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/deeploc21_membrane.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/deeploc21_subtype_aware.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/processed/deeploc21_subtype_aware_report.json"))
    parser.add_argument("--kingdom", default="Eukaryota")
    parser.add_argument("--max-residues", type=int, default=1022)
    parser.add_argument("--peripheral-weight", type=float, default=3.0)
    parser.add_argument("--lipid-anchor-weight", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_residues < 1 or args.peripheral_weight <= 0 or args.lipid_anchor_weight <= 0:
        raise ValueError("Length and sampling weights must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
