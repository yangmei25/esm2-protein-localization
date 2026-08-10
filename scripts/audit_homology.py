#!/usr/bin/env python3
"""Audit train-to-validation/test sequence similarity with local BLASTP."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


REQUIRED_COLUMNS = {"protein_id", "sequence", "split"}
THRESHOLDS = (30.0, 50.0, 70.0, 90.0)


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in frame.itertuples(index=False):
            handle.write(f">{row.protein_id}\n{row.sequence}\n")


def summarize_hits(hits: pd.DataFrame, query_frame: pd.DataFrame) -> dict:
    """Summarize cross-split hits at prespecified identity/coverage thresholds."""
    summary: dict[str, object] = {
        "query_proteins": int(len(query_frame)),
        "queries_with_any_blast_hit": int(hits["query_id"].nunique()),
        "threshold_definition": (
            "percent identity at or above threshold and alignment covering at "
            "least 80% of the shorter sequence"
        ),
        "threshold_counts": {},
    }
    for identity in THRESHOLDS:
        passing = hits[
            hits["percent_identity"].ge(identity)
            & hits["shorter_sequence_coverage"].ge(0.80)
        ]
        count = int(passing["query_id"].nunique())
        summary["threshold_counts"][f"identity_at_least_{int(identity)}"] = {
            "queries": count,
            "fraction": count / len(query_frame) if len(query_frame) else 0.0,
        }
    return summary


def run(args: argparse.Namespace) -> None:
    if shutil.which("makeblastdb") is None or shutil.which("blastp") is None:
        raise RuntimeError("BLAST+ executables makeblastdb and blastp are required")
    frame = pd.read_csv(args.data)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Processed data is missing columns: {sorted(missing)}")
    if frame["protein_id"].duplicated().any():
        raise ValueError("Protein IDs must be unique")

    train = frame[frame["split"].eq("train")].copy()
    queries = frame[frame["split"].isin(["validation", "test"])].copy()
    if train.empty or queries.empty:
        raise ValueError("Expected non-empty train and validation/test data")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hits_path = args.output_dir / "cross_split_blast_hits.csv"
    summary_path = args.output_dir / "summary.json"
    if not args.overwrite and (hits_path.exists() or summary_path.exists()):
        raise FileExistsError("Audit output exists; pass --overwrite to replace it")

    with TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        train_fasta = temporary / "train.fasta"
        query_fasta = temporary / "queries.fasta"
        database = temporary / "train_db"
        blast_output = temporary / "hits.tsv"
        write_fasta(train, train_fasta)
        write_fasta(queries, query_fasta)

        subprocess.run(
            [
                "makeblastdb",
                "-in",
                str(train_fasta),
                "-dbtype",
                "prot",
                "-out",
                str(database),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "blastp",
                "-query",
                str(query_fasta),
                "-db",
                str(database),
                "-evalue",
                str(args.evalue),
                "-max_target_seqs",
                str(args.max_target_seqs),
                "-num_threads",
                str(args.threads),
                "-outfmt",
                "6 qseqid sseqid pident length qlen slen evalue bitscore",
                "-out",
                str(blast_output),
            ],
            check=True,
        )
        columns = [
            "query_id",
            "train_id",
            "percent_identity",
            "alignment_length",
            "query_length",
            "train_length",
            "evalue",
            "bit_score",
        ]
        if blast_output.stat().st_size:
            hits = pd.read_csv(blast_output, sep="\t", names=columns)
        else:
            hits = pd.DataFrame(columns=columns)

    hits["shorter_sequence_coverage"] = hits["alignment_length"] / hits[
        ["query_length", "train_length"]
    ].min(axis=1)
    split_map = queries.set_index("protein_id")["split"]
    hits["query_split"] = hits["query_id"].map(split_map)
    hits = hits.sort_values(
        ["query_id", "bit_score", "percent_identity"],
        ascending=[True, False, False],
        ignore_index=True,
    )
    hits.to_csv(hits_path, index=False)

    summary = {
        "method": "BLASTP cross-split similarity audit",
        "blast_version": subprocess.run(
            ["blastp", "-version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0],
        "data": str(args.data),
        "training_proteins": int(len(train)),
        "evalue": args.evalue,
        "max_target_sequences_per_query": args.max_target_seqs,
        "interpretation": (
            "This is a diagnostic of the existing split, not a homology-aware "
            "split or an independent performance estimate."
        ),
        "all_queries": summarize_hits(hits, queries),
        "by_split": {
            split: summarize_hits(
                hits[hits["query_split"].eq(split)],
                queries[queries["split"].eq(split)],
            )
            for split in ("validation", "test")
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved hits to {hits_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit validation/test similarity to training proteins using BLASTP."
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/deeploc_binary.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/homology_audit")
    )
    parser.add_argument("--evalue", type=float, default=1e-3)
    parser.add_argument("--max-target-seqs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.evalue <= 0 or args.max_target_seqs < 1 or args.threads < 1:
        raise ValueError("evalue, max-target-seqs, and threads must be positive")
    run(args)


if __name__ == "__main__":
    main()
