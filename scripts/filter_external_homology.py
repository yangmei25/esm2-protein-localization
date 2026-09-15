#!/usr/bin/env python3
"""Remove external proteins homologous to V1 training proteins using BLASTP."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in frame.itertuples(index=False):
            handle.write(f">{row.protein_id}\n{row.sequence}\n")


def run(args: argparse.Namespace) -> None:
    if not shutil.which("makeblastdb") or not shutil.which("blastp"):
        raise RuntimeError("NCBI BLAST+ (makeblastdb and blastp) is required")
    reference = pd.read_csv(args.reference)
    train = reference[reference["split"].eq("train")][["protein_id", "sequence"]]
    external = pd.read_csv(args.external)
    if train.empty or external.empty:
        raise ValueError("Training reference and external cohort must be non-empty")
    if external["protein_id"].duplicated().any():
        raise ValueError("External protein IDs must be unique")

    with TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        train_fasta = temporary / "train.fasta"
        external_fasta = temporary / "external.fasta"
        database = temporary / "train_db"
        blast_output = temporary / "hits.tsv"
        write_fasta(train, train_fasta)
        write_fasta(external, external_fasta)
        subprocess.run(
            ["makeblastdb", "-in", str(train_fasta), "-dbtype", "prot", "-out", str(database)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            [
                "blastp", "-query", str(external_fasta), "-db", str(database),
                "-evalue", str(args.evalue), "-max_target_seqs", str(args.max_target_seqs),
                "-num_threads", str(args.threads),
                "-outfmt", "6 qseqid sseqid pident length qlen slen evalue bitscore",
                "-out", str(blast_output),
            ],
            check=True,
        )
        columns = [
            "query_id", "train_id", "percent_identity", "alignment_length",
            "query_length", "train_length", "evalue", "bit_score",
        ]
        hits = (
            pd.read_csv(blast_output, sep="\t", names=columns)
            if blast_output.stat().st_size else pd.DataFrame(columns=columns)
        )

    hits["shorter_sequence_coverage"] = hits["alignment_length"] / hits[
        ["query_length", "train_length"]
    ].min(axis=1)
    hits["fails_homology_gate"] = (
        hits["percent_identity"].ge(args.identity)
        & hits["shorter_sequence_coverage"].ge(args.coverage)
    )
    excluded_ids = set(hits.loc[hits["fails_homology_gate"], "query_id"])
    filtered = external[~external["protein_id"].isin(excluded_ids)].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.hits.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output, index=False)
    hits.sort_values(["query_id", "bit_score"], ascending=[True, False]).to_csv(
        args.hits, index=False
    )
    report = {
        "method": "BLASTP homology exclusion against V1 training proteins",
        "blast_version": subprocess.run(
            ["blastp", "-version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0],
        "external_candidates": len(external),
        "v1_training_proteins": len(train),
        "identity_threshold_percent": args.identity,
        "shorter_sequence_coverage_threshold": args.coverage,
        "homologous_external_proteins_excluded": len(excluded_ids),
        "external_proteins_retained": len(filtered),
        "retained_class_counts": {
            "soluble": int(filtered["label"].eq(0).sum()),
            "membrane": int(filtered["label"].eq(1).sum()),
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", type=Path, default=Path("data/processed/deeploc21_external_prefilter.csv"))
    parser.add_argument("--reference", type=Path, default=Path("data/processed/deeploc_binary.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/deeploc21_external_homology_filtered.csv"))
    parser.add_argument("--hits", type=Path, default=Path("results/v2_external_validation/homology_hits.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/v2_external_validation/homology_report.json"))
    parser.add_argument("--identity", type=float, default=30.0)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--evalue", type=float, default=1e-3)
    parser.add_argument("--max-target-seqs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
