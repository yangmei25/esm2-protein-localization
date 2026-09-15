#!/usr/bin/env python3
"""Build CPU-only V2 report tables and the multi-seed scaling figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from esm2_localization.v2_report import export_tables, load_summary, plot_multiseed_scaling


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", type=Path,
        default=PROJECT_ROOT / "results/metrics/v2_project_summary.json",
    )
    parser.add_argument(
        "--table-dir", type=Path,
        default=PROJECT_ROOT / "results/metrics/v2_report_tables",
    )
    parser.add_argument(
        "--figure", type=Path,
        default=PROJECT_ROOT / "results/figures/v2_result_2_multiseed_scaling.png",
    )
    args = parser.parse_args()
    summary = load_summary(args.summary)
    paths = export_tables(summary, args.table_dir)
    figure, _ = plot_multiseed_scaling(summary, args.figure)
    figure.clear()
    print(f"Saved figure: {args.figure}")
    print(f"Saved {len(paths)} tables under: {args.table_dir}")


if __name__ == "__main__":
    main()
