#!/usr/bin/env python3
"""Generate the selected fine-tuned model's validation confusion matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def run(predictions_path: Path, metrics_path: Path, output_path: Path) -> None:
    predictions = pd.read_csv(predictions_path)
    required = {"label", "prediction"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

    matrix = confusion_matrix(
        predictions["label"], predictions["prediction"], labels=[0, 1]
    )
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = [[saved["tn"], saved["fp"]], [saved["fn"], saved["tp"]]]
    if matrix.tolist() != expected:
        raise ValueError(
            f"Predictions produce {matrix.tolist()}, but saved metrics report {expected}"
        )

    figure, axis = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Soluble", "Membrane"],
    )
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format=",d")
    axis.set_title("Fine-tuned ESM-2 validation confusion matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Saved {output_path}")
    print(f"TN={matrix[0, 0]}, FP={matrix[0, 1]}, FN={matrix[1, 0]}, TP={matrix[1, 1]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "results/finetune/esm2_t6_8M_mean/best_validation_predictions.csv"
        ),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "results/finetune/esm2_t6_8M_mean/best_validation_metrics.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/finetuned_validation_confusion_matrix.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.predictions, args.metrics, args.output)


if __name__ == "__main__":
    main()
