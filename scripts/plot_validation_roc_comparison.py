#!/usr/bin/env python3
"""Compare three model tracks on the same ESM-compatible validation proteins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay, roc_auc_score

try:
    from scripts.train_classical_baselines import build_models, sequence_features
except ModuleNotFoundError:  # Support direct execution from scripts/.
    from train_classical_baselines import build_models, sequence_features


def load_predictions(path: Path, representation: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if representation is not None:
        if "representation" not in frame.columns:
            raise ValueError(f"{path} has no representation column")
        frame = frame[frame["representation"].eq(representation)].copy()
    required = {"protein_id", "label", "membrane_probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["protein_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate protein IDs")
    return frame[list(required)].copy()


def run(args: argparse.Namespace) -> None:
    data = pd.read_csv(args.data)
    train = data[data["split"].eq("train")].copy()
    validation = data[data["split"].eq("validation")].copy()
    if train.empty or validation.empty:
        raise ValueError("Processed data must contain train and validation splits")

    train_features = pd.DataFrame(
        [sequence_features(sequence) for sequence in train["sequence"]]
    )
    validation_features = pd.DataFrame(
        [sequence_features(sequence) for sequence in validation["sequence"]]
    )
    classical = build_models(args.seed)["random_forest"]
    classical.fit(train_features, train["label"])
    classical_frame = validation[["protein_id", "label"]].copy()
    classical_frame["membrane_probability"] = classical.predict_proba(
        validation_features
    )[:, 1]

    frozen = load_predictions(args.frozen_predictions, representation="mean")
    finetuned = load_predictions(args.finetuned_predictions)
    expected_ids = set(validation["protein_id"])
    for name, frame in {"frozen mean": frozen, "fine-tuned": finetuned}.items():
        if set(frame["protein_id"]) != expected_ids:
            raise ValueError(f"{name} predictions do not match the validation split")

    ordered = validation[["protein_id", "label"]].copy()
    series = {
        "Classical Random Forest": classical_frame,
        "Frozen ESM-2 + Logistic Regression": frozen,
        "Fine-tuned ESM-2": finetuned,
    }
    auc_values: dict[str, float] = {}
    figure, axis = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for (name, frame), color in zip(series.items(), colors):
        aligned = ordered.merge(
            frame[["protein_id", "label", "membrane_probability"]],
            on="protein_id",
            suffixes=("_expected", "_predictions"),
            validate="one_to_one",
        )
        if not aligned["label_expected"].equals(aligned["label_predictions"]):
            raise ValueError(f"{name} labels do not match processed validation data")
        auc = float(
            roc_auc_score(
                aligned["label_expected"], aligned["membrane_probability"]
            )
        )
        auc_values[name] = auc
        RocCurveDisplay.from_predictions(
            aligned["label_expected"],
            aligned["membrane_probability"],
            name=name,
            curve_kwargs={"color": color, "linewidth": 2},
            ax=axis,
        )

    axis.plot(
        [0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
        label="Random ranking (AUC = 0.500)",
    )
    axis.set_title(
        f"Validation ROC comparison on the same {len(validation):,} proteins"
    )
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)

    result = {
        "split": "validation",
        "proteins": int(len(validation)),
        "shared_cohort": True,
        "classical_model_note": (
            "Random Forest retrained on the ESM-compatible training split using "
            "the same 35 handcrafted features and fixed hyperparameters."
        ),
        "roc_auc": auc_values,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/deeploc_binary.csv")
    )
    parser.add_argument(
        "--frozen-predictions",
        type=Path,
        default=Path(
            "results/metrics/embedding_classifiers/validation_predictions.csv"
        ),
    )
    parser.add_argument(
        "--finetuned-predictions",
        type=Path,
        default=Path(
            "results/finetune/esm2_t6_8M_mean/best_validation_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/validation_roc_model_comparison.png"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/metrics/validation_roc_comparison.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
