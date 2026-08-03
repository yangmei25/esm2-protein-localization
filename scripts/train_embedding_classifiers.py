#!/usr/bin/env python3
"""Compare Logistic Regression on three frozen ESM-2 representations.

Only the training and validation splits are used. The official test split is
intentionally left untouched until one representation has been selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPRESENTATIONS = ("first_token", "mean", "max")


def load_embedding_cache(path: Path) -> dict[str, np.ndarray]:
    """Load the NPZ safely and verify required aligned arrays."""
    if not path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {path}")

    with np.load(path, allow_pickle=False) as cache:
        required = {
            "protein_id",
            "label",
            "split",
            "original_length",
            *REPRESENTATIONS,
        }
        missing = required - set(cache.files)
        if missing:
            raise ValueError(f"Embedding cache is missing arrays: {sorted(missing)}")
        arrays = {name: cache[name].copy() for name in required}

    row_count = len(arrays["label"])
    for name, array in arrays.items():
        if len(array) != row_count:
            raise ValueError(f"{name} is not aligned with the label array")
    if not set(np.unique(arrays["label"])).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1")
    if set(np.unique(arrays["split"])) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, and test splits")
    for representation in REPRESENTATIONS:
        if arrays[representation].ndim != 2:
            raise ValueError(f"{representation} must be a two-dimensional matrix")
        if not np.isfinite(arrays[representation]).all():
            raise ValueError(f"{representation} contains non-finite values")
    return arrays


def build_pipeline(random_seed: int, c_value: float) -> Pipeline:
    """Create a fresh scaler and class-balanced classifier."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Calculate validation metrics at the fixed 0.5 probability threshold."""
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Logistic Regression across frozen ESM-2 representations."
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("results/embeddings/esm2_t6_8M_deeploc.npz"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("results/models/embedding_classifiers"),
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("results/metrics/embedding_classifiers"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.c_value <= 0:
        raise ValueError("--c-value must be positive")

    summary_path = args.metrics_dir / "validation_metrics.csv"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError("Results already exist; pass --overwrite to replace them")

    arrays = load_embedding_cache(args.embeddings)
    labels = arrays["label"].astype(np.int64, copy=False)
    splits = arrays["split"]
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"

    print(f"Training proteins: {int(train_mask.sum()):,}")
    print(f"Validation proteins: {int(validation_mask.sum()):,}")
    print(f"Reserved test proteins: {int(test_mask.sum()):,} (not evaluated)")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    prediction_rows = []

    for representation in REPRESENTATIONS:
        features = arrays[representation]
        pipeline = build_pipeline(args.seed, args.c_value)
        pipeline.fit(features[train_mask], labels[train_mask])

        validation_probabilities = pipeline.predict_proba(features[validation_mask])[:, 1]
        validation_predictions = (validation_probabilities >= 0.5).astype(np.int64)
        metrics = classification_metrics(
            labels[validation_mask], validation_probabilities
        )
        metric_rows.append({"representation": representation, **metrics})

        validation_indices = np.flatnonzero(validation_mask)
        for index, probability, prediction in zip(
            validation_indices, validation_probabilities, validation_predictions
        ):
            prediction_rows.append(
                {
                    "representation": representation,
                    "protein_id": arrays["protein_id"][index],
                    "label": int(labels[index]),
                    "prediction": int(prediction),
                    "membrane_probability": float(probability),
                }
            )

        model_path = args.model_dir / f"{representation}_logistic_regression.joblib"
        joblib.dump(pipeline, model_path)
        print(
            f"{representation:>11}: "
            f"F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}"
        )

    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["f1", "roc_auc"], ascending=False, ignore_index=True
    )
    predictions_frame = pd.DataFrame(prediction_rows)
    selected = metrics_frame.iloc[0].to_dict()

    metrics_frame.to_csv(summary_path, index=False)
    predictions_frame.to_csv(args.metrics_dir / "validation_predictions.csv", index=False)
    (args.metrics_dir / "validation_metrics.json").write_text(
        json.dumps(metric_rows, indent=2) + "\n", encoding="utf-8"
    )
    selection = {
        "selection_split": "validation",
        "primary_metric": "f1",
        "tie_breaker": "roc_auc",
        "probability_threshold": 0.5,
        "selected_representation": selected["representation"],
        "selected_validation_metrics": {
            key: float(selected[key])
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
        },
        "logistic_regression_C": args.c_value,
        "class_weight": "balanced",
        "random_seed": args.seed,
        "test_set_evaluated": False,
    }
    (args.metrics_dir / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )

    print("\nValidation comparison:")
    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected representation: {selection['selected_representation']}")
    print("The official test split has not been evaluated.")


if __name__ == "__main__":
    main()
