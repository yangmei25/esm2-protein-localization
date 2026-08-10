#!/usr/bin/env python3
"""Train reproducible full-length hydrophobicity-based classical baselines."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from importlib.metadata import version

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scripts.prepare_data import parse_deeploc_header, read_fasta, validate_duplicates
except ModuleNotFoundError:  # Support direct execution from the scripts directory.
    from prepare_data import (  # type: ignore[no-redef]
        parse_deeploc_header,
        read_fasta,
        validate_duplicates,
    )


STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
KYTE_DOOLITTLE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
    "B": -3.5,
    "Z": -3.5,
    "X": 0.0,
    "U": 2.5,
    "O": -3.9,
}
HYDROPHOBIC_THRESHOLD = 1.6
LABEL_MAP = {"S": 0, "M": 1}


def longest_true_run(mask: np.ndarray) -> int:
    """Return the longest consecutive run of true values."""
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def window_means(values: np.ndarray, window_size: int) -> np.ndarray:
    """Calculate sliding-window means; short sequences use one global mean."""
    if len(values) < window_size:
        return np.array([values.mean()])
    return np.convolve(values, np.ones(window_size) / window_size, mode="valid")


def sequence_features(sequence: str) -> dict[str, float]:
    """Convert one full-length protein into 35 interpretable features."""
    if not sequence:
        raise ValueError("Protein sequence is empty")
    invalid = sorted(set(sequence) - set(KYTE_DOOLITTLE))
    if invalid:
        raise ValueError(f"Invalid amino-acid symbols: {invalid}")

    length = len(sequence)
    hydropathy = np.array([KYTE_DOOLITTLE[residue] for residue in sequence])
    features: dict[str, float] = {
        f"fraction_{residue}": sequence.count(residue) / length
        for residue in STANDARD_AMINO_ACIDS
    }
    features.update(
        {
            "ambiguous_fraction": sum(sequence.count(r) for r in "BXZOU") / length,
            "length": float(length),
            "log_length": float(np.log1p(length)),
            "mean_hydropathy": float(hydropathy.mean()),
            "std_hydropathy": float(hydropathy.std()),
            "min_hydropathy": float(hydropathy.min()),
            "max_hydropathy": float(hydropathy.max()),
            "hydrophobic_residue_fraction": float(
                (hydropathy > HYDROPHOBIC_THRESHOLD).mean()
            ),
            "charged_residue_fraction": sum(sequence.count(r) for r in "DEKR")
            / length,
            "aromatic_residue_fraction": sum(sequence.count(r) for r in "FWY")
            / length,
            "longest_hydrophobic_run": float(
                longest_true_run(hydropathy > HYDROPHOBIC_THRESHOLD)
            ),
        }
    )
    for window_size in (9, 19, 21):
        means = window_means(hydropathy, window_size)
        features[f"max_window_{window_size}_hydropathy"] = float(means.max())
        if window_size == 19:
            features["fraction_window_19_above_1.6"] = float(
                (means > HYDROPHOBIC_THRESHOLD).mean()
            )
    if len(features) != 35:
        raise AssertionError(f"Expected 35 features, produced {len(features)}")
    return features


def prepare_full_length_data(
    input_path: Path, validation_fraction: float, random_seed: int
) -> pd.DataFrame:
    """Build fixed binary splits without applying ESM-2's length limit."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    rows = []
    for header, sequence in read_fasta(input_path):
        record = parse_deeploc_header(header, sequence)
        if record.original_label not in LABEL_MAP:
            continue
        rows.append(
            {
                "protein_id": record.protein_id,
                "sequence": record.sequence,
                "label": LABEL_MAP[record.original_label],
                "official_split": record.official_split,
                "original_length": len(record.sequence),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No membrane or soluble proteins found")
    if frame["protein_id"].duplicated().any():
        raise ValueError("Duplicate protein IDs found")
    validate_duplicates(frame)

    development = frame[frame["official_split"].eq("development")].copy()
    development = development.drop_duplicates("sequence", keep="first")
    official_test = frame[frame["official_split"].eq("test")].copy()
    train_indices, validation_indices = train_test_split(
        development.index,
        test_size=validation_fraction,
        stratify=development["label"],
        random_state=random_seed,
    )
    development.loc[train_indices, "split"] = "train"
    development.loc[validation_indices, "split"] = "validation"
    official_test["split"] = "test"
    return pd.concat([development, official_test], ignore_index=True)


def build_models(random_seed: int) -> dict[str, object]:
    """Construct the two prespecified classifiers."""
    logistic_regression = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    random_forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )
    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
    }


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run(args: argparse.Namespace) -> None:
    """Train, evaluate, and save both classical baselines."""
    expected_outputs = [
        args.metrics_dir / "metrics.csv",
        args.metrics_dir / "metrics.json",
        args.metrics_dir / "predictions.csv",
        args.metrics_dir / "run_config.json",
        args.figure,
    ]
    if not args.overwrite and any(path.exists() for path in expected_outputs):
        raise FileExistsError("Baseline output exists; pass --overwrite to replace it")

    data = prepare_full_length_data(args.input, args.validation_fraction, args.seed)
    features = pd.DataFrame([sequence_features(sequence) for sequence in data["sequence"]])
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Feature matrix contains a non-finite value")

    labels = data["label"].to_numpy(dtype=np.int64)
    splits = data["split"].to_numpy(dtype=str)
    train_mask = splits == "train"
    models = build_models(args.seed)
    metrics_rows: list[dict] = []
    prediction_rows: list[dict] = []
    validation_probabilities: dict[str, np.ndarray] = {}

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        model.fit(features[train_mask], labels[train_mask])
        joblib.dump(model, args.model_dir / f"{model_name}.joblib")
        for split_name in ("validation", "test"):
            mask = splits == split_name
            probabilities = model.predict_proba(features[mask])[:, 1]
            predictions = (probabilities >= 0.5).astype(np.int64)
            metrics = calculate_metrics(labels[mask], probabilities)
            metrics_rows.append({"model": model_name, "split": split_name, **metrics})
            if split_name == "validation":
                validation_probabilities[model_name] = probabilities

            indices = np.flatnonzero(mask)
            for index, probability, prediction in zip(
                indices, probabilities, predictions
            ):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "protein_id": data.iloc[index]["protein_id"],
                        "label": int(labels[index]),
                        "prediction": int(prediction),
                        "membrane_probability": float(probability),
                    }
                )

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(args.metrics_dir / "metrics.csv", index=False)
    (args.metrics_dir / "metrics.json").write_text(
        json.dumps(metrics_rows, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(prediction_rows).to_csv(
        args.metrics_dir / "predictions.csv", index=False
    )
    run_config = {
        "input": str(args.input),
        "random_seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "probability_threshold": 0.5,
        "proteins": len(data),
        "features": len(features.columns),
        "feature_names": features.columns.tolist(),
        "maximum_sequence_length": int(data["original_length"].max()),
        "proteins_longer_than_1022": int((data["original_length"] > 1022).sum()),
        "split_counts": data["split"].value_counts().sort_index().to_dict(),
        "software_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": version("scikit-learn"),
            "matplotlib": version("matplotlib"),
            "joblib": version("joblib"),
        },
    }
    (args.metrics_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
    }
    validation_mask = splits == "validation"
    for model_name, probabilities in validation_probabilities.items():
        RocCurveDisplay.from_predictions(
            labels[validation_mask],
            probabilities,
            name=display_names[model_name],
            ax=axis,
        )
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1,
        label="Random ranking (AUC = 0.5)",
    )
    axis.set_title("Validation ROC: hydrophobicity classifiers")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.savefig(args.figure, dpi=180)
    plt.close(figure)

    print(f"Proteins: {len(data):,}; features: {len(features.columns)}")
    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved metrics to {args.metrics_dir}")
    print(f"Saved ROC figure to {args.figure}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train full-length hydrophobicity-based classical baselines."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/raw/deeploc_data.fasta")
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("results/metrics/classical_baselines"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("results/models/classical_baselines"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(
            "results/figures/validation_roc_hydrophobicity_classifiers.png"
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Raw DeepLoc FASTA not found: {args.input}")
    run(args)


if __name__ == "__main__":
    main()
