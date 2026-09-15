"""Reusable, CPU-only tables and figures for the V2 outcome report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
METRIC_LABELS = ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")
SIZES = ("8M", "35M", "150M")


def load_summary(path: str | Path) -> dict:
    """Load the compact, machine-readable V2 summary."""
    return json.loads(Path(path).read_text())


def model_selection_table(summary: dict) -> pd.DataFrame:
    """Compare frozen and fine-tuned validation metrics at each ESM-2 size."""
    scaling = summary["model_scaling"]
    rows = []
    for size in SIZES:
        frozen = scaling["frozen_mean_pooling"][size]
        fine = scaling["finetuned_three_seed_mean"][size]
        for method, values in (("Frozen", frozen), ("Fine-tuned", fine)):
            rows.append(
                {
                    "ESM-2 size": size,
                    "Training method": method,
                    "Precision": values["precision"],
                    "Recall": values["recall"],
                    "F1": values["f1"],
                    "ROC-AUC": values["roc_auc"],
                }
            )
    return pd.DataFrame(rows).set_index(["ESM-2 size", "Training method"])


def multiseed_table(summary: dict) -> pd.DataFrame:
    """Return mean and SD for all fine-tuned model-size metrics."""
    scaling = summary["model_scaling"]["finetuned_three_seed_mean"]
    rows = []
    for size in SIZES:
        for metric, label in zip(METRICS, METRIC_LABELS):
            rows.append(
                {
                    "ESM-2 size": size,
                    "Metric": label,
                    "Mean": scaling[size][metric],
                    "SD": scaling[size][f"{metric}_sd"],
                }
            )
    return pd.DataFrame(rows)


def plot_multiseed_scaling(summary: dict, output: str | Path | None = None):
    """Plot fine-tuned validation means with three-seed SD error bars."""
    data = summary["model_scaling"]["finetuned_three_seed_mean"]
    x = np.array([8, 35, 150])
    styles = {
        "accuracy": ("#2f6690", "o"),
        "precision": ("#d43d2c", "s"),
        "recall": ("#2a8f74", "^"),
        "f1": ("#c78300", "D"),
        "roc_auc": ("#6f4fa3", "v"),
    }
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for metric, label in zip(METRICS, METRIC_LABELS):
        means = [data[size][metric] for size in SIZES]
        errors = [data[size][f"{metric}_sd"] for size in SIZES]
        color, marker = styles[metric]
        ax.errorbar(
            x, means, yerr=errors, label=label, color=color, marker=marker,
            markersize=6.5, linewidth=1.9, capsize=3.5, elinewidth=1.3,
        )
    ax.set_xscale("log")
    ax.set_xticks(x, SIZES)
    ax.set_ylim(0.83, 1.0)
    ax.set_xlabel("ESM-2 model size (log scale)", labelpad=8)
    ax.set_ylabel("Validation score (mean ± SD, 3 seeds)", labelpad=8)
    ax.set_title("Fine-tuned ESM-2 performance by model size", weight="bold", pad=13)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#dddddd", fontsize=9)
    ax.grid(axis="y", alpha=0.28)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0.06)
    fig.tight_layout(pad=1.1)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    return fig, ax


def external_tables(summary: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return overall internal/external comparison and subtype recall diagnosis."""
    internal = summary["model_scaling"]["finetuned_three_seed_mean"]["150M"]
    external = summary["external_binary"]
    overall = pd.DataFrame(
        {
            "Internal validation": [internal[key] for key in METRICS],
            "External cohort": [external["three_seed_mean"][key] for key in METRICS],
        },
        index=METRIC_LABELS,
    )
    overall["Change"] = overall["External cohort"] - overall["Internal validation"]
    overall.index.name = "Metric"
    subtype = pd.DataFrame(
        {
            "Proteins": external["subtype_counts"],
            "Recall": external["subtype_recall"],
            "Mean false negatives": external["mean_false_negatives"],
        }
    ).rename_axis("Membrane subtype").reset_index()
    subtype["Membrane subtype"] = subtype["Membrane subtype"].replace(
        {"transmembrane": "Transmembrane", "peripheral": "Peripheral", "lipidanchor": "LipidAnchor"}
    )
    return overall, subtype


def subtype_tradeoff_tables(summary: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return subtype-recall changes and legacy-binary compatibility changes."""
    previous = summary["external_binary"]["subtype_recall"]
    current = summary["subtype_aware_seed42"]["external"]["subtype_recall"]
    subtype = pd.DataFrame({"previous_binary": previous, "subtype_aware": current})
    subtype["change"] = subtype["subtype_aware"] - subtype["previous_binary"]
    old_binary = summary["model_scaling"]["finetuned_three_seed_mean"]["150M"]
    new_binary = summary["subtype_aware_seed42"]["legacy_binary"]
    binary = pd.DataFrame(
        {
            "previous_binary": [old_binary[key] for key in METRICS],
            "subtype_aware": [new_binary[key] for key in METRICS],
        },
        index=METRICS,
    )
    binary["change"] = binary["subtype_aware"] - binary["previous_binary"]
    return subtype, binary


def long_protein_tables(summary: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return overlap selection and final three-seed long-protein results."""
    long = summary["long_proteins"]
    overlap = pd.DataFrame(long["overlap_selection_seed42"]).T
    overlap.index.name = "Overlap (aa)"
    final = pd.DataFrame(
        {"Mean ± SD": [long[key] for key in METRICS]}, index=METRIC_LABELS
    )
    return overlap, final


def export_tables(summary: dict, output_dir: str | Path) -> list[Path]:
    """Write every V2 report table to CSV and return the created paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {"model_selection": model_selection_table(summary), "multiseed": multiseed_table(summary)}
    tables["external_overall"], tables["external_subtypes"] = external_tables(summary)
    tables["subtype_recall_tradeoff"], tables["binary_tradeoff"] = subtype_tradeoff_tables(summary)
    tables["long_overlap_selection"], tables["long_final"] = long_protein_tables(summary)
    paths = []
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path)
        paths.append(path)
    return paths
