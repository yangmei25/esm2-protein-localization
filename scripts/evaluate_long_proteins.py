#!/usr/bin/env python3
"""Evaluate overlapping-window aggregation on labeled long proteins."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from predict import MAX_RESIDUES, make_sequence_windows
from train_finetune import (
    ESM2MeanPoolingClassifier,
    choose_device,
    classification_metrics,
    load_model_state_compatibly,
    resolve_amp,
)


REQUIRED_COLUMNS = {"protein_id", "sequence", "label", "original_length"}


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype | None):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=dtype)
    from contextlib import nullcontext
    return nullcontext()


def evaluate_overlap(
    frame: pd.DataFrame,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    overlap: int,
    batch_size: int,
    threshold: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    stride = MAX_RESIDUES - overlap
    flat_windows: list[dict] = []
    for protein_index, row in enumerate(frame.itertuples(index=False)):
        for window in make_sequence_windows(row.sequence, MAX_RESIDUES, stride):
            flat_windows.append(
                {
                    "protein_index": protein_index,
                    "protein_id": row.protein_id,
                    **window,
                }
            )

    probabilities: list[float] = []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(flat_windows), batch_size):
            batch = flat_windows[offset : offset + batch_size]
            tokenized = tokenizer(
                [window["sequence"] for window in batch],
                padding=True,
                truncation=False,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            inputs = {
                key: tokenized[key].to(device)
                for key in ("input_ids", "attention_mask", "special_tokens_mask")
            }
            with autocast_context(device, amp_enabled, amp_dtype):
                logits = model(**inputs)
            probabilities.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    elapsed = time.perf_counter() - started

    window_frame = pd.DataFrame(
        {
            "protein_index": [window["protein_index"] for window in flat_windows],
            "protein_id": [window["protein_id"] for window in flat_windows],
            "start": [window["start"] for window in flat_windows],
            "end": [window["end"] for window in flat_windows],
            "membrane_probability": probabilities,
        }
    )
    top_indices = window_frame.groupby("protein_index")["membrane_probability"].idxmax()
    top = window_frame.loc[top_indices].sort_values("protein_index").reset_index(drop=True)
    protein_predictions = frame[["protein_id", "label", "original_length"]].copy()
    protein_predictions["membrane_probability"] = top["membrane_probability"].to_numpy()
    protein_predictions["prediction"] = (
        protein_predictions["membrane_probability"].ge(threshold).astype(int)
    )
    protein_predictions["top_window_start"] = top["start"].to_numpy()
    protein_predictions["top_window_end"] = top["end"].to_numpy()
    counts = window_frame.groupby("protein_index").size().sort_index()
    protein_predictions["number_of_windows"] = counts.to_numpy()

    labels = protein_predictions["label"].to_numpy(dtype=int)
    scores = protein_predictions["membrane_probability"].to_numpy(dtype=float)
    if threshold != 0.5:
        raise ValueError("Long-protein comparison currently requires threshold 0.5")
    metrics = classification_metrics(labels, scores)
    metrics.update(
        {
            "overlap": overlap,
            "stride": stride,
            "proteins": len(frame),
            "windows": len(window_frame),
            "average_windows_per_protein": len(window_frame) / len(frame),
            "elapsed_seconds": elapsed,
            "proteins_per_second": len(frame) / elapsed,
            "false_positive_rate": metrics["fp"] / (metrics["fp"] + metrics["tn"]),
        }
    )
    return metrics, protein_predictions, window_frame.drop(columns="protein_index")


def run(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.data)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Long-protein data is missing columns: {sorted(missing)}")
    if frame.empty or not frame["original_length"].gt(MAX_RESIDUES).all():
        raise ValueError("Every evaluation protein must be longer than 1,022 residues")
    if any(overlap < 0 or overlap >= MAX_RESIDUES for overlap in args.overlaps):
        raise ValueError("Every overlap must be between 0 and 1,021")

    device = choose_device(args.device)
    amp_enabled, amp_dtype = resolve_amp(device, args.mixed_precision)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_name"])
    model = ESM2MeanPoolingClassifier(checkpoint["model_name"], checkpoint["dropout"])
    load_model_state_compatibly(model, checkpoint["model_state_dict"])
    model.to(device).eval()
    threshold = float(checkpoint.get("threshold", 0.5))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for overlap in args.overlaps:
        print(f"Evaluating overlap {overlap}", flush=True)
        metrics, protein_predictions, window_predictions = evaluate_overlap(
            frame, model, tokenizer, device, amp_enabled, amp_dtype,
            overlap, args.batch_size, threshold,
        )
        rows.append(metrics)
        protein_predictions.to_csv(
            args.output_dir / f"protein_predictions_overlap_{overlap}.csv", index=False
        )
        window_predictions.to_csv(
            args.output_dir / f"window_predictions_overlap_{overlap}.csv", index=False
        )
    summary = pd.DataFrame(rows).sort_values("overlap").reset_index(drop=True)
    summary.to_csv(args.output_dir / "overlap_metrics.csv", index=False)
    metadata = {
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "model_name": checkpoint["model_name"],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "aggregation": "maximum_window_probability",
        "window_size": MAX_RESIDUES,
        "threshold": threshold,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlaps", type=int, nargs="+", default=[128, 256, 511])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--mixed-precision", choices=["auto", "none", "fp16", "bf16"], default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
