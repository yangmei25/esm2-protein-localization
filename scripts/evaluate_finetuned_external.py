#!/usr/bin/env python3
"""Evaluate a saved fine-tuned ESM-2 checkpoint on an external binary cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_finetune import (
    ESM2MeanPoolingClassifier,
    ProteinDataset,
    choose_device,
    evaluate,
    load_model_state_compatibly,
    make_collate_fn,
    resolve_amp,
    seed_everything,
)


REQUIRED_COLUMNS = {"protein_id", "sequence", "label", "original_length"}


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    frame = pd.read_csv(args.data)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"External data is missing columns: {sorted(missing)}")
    if frame.empty or set(frame["label"].unique()) != {0, 1}:
        raise ValueError("External data must contain both labels 0 and 1")
    if frame["original_length"].gt(1022).any():
        raise ValueError("External data contains a sequence longer than 1,022 residues")

    device = choose_device(args.device)
    amp_enabled, amp_dtype = resolve_amp(device, args.mixed_precision)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    loader = DataLoader(
        ProteinDataset(frame),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = ESM2MeanPoolingClassifier(model_name, float(checkpoint["dropout"]))
    load_model_state_compatibly(model, checkpoint["model_state_dict"])
    model = model.to(device)
    metrics, predictions = evaluate(
        model, loader, nn.CrossEntropyLoss(), device, amp_enabled, amp_dtype
    )
    metrics.update(
        {
            "external_proteins": len(frame),
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "model_name": model_name,
            "threshold": float(checkpoint.get("threshold", 0.5)),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "external_predictions.csv", index=False)
    (args.output_dir / "external_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--mixed-precision", choices=["auto", "none", "fp16", "bf16"], default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
