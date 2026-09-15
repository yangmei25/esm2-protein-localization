#!/usr/bin/env python3
"""Evaluate a four-label ESM-2 checkpoint on multi-label or legacy binary data.

Evaluator version: restart-safe-v3
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from train_finetune import classification_metrics, choose_device, resolve_amp
    from train_multilabel_finetune import ESM2MultiLabelClassifier, LABELS, multilabel_metrics
except ModuleNotFoundError:
    from scripts.train_finetune import classification_metrics, choose_device, resolve_amp
    from scripts.train_multilabel_finetune import ESM2MultiLabelClassifier, LABELS, multilabel_metrics


class SequenceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.ids = frame["protein_id"].astype(str).tolist()
        self.sequences = frame["sequence"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        return {"protein_id": self.ids[index], "sequence": self.sequences[index]}


def make_collate(tokenizer):
    def collate(examples):
        batch = tokenizer(
            [item["sequence"] for item in examples], padding=True, truncation=False,
            return_special_tokens_mask=True, return_tensors="pt",
        )
        batch["protein_ids"] = [item["protein_id"] for item in examples]
        return batch
    return collate


@torch.inference_mode()
def predict(model, loader, device, amp_enabled, amp_dtype) -> pd.DataFrame:
    model.eval(); ids, probabilities = [], []
    for batch in tqdm(loader, desc="Inference"):
        inputs = {key: batch[key].to(device) for key in ("input_ids", "attention_mask", "special_tokens_mask")}
        context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else nullcontext()
        with context:
            logits = model(**inputs)
        ids.extend(batch["protein_ids"])
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    probability_array = np.concatenate(probabilities)
    output = {"protein_id": ids}
    for index, label in enumerate(LABELS):
        key = label.lower()
        output[f"{key}_probability"] = probability_array[:, index]
        output[f"{key}_prediction"] = (probability_array[:, index] >= 0.5).astype(int)
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--mode", choices=["binary", "multilabel"], required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--mixed-precision", choices=["auto", "none", "fp16", "bf16"], default="auto")
    args = parser.parse_args()

    print(f"Reading evaluation data: {args.data}", flush=True)
    frame = pd.read_csv(args.data)
    required = {"protein_id", "sequence", "split", "original_length"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Data is missing columns: {sorted(missing)}")
    frame = frame.loc[frame["split"].eq(args.split)].reset_index(drop=True)
    if frame.empty or frame["original_length"].gt(1022).any():
        raise ValueError("Selected split is empty or contains an over-length sequence")
    print(f"Selected {len(frame):,} proteins from split={args.split!r}.", flush=True)

    device = choose_device(args.device)
    amp_enabled, amp_dtype = resolve_amp(device, args.mixed_precision)
    print(f"Loading checkpoint from Drive: {args.checkpoint}", flush=True)
    # Load on CPU first to avoid temporarily keeping both the checkpoint and model on GPU.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("labels") != LABELS:
        raise ValueError("Checkpoint label order does not match the evaluator")
    checkpoint_epoch = int(checkpoint["epoch"])
    print(f"Loading tokenizer: {checkpoint['model_name']}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_name"])
    # num_workers=0 is more reliable inside a restarted Colab notebook runtime.
    loader = DataLoader(SequenceDataset(frame), batch_size=args.batch_size, shuffle=False, collate_fn=make_collate(tokenizer), num_workers=0, pin_memory=device.type == "cuda")
    print(f"Loading model on {device}; inference batch size={args.batch_size}.", flush=True)
    model = ESM2MultiLabelClassifier(checkpoint["model_name"], float(checkpoint["dropout"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    del checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Starting inference...", flush=True)
    predictions = predict(model, loader, device, amp_enabled, amp_dtype)
    evaluated = frame.merge(predictions, on="protein_id", validate="one_to_one")

    if args.mode == "binary":
        if "label" not in evaluated:
            raise ValueError("Binary evaluation requires a label column")
        membrane_columns = [f"{label.lower()}_probability" for label in LABELS if label != "Soluble"]
        membrane_probability = evaluated[membrane_columns].max(axis=1).to_numpy()
        metrics = classification_metrics(evaluated["label"].to_numpy(), membrane_probability)
        evaluated["membrane_probability"] = membrane_probability
        evaluated["binary_prediction"] = (membrane_probability >= 0.5).astype(int)
    else:
        missing = set(LABELS) - set(evaluated.columns)
        if missing:
            raise ValueError(f"Multi-label evaluation requires: {sorted(missing)}")
        targets = evaluated[LABELS].astype(int).to_numpy()
        probabilities = evaluated[[f"{label.lower()}_probability" for label in LABELS]].to_numpy()
        metrics = multilabel_metrics(targets, probabilities)

    metrics.update({"mode": args.mode, "split": args.split, "proteins": len(evaluated), "checkpoint_epoch": checkpoint_epoch})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
