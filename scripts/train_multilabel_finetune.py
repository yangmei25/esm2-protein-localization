#!/usr/bin/env python3
"""Fine-tune ESM-2 for four-label membrane-association prediction."""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from train_finetune import choose_device, resolve_amp, seed_everything
except ModuleNotFoundError:  # Support importing as scripts.train_multilabel_finetune.
    from scripts.train_finetune import choose_device, resolve_amp, seed_everything


LABELS = ["Soluble", "Transmembrane", "Peripheral", "LipidAnchor"]
REQUIRED = {"protein_id", "sequence", "split", "original_length", *LABELS}


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Multi-label CSV is missing columns: {sorted(missing)}")
    if frame.empty or frame["protein_id"].duplicated().any():
        raise ValueError("Dataset must be non-empty with unique protein IDs")
    if set(frame["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, and test splits")
    if frame["original_length"].gt(1022).any():
        raise ValueError("Sequence exceeds the ESM-2 1,022-residue limit")
    for label in LABELS:
        if not set(frame[label].unique()).issubset({0, 1, False, True}):
            raise ValueError(f"{label} must be binary")
    if frame[LABELS].sum(axis=1).eq(0).any():
        raise ValueError("Every protein must have at least one label")
    return frame


class MultiLabelProteinDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.ids = frame["protein_id"].astype(str).tolist()
        self.sequences = frame["sequence"].astype(str).tolist()
        self.targets = frame[LABELS].astype(np.float32).to_numpy()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        return {"protein_id": self.ids[index], "sequence": self.sequences[index], "targets": self.targets[index]}


def make_collate_fn(tokenizer):
    def collate(examples: list[dict]) -> dict:
        batch = tokenizer(
            [item["sequence"] for item in examples], padding=True, truncation=False,
            return_special_tokens_mask=True, return_tensors="pt",
        )
        batch["targets"] = torch.tensor(np.stack([item["targets"] for item in examples]), dtype=torch.float32)
        batch["protein_ids"] = [item["protein_id"] for item in examples]
        return batch
    return collate


class ESM2MultiLabelClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(int(self.encoder.config.hidden_size), len(LABELS))

    def forward(self, input_ids, attention_mask, special_tokens_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        residue_mask = attention_mask.bool() & ~special_tokens_mask.bool()
        mask = residue_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.classifier(self.dropout(pooled))


def model_inputs(batch: dict, device: torch.device) -> dict:
    return {key: batch[key].to(device) for key in ("input_ids", "attention_mask", "special_tokens_mask")}


def amp_context(device, enabled, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if enabled else nullcontext()


def multilabel_metrics(targets: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    result = {
        "exact_match_accuracy": float(accuracy_score(targets, predictions)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }
    for index, label in enumerate(LABELS):
        key = label.lower()
        result[f"{key}_precision"] = float(precision_score(targets[:, index], predictions[:, index], zero_division=0))
        result[f"{key}_recall"] = float(recall_score(targets[:, index], predictions[:, index], zero_division=0))
        result[f"{key}_f1"] = float(f1_score(targets[:, index], predictions[:, index], zero_division=0))
        result[f"{key}_roc_auc"] = float(roc_auc_score(targets[:, index], probabilities[:, index]))
    return result


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device, amp_enabled, amp_dtype):
    model.eval()
    losses, ids, targets, probabilities = [], [], [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        target = batch["targets"].to(device)
        with amp_context(device, amp_enabled, amp_dtype):
            logits = model(**model_inputs(batch, device))
            loss = loss_fn(logits, target)
        losses.append(float(loss.cpu()))
        ids.extend(batch["protein_ids"])
        targets.append(target.cpu().numpy())
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    target_array, probability_array = np.concatenate(targets), np.concatenate(probabilities)
    metrics = multilabel_metrics(target_array, probability_array)
    metrics["loss"] = float(np.mean(losses))
    output = {"protein_id": ids}
    for i, label in enumerate(LABELS):
        key = label.lower()
        output[f"{key}_label"] = target_array[:, i].astype(int)
        output[f"{key}_probability"] = probability_array[:, i]
        output[f"{key}_prediction"] = (probability_array[:, i] >= 0.5).astype(int)
    return metrics, pd.DataFrame(output)


def train(args: argparse.Namespace) -> None:
    print("Starting multi-label ESM-2 training process.", flush=True)
    seed_everything(args.seed)
    print(f"Loading prepared dataset: {args.data}", flush=True)
    frame = load_data(args.data)
    train_frame = frame[frame["split"].eq("train")].reset_index(drop=True)
    validation_frame = frame[frame["split"].eq("validation")].reset_index(drop=True)
    device = choose_device(args.device)
    amp_enabled, amp_dtype = resolve_amp(device, args.mixed_precision)
    print(
        f"Dataset ready — train={len(train_frame):,}, validation={len(validation_frame):,}; "
        f"device={device}; mixed_precision={amp_dtype}",
        flush=True,
    )
    print(f"Loading tokenizer: {args.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate = make_collate_fn(tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(MultiLabelProteinDataset(train_frame), batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=generator, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(MultiLabelProteinDataset(validation_frame), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    print(
        f"Loading pretrained encoder: {args.model_name}. "
        "The first download can take several minutes.",
        flush=True,
    )
    model = ESM2MultiLabelClassifier(args.model_name, args.dropout)
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
        if hasattr(model.encoder.config, "use_cache"):
            model.encoder.config.use_cache = False
    model = model.to(device)
    print("Model loaded on GPU; configuring loss and optimizer.", flush=True)
    counts = train_frame[LABELS].sum().to_numpy(dtype=float)
    pos_weight = (len(train_frame) - counts) / counts
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates * args.warmup_ratio), total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "data": str(args.data), "output_dir": str(args.output_dir), "labels": LABELS, "positive_weights": dict(zip(LABELS, pos_weight.tolist())), "selection_metric": "validation_macro_f1"}
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    history, best_score, stale = [], -1.0, 0
    print(f"Training setup ready; pos_weight={dict(zip(LABELS, pos_weight.round(2)))}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); running = 0.0
        print(f"Starting epoch {epoch}/{args.epochs} ({len(train_loader):,} batches).", flush=True)
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", mininterval=1.0)
        for step, batch in enumerate(progress, 1):
            targets = batch["targets"].to(device)
            with amp_context(device, amp_enabled, amp_dtype):
                loss = loss_fn(model(**model_inputs(batch, device)), targets) / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            running += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
            progress.set_postfix(loss=f"{running / step:.4f}")
            if step == 1 or step % 100 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{args.epochs} — batch {step:,}/{len(train_loader):,} "
                    f"({100 * step / len(train_loader):.1f}%), running_loss={running / step:.4f}",
                    flush=True,
                )
        print(f"Epoch {epoch} training complete; running validation.", flush=True)
        metrics, predictions = evaluate(model, validation_loader, loss_fn, device, amp_enabled, amp_dtype)
        row = {"epoch": epoch, "train_loss": running / len(train_loader), **{f"validation_{k}": v for k, v in metrics.items()}}
        history.append(row); pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        print(f"Epoch {epoch}: macro_f1={metrics['macro_f1']:.4f}, peripheral_recall={metrics['peripheral_recall']:.4f}, transmembrane_recall={metrics['transmembrane_recall']:.4f}", flush=True)
        if metrics["macro_f1"] > best_score:
            best_score, stale = metrics["macro_f1"], 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "validation_metrics": metrics, "model_name": args.model_name, "dropout": args.dropout, "labels": LABELS, "threshold": 0.5}, args.output_dir / "best_checkpoint.pt")
            predictions.to_csv(args.output_dir / "best_validation_predictions.csv", index=False)
            (args.output_dir / "best_validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                print("Early stopping triggered", flush=True); break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--mixed-precision", choices=["auto", "none", "fp16", "bf16"], default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
