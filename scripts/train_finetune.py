#!/usr/bin/env python3
"""Fine-tune ESM-2 8M for membrane-versus-soluble classification.

The model mean-pools residue embeddings while excluding special tokens and
padding, then applies dropout and a two-class linear head. Model selection uses
validation F1. The official test split is evaluated only with an explicit flag.
"""

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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"
REQUIRED_COLUMNS = {"protein_id", "sequence", "label", "split", "original_length"}


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda/mps and fail clearly when unavailable."""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data(path: Path) -> pd.DataFrame:
    """Load and validate the ESM-compatible processed dataset."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Processed CSV is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Processed CSV is empty")
    if frame["protein_id"].duplicated().any():
        raise ValueError("Protein IDs must be unique")
    if set(frame["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation, and test splits")
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1")
    if not (frame["sequence"].str.len() == frame["original_length"]).all():
        raise ValueError("Sequence length does not match original_length")
    if frame["original_length"].max() > 1022:
        raise ValueError("Fine-tuning data contains a sequence longer than 1,022 residues")
    return frame


class ProteinDataset(Dataset):
    """Protein sequences, labels, and identifiers for one split."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.ids = frame["protein_id"].astype(str).tolist()
        self.sequences = frame["sequence"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {
            "protein_id": self.ids[index],
            "sequence": self.sequences[index],
            "label": self.labels[index],
        }


def make_collate_fn(tokenizer):
    """Build a dynamic-padding collator with special-token masks."""

    def collate(examples: list[dict]) -> dict:
        tokenized = tokenizer(
            [example["sequence"] for example in examples],
            padding=True,
            truncation=False,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        tokenized["labels"] = torch.tensor(
            [example["label"] for example in examples], dtype=torch.long
        )
        tokenized["protein_ids"] = [example["protein_id"] for example in examples]
        return tokenized

    return collate


class ESM2MeanPoolingClassifier(nn.Module):
    """Trainable ESM-2 encoder with residue mean pooling and a linear head."""

    def __init__(self, model_name: str, dropout: float) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        residue_mask = attention_mask.bool() & ~special_tokens_mask.bool()
        float_mask = residue_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp_min(1)
        return self.classifier(self.dropout(pooled))


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Calculate binary metrics at the fixed 0.5 threshold."""
    predictions = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
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


def resolve_amp(device: torch.device, requested: str) -> tuple[bool, torch.dtype | None]:
    """Resolve mixed precision without enabling unsupported modes."""
    if requested == "none" or device.type != "cuda":
        return False, None
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but is unsupported by this GPU")
        return True, torch.bfloat16
    if requested == "fp16":
        return True, torch.float16
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype | None):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def move_model_inputs(batch: dict, device: torch.device) -> dict:
    return {
        key: batch[key].to(device)
        for key in ("input_ids", "attention_mask", "special_tokens_mask")
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate one split and return metrics plus protein-level predictions."""
    model.eval()
    losses: list[float] = []
    all_ids: list[str] = []
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        labels = batch["labels"].to(device)
        with autocast_context(device, amp_enabled, amp_dtype):
            logits = model(**move_model_inputs(batch, device))
            loss = loss_fn(logits, labels)
        probabilities = torch.softmax(logits.float(), dim=-1)[:, 1]
        losses.append(float(loss.detach().cpu()))
        all_ids.extend(batch["protein_ids"])
        all_labels.append(labels.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    labels_array = np.concatenate(all_labels)
    probability_array = np.concatenate(all_probabilities)
    metrics = classification_metrics(labels_array, probability_array)
    metrics["loss"] = float(np.mean(losses))
    predictions = (probability_array >= 0.5).astype(np.int64)
    prediction_frame = pd.DataFrame(
        {
            "protein_id": all_ids,
            "label": labels_array,
            "prediction": predictions,
            "membrane_probability": probability_array,
        }
    )
    return metrics, prediction_frame


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    validation_metrics: dict,
    args: argparse.Namespace,
) -> None:
    """Save a portable state-dict checkpoint and its configuration."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_metrics": validation_metrics,
            "model_name": args.model_name,
            "dropout": args.dropout,
            "pooling": "residue_mean",
            "threshold": 0.5,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    """Run fine-tuning, validation selection, and optional test evaluation."""
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("epochs, batch size, and accumulation steps must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")

    seed_everything(args.seed)
    device = choose_device(args.device)
    amp_enabled, amp_dtype = resolve_amp(device, args.mixed_precision)
    print(f"Loading dataset: {args.data}", flush=True)
    frame = load_data(args.data)
    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    validation_frame = frame[frame["split"] == "validation"].reset_index(drop=True)
    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)

    print(
        f"Dataset ready — train: {len(train_frame)}, validation: "
        f"{len(validation_frame)}, test: {len(test_frame)}",
        flush=True,
    )
    print(f"Loading pretrained model: {args.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fn = make_collate_fn(tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        ProteinDataset(train_frame),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        ProteinDataset(validation_frame),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ProteinDataset(test_frame),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ESM2MeanPoolingClassifier(args.model_name, args.dropout).to(device)
    print(f"Model ready on {device}; starting training setup", flush=True)
    label_counts = np.bincount(train_frame["label"].to_numpy(), minlength=2)
    class_weights = len(train_frame) / (2.0 * label_counts)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = args.output_dir / "best_checkpoint.pt"
    if args.test_only:
        if not best_checkpoint.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint}")
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics, test_predictions = evaluate(
            model, test_loader, loss_fn, device, amp_enabled, amp_dtype
        )
        test_predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
        (args.output_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Loaded epoch {checkpoint['epoch']} checkpoint with validation "
            f"F1={checkpoint['validation_metrics']['f1']:.4f}"
        )
        print("Exploratory test metrics:")
        print(json.dumps(test_metrics, indent=2))
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )

    history_path = args.output_dir / "history.csv"
    run_config = {
        **vars(args),
        "data": str(args.data),
        "output_dir": str(args.output_dir),
        "device_used": str(device),
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype),
        "train_proteins": len(train_frame),
        "validation_proteins": len(validation_frame),
        "test_proteins": len(test_frame),
        "class_weights": class_weights.tolist(),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Device: {device}")
    print(f"Mixed precision: {amp_enabled} ({amp_dtype})")
    print(f"Train/validation/test: {len(train_frame)}/{len(validation_frame)}/{len(test_frame)}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Optimizer updates: {total_updates} ({warmup_steps} warmup)")

    history: list[dict] = []
    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(progress, start=1):
            labels = batch["labels"].to(device)
            with autocast_context(device, amp_enabled, amp_dtype):
                logits = model(**move_model_inputs(batch, device))
                loss = loss_fn(logits, labels) / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps

            should_update = (
                step % args.gradient_accumulation_steps == 0 or step == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        validation_metrics, validation_predictions = evaluate(
            model,
            validation_loader,
            loss_fn,
            device,
            amp_enabled,
            amp_dtype,
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "learning_rate": scheduler.get_last_lr()[0],
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(
            f"Epoch {epoch}: train_loss={row['train_loss']:.4f}, "
            f"val_loss={validation_metrics['loss']:.4f}, "
            f"val_f1={validation_metrics['f1']:.4f}, "
            f"val_auc={validation_metrics['roc_auc']:.4f}"
        )

        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            epochs_without_improvement = 0
            save_checkpoint(best_checkpoint, model, epoch, validation_metrics, args)
            validation_predictions.to_csv(
                args.output_dir / "best_validation_predictions.csv", index=False
            )
            (args.output_dir / "best_validation_metrics.json").write_text(
                json.dumps(validation_metrics, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Saved new best checkpoint (validation F1={best_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print("Early stopping triggered")
                break

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Best checkpoint: epoch {checkpoint['epoch']}, "
        f"validation F1={checkpoint['validation_metrics']['f1']:.4f}"
    )

    if args.evaluate_test:
        test_metrics, test_predictions = evaluate(
            model, test_loader, loss_fn, device, amp_enabled, amp_dtype
        )
        test_predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
        (args.output_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8"
        )
        print("Exploratory test metrics:")
        print(json.dumps(test_metrics, indent=2))
    else:
        print("Test set not evaluated. Pass --evaluate-test to run exploratory testing.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune ESM-2 8M with residue mean pooling."
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/deeploc_binary.csv")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/finetune/esm2_t6_8M_mean"),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["auto", "none", "fp16", "bf16"],
        default="auto",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the previously inspected official test split after selection.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Load output-dir/best_checkpoint.pt and evaluate test without retraining.",
    )
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
