#!/usr/bin/env python3
"""Extract three frozen ESM-2 representations in one dataset pass.

The output NPZ contains first-token, residue-mean, and residue-max embeddings,
plus the protein IDs, labels, splits, and original sequence lengths needed by
the downstream Logistic Regression experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"
REQUIRED_COLUMNS = {"protein_id", "sequence", "label", "split", "original_length"}


class ProteinDataset(Dataset):
    """A minimal dataset that preserves CSV row order."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.sequences = frame["sequence"].tolist()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> str:
        return self.sequences[index]


def file_sha256(path: Path) -> str:
    """Calculate a file checksum without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda/mps and fail clearly when unavailable."""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_processed_data(path: Path, limit: int | None) -> pd.DataFrame:
    """Load and validate the model-ready DeepLoc table."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Processed CSV is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Processed CSV is empty")
    if frame["protein_id"].duplicated().any():
        raise ValueError("Processed CSV contains duplicate protein IDs")
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1")
    if not set(frame["split"].unique()).issubset({"train", "validation", "test"}):
        raise ValueError("Unexpected split value")
    if not (frame["sequence"].str.len() == frame["original_length"]).all():
        raise ValueError("At least one sequence is incomplete or altered")
    if frame["sequence"].str.len().max() > 1022:
        raise ValueError("At least one sequence exceeds the 1,022-residue limit")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer")
        frame = frame.head(limit).copy()
    return frame.reset_index(drop=True)


def make_collate_fn(tokenizer):
    """Create a dynamic-padding collator that also returns special-token masks."""

    def collate(sequences: list[str]):
        return tokenizer(
            sequences,
            padding=True,
            truncation=False,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

    return collate


def extract_batch_representations(model, token_batch, device: torch.device):
    """Run one forward pass and derive all three sequence representations."""
    attention_mask = token_batch["attention_mask"].bool().to(device)
    special_tokens_mask = token_batch.pop("special_tokens_mask").bool().to(device)
    model_inputs = {key: value.to(device) for key, value in token_batch.items()}

    with torch.inference_mode():
        token_embeddings = model(**model_inputs).last_hidden_state

    residue_mask = attention_mask & ~special_tokens_mask
    residue_counts = residue_mask.sum(dim=1)
    if torch.any(residue_counts == 0):
        raise ValueError("A tokenized protein contains no residue tokens")

    first_token = token_embeddings[:, 0, :]
    float_mask = residue_mask.unsqueeze(-1).to(token_embeddings.dtype)
    residue_mean = (token_embeddings * float_mask).sum(dim=1) / float_mask.sum(dim=1)
    masked = token_embeddings.masked_fill(
        ~residue_mask.unsqueeze(-1), torch.finfo(token_embeddings.dtype).min
    )
    residue_max = masked.max(dim=1).values

    return tuple(
        representation.detach().float().cpu().numpy()
        for representation in (first_token, residue_mean, residue_max)
    )


def extract_embeddings(
    frame: pd.DataFrame,
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract all representations while preserving input row order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

  
  #load the pretrained ESM-2 model and tokenizer from Hugging Face, ensuring that the model is in evaluation mode and does not require gradients. Set up a DataLoader to process the protein sequences in batches, and iterate through the batches to extract the first-token, residue-mean, and residue-max embeddings. Finally, concatenate the embeddings and validate their shapes and values before returning them along with the hidden size of the model.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ESM-2 checkpoints do not train Hugging Face's optional dense pooler.
    # We use last_hidden_state directly, so disable that unused random layer.
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    model.eval()
    model.requires_grad_(False)
    model.to(device)

    loader = DataLoader(
        ProteinDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
    )
    first_batches: list[np.ndarray] = []
    mean_batches: list[np.ndarray] = []
    max_batches: list[np.ndarray] = []

    for token_batch in tqdm(loader, desc="Extracting ESM-2 embeddings"):
        first_token, residue_mean, residue_max = extract_batch_representations(
            model, token_batch, device
        )
        first_batches.append(first_token)
        mean_batches.append(residue_mean)
        max_batches.append(residue_max)

    first_embeddings = np.concatenate(first_batches).astype(np.float32, copy=False)
    mean_embeddings = np.concatenate(mean_batches).astype(np.float32, copy=False)
    max_embeddings = np.concatenate(max_batches).astype(np.float32, copy=False)
    hidden_size = int(model.config.hidden_size)

    expected_shape = (len(frame), hidden_size)
    for name, array in {
        "first_token": first_embeddings,
        "mean": mean_embeddings,
        "max": max_embeddings,
    }.items():
        if array.shape != expected_shape:
            raise AssertionError(f"{name} has shape {array.shape}, expected {expected_shape}")
        if not np.isfinite(array).all():
            raise AssertionError(f"{name} contains a non-finite value")

    return first_embeddings, mean_embeddings, max_embeddings, hidden_size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache three frozen ESM-2 representations for every protein."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/deeploc_binary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/embeddings/esm2_t6_8M_deeploc.npz"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("results/embeddings/esm2_t6_8M_deeploc.json"),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Extract only the first N rows for a smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing output files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Processed CSV not found: {args.data}")
    if not args.overwrite and (args.output.exists() or args.metadata.exists()):
        raise FileExistsError("Output already exists; pass --overwrite to replace it")

    frame = load_processed_data(args.data, args.limit)
    device = choose_device(args.device)
    print(f"Proteins: {len(frame):,}")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")

    first_embeddings, mean_embeddings, max_embeddings, hidden_size = extract_embeddings(
        frame=frame,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        protein_id=frame["protein_id"].to_numpy(dtype=str),
        label=frame["label"].to_numpy(dtype=np.int64),
        split=frame["split"].to_numpy(dtype=str),
        original_length=frame["original_length"].to_numpy(dtype=np.int64),
        first_token=first_embeddings,
        mean=mean_embeddings,
        max=max_embeddings,
    )
    metadata = {
        "model_name": args.model_name,
        "source_csv": str(args.data),
        "source_csv_sha256": file_sha256(args.data),
        "number_of_proteins": len(frame),
        "hidden_size": hidden_size,
        "embedding_dtype": "float32",
        "representations": ["first_token", "mean", "max"],
        "special_tokens_in_residue_pooling": False,
        "batch_size": args.batch_size,
        "device_used": str(device),
        "limit": args.limit,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved embeddings to {args.output}")
    print(f"Saved metadata to {args.metadata}")


if __name__ == "__main__":
    main()
