#!/usr/bin/env python3
"""Predict membrane versus soluble localization for one protein sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZOU")
DEFAULT_CHECKPOINT = Path("results/models/esm2_150m/best_checkpoint.pt")
MAX_RESIDUES = 1022
DEFAULT_WINDOW_OVERLAP = 128
DEFAULT_WINDOW_STRIDE = MAX_RESIDUES - DEFAULT_WINDOW_OVERLAP


def normalize_sequence(raw_sequence: str) -> str:
    """Normalize and validate one amino-acid sequence of any length."""
    sequence = "".join(raw_sequence.split()).upper()
    if not sequence:
        raise ValueError("Protein sequence is empty")
    invalid = sorted(set(sequence) - VALID_AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Invalid amino-acid symbols: {invalid}")
    return sequence


def make_sequence_windows(
    sequence: str,
    window_size: int = MAX_RESIDUES,
    stride: int = DEFAULT_WINDOW_STRIDE,
) -> list[dict]:
    """Split a sequence into overlapping windows with 1-based coordinates."""
    if window_size < 1 or stride < 1 or stride > window_size:
        raise ValueError("Require 1 <= stride <= window_size")
    if len(sequence) <= window_size:
        return [{"start": 1, "end": len(sequence), "sequence": sequence}]
    starts = list(range(0, len(sequence) - window_size + 1, stride))
    final_start = len(sequence) - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return [
        {
            "start": start + 1,
            "end": start + window_size,
            "sequence": sequence[start : start + window_size],
        }
        for start in starts
    ]


def aggregate_window_probabilities(probabilities: list[float]) -> float:
    """Apply max-pooling multiple-instance aggregation to window probabilities."""
    if not probabilities:
        raise ValueError("At least one window probability is required")
    if any(not 0 <= probability <= 1 for probability in probabilities):
        raise ValueError("Window probabilities must be between 0 and 1")
    return max(probabilities)


def read_single_fasta(path: Path) -> tuple[str, str]:
    """Read exactly one FASTA record and return its identifier and sequence."""
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    headers: list[str] = []
    sequence_lines: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                headers.append(line[1:].strip())
                if len(headers) > 1:
                    raise ValueError("FASTA input must contain exactly one protein")
            else:
                if not headers:
                    raise ValueError("FASTA sequence appears before its header")
                sequence_lines.append(line)

    if not headers:
        raise ValueError("FASTA input has no header")
    protein_id = headers[0].split()[0] if headers[0] else "protein"
    return protein_id, normalize_sequence("".join(sequence_lines))


def resolve_input(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve a direct sequence or a single-record FASTA input."""
    if args.sequence is not None:
        return args.protein_id, normalize_sequence(args.sequence)
    return read_single_fasta(args.fasta)


def predict(
    sequence: str,
    protein_id: str,
    checkpoint_path: Path,
    requested_device: str,
    threshold_override: float | None = None,
    window_size: int = MAX_RESIDUES,
    window_stride: int = DEFAULT_WINDOW_STRIDE,
    window_batch_size: int = 4,
) -> dict:
    """Load the selected checkpoint and predict one normalized sequence."""
    import torch
    from transformers import AutoTokenizer

    try:
        from scripts.train_finetune import (
            ESM2MeanPoolingClassifier,
            choose_device,
            load_model_state_compatibly,
        )
    except ModuleNotFoundError:  # Direct execution from the scripts directory.
        from train_finetune import (  # type: ignore[no-redef]
            ESM2MeanPoolingClassifier,
            choose_device,
            load_model_state_compatibly,
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run "
            "`python scripts/download_checkpoint.py` first."
        )

    device = choose_device(requested_device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_fields = {"model_state_dict", "model_name", "dropout"}
    missing_fields = required_fields - set(checkpoint)
    if missing_fields:
        raise ValueError(
            f"Checkpoint is missing required fields: {sorted(missing_fields)}"
        )

    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else float(checkpoint.get("threshold", 0.5))
    )
    if not 0 <= threshold <= 1:
        raise ValueError("Probability threshold must be between 0 and 1")

    model_name = str(checkpoint["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ESM2MeanPoolingClassifier(model_name, float(checkpoint["dropout"]))
    load_model_state_compatibly(model, checkpoint["model_state_dict"])
    model.to(device).eval()

    if window_size > MAX_RESIDUES:
        raise ValueError(f"Window size cannot exceed {MAX_RESIDUES} residues")
    if window_batch_size < 1:
        raise ValueError("Window batch size must be positive")
    windows = make_sequence_windows(sequence, window_size, window_stride)
    probabilities: list[float] = []
    with torch.inference_mode():
        for offset in range(0, len(windows), window_batch_size):
            batch = windows[offset : offset + window_batch_size]
            tokenized = tokenizer(
                [window["sequence"] for window in batch],
                padding=True,
                truncation=False,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            model_inputs = {
                key: tokenized[key].to(device)
                for key in ("input_ids", "attention_mask", "special_tokens_mask")
            }
            logits = model(**model_inputs)
            probabilities.extend(
                torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
            )
    membrane_probability = aggregate_window_probabilities(probabilities)
    predicted_label = "membrane" if membrane_probability >= threshold else "soluble"
    window_results = [
        {
            "window_index": index,
            "start": window["start"],
            "end": window["end"],
            "membrane_probability": probability,
        }
        for index, (window, probability) in enumerate(
            zip(windows, probabilities, strict=True), start=1
        )
    ]
    top_window = max(window_results, key=lambda window: window["membrane_probability"])

    return {
        "protein_id": protein_id,
        "sequence_length": len(sequence),
        "predicted_label": predicted_label,
        "membrane_probability": membrane_probability,
        "soluble_probability": 1.0 - membrane_probability,
        "threshold": threshold,
        "model_name": model_name,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "long_sequence_mode": len(sequence) > window_size,
        "aggregation": "maximum_window_probability",
        "window_size": window_size,
        "window_stride": window_stride,
        "number_of_windows": len(windows),
        "top_window": top_window,
        "windows": window_results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict membrane versus soluble localization for one protein."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sequence", help="Amino-acid sequence (whitespace is removed).")
    inputs.add_argument("--fasta", type=Path, help="Single-record FASTA file.")
    parser.add_argument(
        "--protein-id",
        default="query_protein",
        help="Identifier used with --sequence (default: query_protein).",
    )
    parser.add_argument(
        "--window-size", type=int, default=MAX_RESIDUES,
        help="Residues per long-sequence window (default: 1022).",
    )
    parser.add_argument(
        "--window-stride", type=int, default=DEFAULT_WINDOW_STRIDE,
        help="Stride between overlapping windows (default: 894; overlap: 128).",
    )
    parser.add_argument(
        "--window-batch-size", type=int, default=4,
        help="Number of windows evaluated together (default: 4).",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the checkpoint threshold (default: checkpoint value).",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protein_id, sequence = resolve_input(args)
    result = predict(
        sequence=sequence,
        protein_id=protein_id,
        checkpoint_path=args.checkpoint,
        requested_device=args.device,
        threshold_override=args.threshold,
        window_size=args.window_size,
        window_stride=args.window_stride,
        window_batch_size=args.window_batch_size,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
