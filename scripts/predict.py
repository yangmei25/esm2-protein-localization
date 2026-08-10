#!/usr/bin/env python3
"""Predict membrane versus soluble localization for one protein sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZOU")
DEFAULT_CHECKPOINT = Path("results/finetune/esm2_t6_8M_mean/best_checkpoint.pt")
MAX_RESIDUES = 1022


def normalize_sequence(raw_sequence: str) -> str:
    """Normalize and validate one ESM-compatible amino-acid sequence."""
    sequence = "".join(raw_sequence.split()).upper()
    if not sequence:
        raise ValueError("Protein sequence is empty")
    invalid = sorted(set(sequence) - VALID_AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Invalid amino-acid symbols: {invalid}")
    if len(sequence) > MAX_RESIDUES:
        raise ValueError(
            f"Sequence has {len(sequence)} residues; this model supports at most "
            f"{MAX_RESIDUES}. The sequence will not be truncated automatically."
        )
    return sequence


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

    tokenized = tokenizer(
        [sequence],
        padding=False,
        truncation=False,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    model_inputs = {
        key: tokenized[key].to(device)
        for key in ("input_ids", "attention_mask", "special_tokens_mask")
    }
    with torch.inference_mode():
        logits = model(**model_inputs)
        membrane_probability = float(
            torch.softmax(logits.float(), dim=-1)[0, 1].cpu()
        )
    predicted_label = "membrane" if membrane_probability >= threshold else "soluble"

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
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
