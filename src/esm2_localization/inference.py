"""Reusable single-load inference engine for protein localization."""

from __future__ import annotations

from pathlib import Path
from threading import Lock


class ProteinPredictor:
    """Load one checkpoint once and reuse it safely across API requests."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        threshold: float | None = None,
    ) -> None:
        import torch
        from transformers import AutoTokenizer

        from scripts.predict import MAX_RESIDUES
        from scripts.train_finetune import (
            ESM2MeanPoolingClassifier,
            choose_device,
            load_model_state_compatibly,
        )

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        self.device = choose_device(device)
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        required = {"model_state_dict", "model_name", "dropout"}
        if missing := required - set(checkpoint):
            raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
        self.threshold = (
            float(threshold)
            if threshold is not None
            else float(checkpoint.get("threshold", 0.5))
        )
        if not 0 <= self.threshold <= 1:
            raise ValueError("Probability threshold must be between 0 and 1")
        self.model_name = str(checkpoint["model_name"])
        self.checkpoint_epoch = checkpoint.get("epoch")
        self.max_residues = MAX_RESIDUES
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = ESM2MeanPoolingClassifier(
            self.model_name, float(checkpoint["dropout"])
        )
        load_model_state_compatibly(self.model, checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        self._torch = torch
        self._lock = Lock()

    def predict(
        self,
        sequence: str,
        protein_id: str = "query_protein",
        window_size: int = 1022,
        window_stride: int = 894,
        window_batch_size: int = 4,
    ) -> dict:
        """Predict one normalized sequence with validated long-sequence windows."""
        from scripts.predict import (
            aggregate_window_probabilities,
            make_sequence_windows,
            normalize_sequence,
        )

        sequence = normalize_sequence(sequence)
        if window_size > self.max_residues:
            raise ValueError(f"Window size cannot exceed {self.max_residues} residues")
        if window_batch_size < 1:
            raise ValueError("Window batch size must be positive")
        windows = make_sequence_windows(sequence, window_size, window_stride)
        probabilities: list[float] = []
        with self._lock, self._torch.inference_mode():
            for offset in range(0, len(windows), window_batch_size):
                batch = windows[offset : offset + window_batch_size]
                tokenized = self.tokenizer(
                    [window["sequence"] for window in batch],
                    padding=True,
                    truncation=False,
                    return_special_tokens_mask=True,
                    return_tensors="pt",
                )
                inputs = {
                    key: tokenized[key].to(self.device)
                    for key in ("input_ids", "attention_mask", "special_tokens_mask")
                }
                logits = self.model(**inputs)
                probabilities.extend(
                    self._torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
                )
        membrane_probability = aggregate_window_probabilities(probabilities)
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
        top_window = max(
            window_results, key=lambda item: item["membrane_probability"]
        )
        return {
            "protein_id": protein_id,
            "sequence_length": len(sequence),
            "predicted_label": (
                "membrane" if membrane_probability >= self.threshold else "soluble"
            ),
            "membrane_probability": membrane_probability,
            "soluble_probability": 1.0 - membrane_probability,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "checkpoint_epoch": self.checkpoint_epoch,
            "device": str(self.device),
            "long_sequence_mode": len(sequence) > window_size,
            "aggregation": "maximum_window_probability",
            "window_size": window_size,
            "window_stride": window_stride,
            "number_of_windows": len(windows),
            "top_window": top_window,
            "windows": window_results,
        }
