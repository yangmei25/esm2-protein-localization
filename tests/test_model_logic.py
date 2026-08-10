from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from scripts.extract_embeddings import extract_batch_representations
from scripts.train_embedding_classifiers import classification_metrics as frozen_metrics
from scripts.train_finetune import (
    ESM2MeanPoolingClassifier,
    classification_metrics as finetuned_metrics,
)


class DummyEncoder(nn.Module):
    def __init__(self, hidden: torch.Tensor) -> None:
        super().__init__()
        self.hidden = hidden

    def forward(self, **_kwargs):
        return SimpleNamespace(last_hidden_state=self.hidden)


class ModelLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        # Token order: beginning token, three residues, ending token.
        self.hidden = torch.tensor(
            [[[100.0, 100.0], [1.0, 5.0], [3.0, 2.0], [2.0, 7.0], [200.0, 200.0]]]
        )
        self.token_batch = {
            "input_ids": torch.tensor([[0, 1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "special_tokens_mask": torch.tensor([[1, 0, 0, 0, 1]]),
        }

    def test_frozen_pooling_excludes_special_tokens(self) -> None:
        first, mean, maximum = extract_batch_representations(
            DummyEncoder(self.hidden), self.token_batch.copy(), torch.device("cpu")
        )
        np.testing.assert_allclose(first, [[100.0, 100.0]])
        np.testing.assert_allclose(mean, [[2.0, 14.0 / 3.0]])
        np.testing.assert_allclose(maximum, [[3.0, 7.0]])

    def test_frozen_pooling_rejects_input_without_residues(self) -> None:
        batch = {
            "input_ids": torch.tensor([[0, 4]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "special_tokens_mask": torch.tensor([[1, 1]]),
        }
        hidden = torch.zeros((1, 2, 2))
        with self.assertRaisesRegex(ValueError, "no residue tokens"):
            extract_batch_representations(
                DummyEncoder(hidden), batch, torch.device("cpu")
            )

    def test_finetuned_forward_mean_pools_only_residues(self) -> None:
        model = ESM2MeanPoolingClassifier.__new__(ESM2MeanPoolingClassifier)
        nn.Module.__init__(model)
        model.encoder = DummyEncoder(self.hidden)
        model.dropout = nn.Identity()
        model.classifier = nn.Identity()

        logits = model(
            input_ids=self.token_batch["input_ids"],
            attention_mask=self.token_batch["attention_mask"],
            special_tokens_mask=self.token_batch["special_tokens_mask"],
        )
        torch.testing.assert_close(logits, torch.tensor([[2.0, 14.0 / 3.0]]))

    def test_metric_functions_use_probability_threshold_point_five(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.7, 0.8, 0.4])

        frozen = frozen_metrics(labels, probabilities)
        finetuned = finetuned_metrics(labels, probabilities)
        self.assertAlmostEqual(frozen["accuracy"], 0.5)
        self.assertAlmostEqual(frozen["precision"], 0.5)
        self.assertAlmostEqual(frozen["recall"], 0.5)
        self.assertAlmostEqual(frozen["f1"], 0.5)
        self.assertAlmostEqual(frozen["roc_auc"], 0.75)
        self.assertEqual(
            {key: finetuned[key] for key in ("tn", "fp", "fn", "tp")},
            {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
        )


if __name__ == "__main__":
    unittest.main()
