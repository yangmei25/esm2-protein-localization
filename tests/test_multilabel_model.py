from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from scripts.train_multilabel_finetune import ESM2MultiLabelClassifier, multilabel_metrics


class DummyEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden

    def forward(self, **_kwargs):
        return SimpleNamespace(last_hidden_state=self.hidden)


def test_multilabel_pooling_excludes_special_tokens():
    hidden = torch.tensor([[[99.0, 99.0], [1.0, 3.0], [3.0, 5.0], [88.0, 88.0]]])
    model = ESM2MultiLabelClassifier.__new__(ESM2MultiLabelClassifier)
    nn.Module.__init__(model)
    model.encoder = DummyEncoder(hidden)
    model.dropout = nn.Identity()
    model.classifier = nn.Linear(2, 4, bias=False)
    model.classifier.weight.data.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]))
    logits = model(
        input_ids=torch.tensor([[0, 1, 2, 3]]),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        special_tokens_mask=torch.tensor([[1, 0, 0, 1]]),
    )
    torch.testing.assert_close(logits, torch.tensor([[2.0, 4.0, 6.0, -2.0]]))


def test_multilabel_metrics_allow_overlapping_labels():
    targets = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    probabilities = np.array([[0.9, 0.1, 0.8, 0.2], [0.1, 0.9, 0.2, 0.8]])
    metrics = multilabel_metrics(targets, probabilities)
    assert metrics["exact_match_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["peripheral_recall"] == 1.0
