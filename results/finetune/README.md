# Fine-tuning artifacts

This directory contains lightweight evidence from end-to-end ESM-2
fine-tuning. Each experiment has its own subdirectory named for the model and
pooling strategy.

## Current experiment

`esm2_t6_8M_mean/` contains the five-epoch ESM-2 8M mean-pooling run selected by
validation F1:

| File | Purpose | Tracked in Git? |
|---|---|---|
| `run_config.json` | Exact training arguments and runtime information | Yes |
| `history.csv` | Per-epoch training and validation measurements | Yes |
| `best_validation_metrics.json` | Metrics from the selected checkpoint | Yes |
| `best_validation_predictions.csv` | Per-protein validation predictions | No |
| `test_metrics.json` | Exploratory test metrics from the selected checkpoint | Yes |
| `test_predictions.csv` | Per-protein exploratory test predictions | No |
| `best_checkpoint.pt` | Model and optimizer checkpoint | No |

The checkpoint is excluded by `.gitignore` because trained weights are a large
generated artifact. The persistent copy for this run is stored in Google Drive
under:

```text
MyDrive/esm2-protein-localization/finetune-results/esm2_t6_8M_mean/
```

The public copy is prepared for distribution as the `model-v1.0` GitHub Release
asset. After that release is published, obtain and integrity-check it with:

```bash
python scripts/download_checkpoint.py
```

Expected SHA-256:
`698fca86ff8b973a026529b645b5d4536da2a6c30c96d797da16bc047a158fbd`.

Per-protein prediction tables are also excluded because the source dataset's
redistribution license is unresolved. Aggregate metrics, configuration, and
training history remain public.

To evaluate the local checkpoint without retraining:

```bash
python scripts/train_finetune.py \
  --data data/processed/deeploc_binary.csv \
  --output-dir results/finetune/esm2_t6_8M_mean \
  --device auto \
  --test-only
```

The official test split was previously inspected during exploratory analysis,
so test-only results should be described as exploratory.

## Exploratory test result

The epoch-five checkpoint was evaluated once across all 1,571 proteins in the
fixed test split at the prespecified probability threshold of 0.5.

| Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9160 | 0.9211 | 0.8769 | 0.8985 | 0.9606 | 855 | 50 | 82 | 584 |
