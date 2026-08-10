# Project plan and completion status

## Objective

Predict whether one amino-acid sequence is membrane-associated (`1`) or soluble
(`0`) and return a membrane probability plus predicted class.

## Implemented model tracks

1. Thirty-five full-length composition and hydrophobicity features with
   Logistic Regression and Random Forest.
2. Frozen `facebook/esm2_t6_8M_UR50D` first-token, residue-mean, and residue-max
   embeddings with Logistic Regression.
3. End-to-end fine-tuning of ESM-2 8M with residue mean pooling and a binary
   classification head.

The ESM workflows exclude sequences longer than 1,022 residues without
truncation. The classical baseline retains full-length sequences.

## Data contract

| Column | Meaning |
|---|---|
| `protein_id` | Unique protein identifier |
| `sequence` | Amino-acid sequence |
| `label` | `0` for soluble or `1` for membrane |
| `split` | `train`, `validation`, or `test` |
| `original_length` | Residues before tokenization |

## Evaluation policy

- Primary selection metric: validation F1
- Secondary metrics: accuracy, precision, recall, and ROC-AUC
- Probability threshold: 0.5
- The official test set was inspected during exploratory analysis; all current
  test results must remain labeled exploratory.
- Saved predictions and machine-readable metrics are the source of reported
  numbers.

## Completed

- [x] Dataset source and citation documented
- [x] Raw data preserved locally and excluded from Git
- [x] Sequence, label, duplicate, and split validation
- [x] Reproducible fixed train/validation/test splits
- [x] Full-length biological-feature baselines
- [x] Three frozen ESM-2 representations and classifiers
- [x] End-to-end ESM-2 8M fine-tuning
- [x] Validation and exploratory test evaluation
- [x] Single-sequence command-line inference
- [x] Public results table and ROC figure
- [x] Pinned local and Colab dependency specifications
- [x] Offline automated test suite
- [x] BLASTP cross-split homology audit
- [x] Explicit disclosure that no dataset redistribution license was identified

## Remaining scientific work

- [ ] Confirm dataset redistribution terms with the data provider
- [ ] Repeat complete fine-tuning with multiple random seeds
- [ ] Construct cluster-based homology-aware train/validation/test splits
- [ ] Rerun every baseline and ESM model on the homology-aware protocol
- [ ] Evaluate the selected model once on a separately sourced external dataset
- [ ] Report uncertainty intervals across independent runs or grouped folds

These remaining experiments would strengthen research claims but are beyond the
mentor's minimum deliverable of a clean repository, documented results, one
figure, and runnable single-sequence inference.
