# Project Plan

## Project title

ESM-2 Frozen Embeddings for Protein Localization Prediction

## Minimal-version objective

Build a binary classifier that predicts whether a protein is membrane-associated
or soluble from its amino-acid sequence.

## Input and output

- Input: one amino-acid sequence
- Output: membrane probability and predicted class
- Label `0`: soluble
- Label `1`: membrane

## Model design

```text
Protein sequence
      ↓
facebook/esm2_t6_8M_UR50D (frozen)
      ↓
Mean-pooled sequence embedding
      ↓
Logistic Regression
      ↓
Membrane probability
```

ESM-2 will remain frozen in the minimal version. End-to-end fine-tuning belongs
to the intermediate version and will not be added until the frozen pipeline works.

## Data contract

The cleaned dataset must contain:

| Column | Meaning |
|---|---|
| `protein_id` | Unique protein identifier |
| `sequence` | Amino-acid sequence |
| `label` | `0` for soluble or `1` for membrane |
| `split` | `train`, `validation`, or `test` |
| `length` | Number of amino acids before tokenization |

## Evaluation

- Primary metric: F1
- Secondary metrics: accuracy, precision, recall, and ROC-AUC
- Required figure: confusion matrix
- The test set will be evaluated only after model selection is complete.

## Initial constraints

- Model: `facebook/esm2_t6_8M_UR50D`
- Maximum protein length: 1,022 amino acids
- Random seed: 42
- First split: stratified train/validation/test split
- Long-sequence and homology-leakage limitations must be documented
- No performance number will be reported unless produced by our experiment

## Definition of done

- [ ] A public dataset and its license are documented
- [ ] Raw data is preserved unchanged
- [ ] Sequences and labels are validated
- [ ] Train, validation, and test splits are reproducible
- [ ] Frozen ESM-2 embeddings are extracted and cached
- [ ] Logistic Regression is trained on training embeddings
- [ ] Validation data is used for model decisions
- [ ] Final metrics are calculated on the test set
- [ ] A confusion matrix is generated
- [ ] A single-sequence inference script works
- [ ] README contains real results and known limitations

## Later extensions—not part of the minimal version

- Amino-acid-composition baseline
- End-to-end ESM-2 fine-tuning
- Homology-aware splitting
- Multiple random seeds
- ESM-2 8M versus 35M comparison
- Phage-specific prediction task

