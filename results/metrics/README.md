# Metrics

Machine-readable evaluation outputs are organized by experiment:

- `embedding_classifiers/` contains frozen ESM-2 validation comparisons.
- `classical_baselines/` contains reproducible full-length hydrophobicity and
  composition baseline results.
- `validation_roc_comparison.json` records the shared-cohort ROC-AUC comparison
  used by the main README figure.

The project test split was inspected during exploratory analysis, so saved test
results are labeled exploratory rather than treated as untouched final evidence.
