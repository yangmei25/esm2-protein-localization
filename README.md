# ESM-2 Protein Localization Prediction

This beginner-focused project will use frozen representations from the
8-million-parameter ESM-2 protein language model to predict whether a protein is
membrane-associated or soluble.

## Current status

**Milestone 1: project scaffold and task definition**

No dataset has been downloaded and no model has been trained. Results will be
added only after running a reproducible experiment.

## Minimal pipeline

```text
Protein sequence
      ↓
Frozen ESM-2 (8M parameters)
      ↓
Sequence embedding
      ↓
Logistic Regression
      ↓
Membrane or soluble
```

The minimal version uses ESM-2 as a feature extractor. It does not update the
ESM-2 parameters. This keeps training fast and makes each component easier to
understand and debug.

## Repository structure

```text
.
├── PROJECT_PLAN.md       Fixed scope and definition of done
├── requirements.txt      Python dependencies
├── configs/              Reproducible experiment settings
├── data/
│   ├── raw/              Original data, never manually edited
│   └── processed/        Clean model-ready tables
├── notebooks/            Data exploration only
├── scripts/              Command-line pipeline entry points
├── src/                  Reusable Python modules
├── results/
│   ├── figures/          Evaluation plots
│   ├── metrics/          Machine-readable metrics
│   └── models/           Saved trained classifier
└── tests/                Automated checks
```

## Planned milestones

1. Select and document a public membrane-versus-soluble dataset.
2. Clean sequences and inspect labels, duplicates, and sequence lengths.
3. Create reproducible train, validation, and test splits.
4. Load ESM-2 and understand its tokenized input and output shapes.
5. Extract and cache mean-pooled sequence embeddings.
6. Train Logistic Regression on the training embeddings.
7. Evaluate F1, accuracy, precision, recall, and ROC-AUC.
8. Generate a confusion matrix and inspect prediction errors.
9. Build a command-line tool for predicting one sequence.
10. Add real results, limitations, and reproduction instructions here.

## Environment—not installed yet

The planned environment uses Python 3.10 or 3.11. When we reach the environment
milestone, installation will use the dependencies in `requirements.txt`.

## Results

| Model | Test N | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Frozen ESM-2 + Logistic Regression | TBD | TBD | TBD | TBD |

`TBD` values will be replaced only with results produced by this repository.

## Known scientific limitations to investigate

- Proteins longer than 1,022 amino acids require a documented handling strategy.
- Similar or homologous proteins across data splits may inflate test performance.
- Dataset labels may be incomplete, predicted, or experimentally derived; provenance matters.
- Computational localization predictions do not replace experimental validation.

