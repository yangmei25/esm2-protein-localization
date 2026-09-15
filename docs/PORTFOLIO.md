# Portfolio summary

## ESM-2 protein localization: research to deployment

Built an end-to-end protein localization system that progressed from biological
baselines to a locally validated, deployment-ready inference service. A public
cloud endpoint is intentionally not maintained.

### Problem

Predict whether a protein is membrane-associated or soluble directly from its
amino-acid sequence, then investigate why performance changes across model size,
external cohorts, membrane subtypes, and long proteins.

### Technical work

- Compared hydrophobicity baselines, frozen ESM-2 representations, and
  end-to-end fine-tuning.
- Scaled ESM-2 from 8M to 35M and 150M parameters and confirmed model selection
  with three random seeds.
- Added homology-filtered external evaluation to reduce optimistic conclusions
  from random splits.
- Diagnosed poor Peripheral-protein recall and trained a subtype-aware model,
  while measuring the trade-off against the original binary task.
- Validated overlapping-window inference for proteins longer than the native
  1,022-residue context.
- Refactored notebook experiments into reusable Python modules and CLI scripts.
- Built a typed FastAPI service, responsive web demo, API contract tests,
  Docker image, GitHub Actions CI/CD, and AWS ECS deployment configuration.

### Selected outcomes

- Fine-tuned ESM-2 150M validation F1: **0.9363 ± 0.0040**.
- Fine-tuned ESM-2 150M validation ROC-AUC: **0.9819 ± 0.0016**.
- Exploratory test F1: **0.9220 ± 0.0020**.
- External Peripheral recall improved from **0.3254 to 0.5494**, while remaining
  an explicitly documented unsolved weakness.
- Long-protein window inference F1: **0.8425 ± 0.0157** across three seeds.

### Engineering decisions

- Python modules and scripts are authoritative; notebooks are local learning
  and review tools.
- The API loads the model once and serializes access to the shared model.
- Model weights are not committed or baked into the public container.
- CI tests the HTTP contract with a fake predictor, avoiding checkpoint and GPU
  requirements.
- AWS deployment uses private S3 model storage, ECR images tagged by commit SHA,
  GitHub OIDC, and ECS task-definition revisions.

### Limitations stated transparently

The subtype-aware result is a single-seed, post-hoc pilot; the official test set
was previously inspected; window aggregation is not residue-level topology
prediction; and the external result demonstrates a meaningful domain shift.
The service provides research predictions, not experimental evidence.
