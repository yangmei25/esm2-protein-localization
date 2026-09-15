# Pipeline scripts

Implemented pipeline scripts:

1. `prepare_data.py` prepares the DeepLoc binary dataset and fixed splits.
2. `extract_embeddings.py` caches three frozen ESM-2 representations.
3. `train_embedding_classifiers.py` compares Logistic Regression classifiers.
4. `train_finetune.py` fine-tunes ESM-2 8M with residue mean pooling and
   validation-F1 checkpoint selection; test evaluation requires an explicit
   flag or test-only invocation.
5. `predict.py` loads the selected fine-tuned checkpoint and predicts one
   direct amino-acid sequence or FASTA record; proteins over 1,022 residues use
   overlapping windows and
   max-probability multiple-instance aggregation.
6. `train_classical_baselines.py` reproduces the full-length 35-feature
   hydrophobicity/composition Logistic Regression and Random Forest baselines.
7. `audit_homology.py` uses local BLASTP to quantify train-to-validation/test
   sequence similarity in the existing split.
8. `plot_confusion_matrix.py` verifies saved validation counts and regenerates
   the fine-tuned confusion matrix used in the README.
9. `download_checkpoint.py` downloads and SHA-256-verifies the published model
   checkpoint without adding generated weights to Git history.
10. `plot_validation_roc_comparison.py` compares the strongest classical
    baseline, frozen mean ESM-2, and fine-tuned ESM-2 on the same validation
    proteins.
11. `prepare_external_deeploc21.py` prepares the official DeepLoc 2.1
    membrane-type partition for binary external evaluation.
12. `filter_external_homology.py` removes external proteins homologous to V1
    training sequences using a prespecified BLASTP gate.
13. `evaluate_finetuned_external.py` evaluates a saved checkpoint on the
    filtered external cohort without training.
14. `evaluate_long_proteins.py` compares overlapping-window aggregation rules
    on labeled proteins longer than the ESM-2 context limit.
15. `prepare_subtype_aware_data.py` builds DeepLoc 2.1 train/validation/test
    partitions with membrane subtype columns and configurable sampling weights.
16. `train_multilabel_finetune.py` fine-tunes ESM-2 with independent Soluble,
    Transmembrane, Peripheral, and LipidAnchor outputs and macro-F1 selection.
17. `evaluate_multilabel_checkpoint.py` evaluates a four-label checkpoint on
    the legacy binary validation cohort or a fully labeled external cohort.
18. `build_v2_report.py` regenerates the CPU-only V2 summary tables and
    multi-seed figure from `results/metrics/v2_project_summary.json`.
19. `start_service.py` downloads an optional private S3 checkpoint and starts
    the FastAPI service with one worker so the model is loaded only once.

Each script will have a single responsibility and a `--help` interface.
