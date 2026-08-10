# Pipeline scripts

Implemented pipeline scripts:

1. `prepare_data.py` prepares the DeepLoc binary dataset and fixed splits.
2. `extract_embeddings.py` caches three frozen ESM-2 representations.
3. `train_embedding_classifiers.py` compares Logistic Regression classifiers.
4. `train_finetune.py` fine-tunes ESM-2 8M with residue mean pooling and
   validation-F1 checkpoint selection; test evaluation requires an explicit
   flag or test-only invocation.
5. `predict.py` loads the selected fine-tuned checkpoint and predicts one
   direct amino-acid sequence or one FASTA record.
6. `train_classical_baselines.py` reproduces the full-length 35-feature
   hydrophobicity/composition Logistic Regression and Random Forest baselines.
7. `audit_homology.py` uses local BLASTP to quantify train-to-validation/test
   sequence similarity in the existing split.
8. `plot_confusion_matrix.py` verifies saved validation counts and regenerates
   the fine-tuned confusion matrix used in the README.
9. `download_checkpoint.py` downloads and SHA-256-verifies the published model
   checkpoint without adding generated weights to Git history.

Each script will have a single responsibility and a `--help` interface.
