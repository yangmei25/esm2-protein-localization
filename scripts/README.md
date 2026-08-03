# Pipeline scripts

Implemented pipeline scripts:

1. `prepare_data.py` prepares the DeepLoc binary dataset and fixed splits.
2. `extract_embeddings.py` caches three frozen ESM-2 representations.
3. `train_embedding_classifiers.py` compares Logistic Regression classifiers.
4. `train_finetune.py` fine-tunes ESM-2 8M with residue mean pooling and
   validation-F1 checkpoint selection; test evaluation requires an explicit
   flag or test-only invocation.

Each script will have a single responsibility and a `--help` interface.
