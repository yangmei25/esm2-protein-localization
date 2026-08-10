# ESM-2 Protein Localization

This project predicts whether a protein is **membrane-associated** or
**soluble** from its amino-acid sequence. It compares an interpretable
hydrophobicity baseline with two uses of the small
[`facebook/esm2_t6_8M_UR50D`](https://huggingface.co/facebook/esm2_t6_8M_UR50D)
protein language model:

1. frozen ESM-2 embeddings followed by Logistic Regression; and
2. end-to-end fine-tuning of ESM-2 with a binary classification head.

The 8M-parameter model was chosen deliberately: it is practical on a single
consumer GPU or Google Colab while still providing learned protein-sequence
representations.

## Project status

The data pipeline, three frozen-embedding experiments, biological baselines,
and validation-stage fine-tuning are complete. The best current validation
result is the fine-tuned ESM-2 model with **0.900 F1** and **0.969 ROC-AUC**.

Remaining work includes packaging single-sequence inference, adding automated
tests, documenting dataset provenance more completely, and evaluating with a
homology-aware split. Fine-tuned test evaluation has intentionally not been
reported yet.

## Pipeline

```text
DeepLoc protein sequences
          |
          v
Cleaning, deduplication, binary labels, fixed splits
          |
          +------------------------------+
          |                              |
          v                              v
Hydrophobicity/composition        ESM-2 8M (<=1,022 residues)
features                          |
          |                       +-----------------------+
          v                       |                       |
Logistic Regression /             v                       v
Random Forest              frozen embeddings       end-to-end fine-tuning
                           (first, mean, max)              |
                                  |                       |
                                  v                       v
                           Logistic Regression       classification head
```

The frozen workflow does **not** update ESM-2. It extracts one 320-dimensional
vector per protein using first-token, mean, or max pooling, then trains a
separate Logistic Regression classifier for each representation. The
fine-tuning workflow updates the ESM-2 weights and classification head jointly.

## Dataset and splits

The source data are binary membrane-versus-soluble records derived from
DeepLoc. Raw and processed data are not committed to GitHub.

| Stage | Proteins |
|---|---:|
| Initial records | 14,004 |
| ESM-compatible binary dataset | 7,890 |
| Train | 5,055 |
| Validation | 1,264 |
| Test | 1,571 |

The ESM-compatible dataset excludes duplicates and 737 proteins longer than the
model limit of 1,022 residues; sequences are not truncated. The independent
hydrophobicity baseline uses all 8,627 eligible binary proteins because it has
no transformer length limit.

The validation set is used for model comparison and checkpoint selection. The
test split was previously inspected during exploratory frozen-model analysis,
so any further result on it must be described as exploratory rather than a
fully untouched final estimate.

## Results

### Validation comparison

| Method | Representation | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Hydrophobicity + Logistic Regression | 35 handcrafted features | 0.815 | — | — | 0.768 | 0.871 |
| Hydrophobicity + Random Forest | 35 handcrafted features | 0.847 | — | — | 0.792 | 0.916 |
| Frozen ESM-2 + Logistic Regression | First token | 0.851 | 0.820 | 0.823 | 0.821 | 0.935 |
| Frozen ESM-2 + Logistic Regression | Max pooling | 0.834 | 0.805 | 0.792 | 0.798 | 0.911 |
| Frozen ESM-2 + Logistic Regression | Mean pooling | 0.892 | 0.882 | 0.853 | 0.867 | 0.948 |
| **Fine-tuned ESM-2** | **Mean pooling** | **0.922** | **0.959** | **0.848** | **0.900** | **0.969** |

Mean pooling was selected using validation F1 among the frozen representations.
Fine-tuning improved validation F1 by about **3.25 percentage points** and
ROC-AUC by about **2.08 percentage points** over frozen mean embeddings. Its
validation confusion counts were TN = 720, FP = 19, FN = 80, and TP = 445.

The Random Forest improvement over the hydrophobicity Logistic Regression was
statistically detectable on the paired validation predictions (McNemar
`p = 0.00017`). This comparison is a useful biological baseline, but it does not
replace evaluation under homology-controlled splits.

## Repository structure

```text
.
├── configs/                    Experiment configuration
├── data/                       Local raw and processed data (Git-ignored)
├── notebooks/
│   ├── 01_project_walkthrough.ipynb
│   └── 02_colab_finetuning.ipynb
├── scripts/
│   ├── prepare_data.py
│   ├── extract_embeddings.py
│   ├── train_embedding_classifiers.py
│   └── train_finetune.py
├── src/esm2_localization/      Reusable package code
├── results/                    Metrics, figures, and local model artifacts
└── tests/                      Test suite scaffold
```

The public notebooks provide a high-level walkthrough and a Colab fine-tuning
workflow. Detailed learning notebooks are kept locally in the Git-ignored
`notebooks_for_me/` directory. Command-line scripts are the authoritative
implementation.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/yangmei25/esm2-protein-localization.git
cd esm2-protein-localization
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the original DeepLoc FASTA at the path described in
[`data/raw/README.md`](data/raw/README.md). Large data, cached embeddings, and
model checkpoints should remain local and should not be committed to GitHub.

## Reproduce the frozen-embedding workflow

```bash
python scripts/prepare_data.py
python scripts/extract_embeddings.py --device auto
python scripts/train_embedding_classifiers.py
```

Use `python <script> --help` to see paths and optional settings. Embedding
extraction can be smoke-tested first with `--limit 100`; replacing an existing
cache requires the explicit `--overwrite` flag.

## Fine-tune in Google Colab

Open [`notebooks/02_colab_finetuning.ipynb`](notebooks/02_colab_finetuning.ipynb)
in Colab and run it from top to bottom. The notebook:

- clones this repository from GitHub;
- installs the required packages;
- reads the Git-ignored processed CSV supplied by the user;
- trains ESM-2 8M on a GPU; and
- stores checkpoints and metrics in Google Drive so they survive a Colab
  runtime reset.

The equivalent command-line entry point is:

```bash
python scripts/train_finetune.py \
  --data data/processed/deeploc_binary.csv \
  --output-dir results/finetune/esm2_t6_8M_mean \
  --device auto
```

Test evaluation requires an explicit choice. After training, an exploratory
test-only run can be made with:

```bash
python scripts/train_finetune.py \
  --data data/processed/deeploc_binary.csv \
  --output-dir results/finetune/esm2_t6_8M_mean \
  --device auto \
  --test-only
```

## Limitations

- Random train/validation/test splits may place homologous proteins in multiple
  splits and inflate performance.
- Proteins longer than 1,022 residues are excluded from ESM-2 experiments.
- Dataset labels and provenance require clearer documentation before treating
  the model as a scientific benchmark.
- The official test split has already been used for exploratory analysis.
- Current metrics describe this dataset only; they do not establish reliability
  on proteins from a different organism, database, or experimental protocol.
- Predictions are computational hypotheses and do not replace experimental
  localization evidence.

## Next steps

1. Add a documented `predict.py` interface for new protein sequences.
2. Add unit and smoke tests for preprocessing, pooling, and inference.
3. Make the handcrafted-feature baselines reproducible outside personal
   notebooks.
4. Pin a verified dependency environment.
5. Create similarity-clustered, homology-aware data splits and rerun all models.
6. Evaluate the selected model on a genuinely external dataset.
