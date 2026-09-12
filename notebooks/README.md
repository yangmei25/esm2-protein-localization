# Notebooks

These polished notebooks are intended for mentor and public review. Command-line
scripts remain the authoritative implementation.

1. `01_project_walkthrough.ipynb` summarizes the problem, data, methods,
   baseline results, frozen ESM-2 results, limitations, and next experiment.
2. `02_colab_finetuning.ipynb` clones the GitHub repository, uploads the
   Git-ignored processed CSV, calls `scripts/train_finetune.py`, plots validation
   history, and downloads the selected checkpoint and result files.
3. `03_v2_model_scaling.ipynb` compares frozen and fine-tuned ESM-2 8M, 35M,
   and 150M while saving large artifacts persistently in Google Drive.

Detailed learning and exploratory notebooks are kept locally in the ignored
`notebooks_for_me/` directory.
