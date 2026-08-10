# Tests

Run the automated checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

The offline suite covers:

- single-protein sequence and FASTA input validation;
- DeepLoc FASTA parsing and duplicate leakage safeguards;
- first-token, residue-mean, and residue-max pooling behavior;
- residue-only mean pooling in the fine-tuned classifier; and
- metric calculations and the fixed 0.5 decision threshold.
- the 35-feature classical baseline calculations and hydrophobicity windows.
- homology-audit identity/coverage threshold summaries.

The suite does not download ESM-2 or load the 28 MB checkpoint. The documented
`scripts/predict.py` command is the separate end-to-end inference smoke test.
