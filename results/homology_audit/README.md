# Cross-split homology audit

Run the audit with local BLAST+ (`makeblastdb` and `blastp`):

```bash
python scripts/audit_homology.py --overwrite
```

`summary.json` is tracked as the compact scientific result. The 799 KB
`cross_split_blast_hits.csv` query-level alignment table is reproducible and
Git-ignored.

A threshold match means that a validation/test protein has a training BLASTP
alignment at or above the stated percent identity and covering at least 80% of
the shorter sequence. The audit diagnoses the existing split; it does not
create a homology-aware split.
