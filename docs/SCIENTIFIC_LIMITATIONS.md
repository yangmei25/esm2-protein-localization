# Scientific limitations and research-quality roadmap

## Status

| Item | Status | Evidence or next action |
|---|---|---|
| Dataset source and citation | Documented | `data/raw/README.md` |
| Dataset license | Unresolved | No explicit dataset license identified; raw data are not redistributed |
| Download date | Partially documented | Local file timestamp recorded; original download event was not logged |
| Exact duplicate leakage | Checked | Preprocessing rejects duplicates crossing the official test boundary |
| Cross-split homology | Audited | `scripts/audit_homology.py` and `results/homology_audit/` |
| Homology-aware holdout | Not completed | Cluster sequences first, assign whole clusters to splits, then retrain every model |
| Multiple seeds | Not completed | Fine-tuning currently reports seed 42 only; repeat complete training before estimating variance |
| Cross-validation | Not completed | Prefer grouped or cluster-aware folds rather than random folds |
| Untouched test estimate | Not available | The official test set was inspected during exploratory model comparisons |
| External validation | Not completed | Evaluate once on a separately sourced, label-compatible dataset |

## Interpretation

The current results establish that the implemented methods work well on this
particular DeepLoc-derived split. They do not yet establish generalization to
distant homologs or proteins from a different database, organism distribution,
or annotation process.

Sequence similarity between training and held-out proteins can make random
split performance optimistic. The BLASTP audit quantifies similarity in the
existing split but does not remove it. A stronger follow-up experiment should
cluster all proteins by sequence similarity before assigning clusters—not
individual proteins—to train, validation, and test sets.

Multiple random seeds are especially important for end-to-end fine-tuning.
Reporting mean and standard deviation across repeated runs would distinguish a
stable improvement from one favorable initialization. Classical models are
cheap to repeat, but repeating them alone does not quantify fine-tuning
variance.

Finally, all current test numbers must remain labeled exploratory. A new
external dataset or a newly constructed cluster-aware holdout is required for a
clean final generalization estimate.
