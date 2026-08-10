# Raw DeepLoc 1.0 data

The local input is `deeploc_data.fasta`. It is excluded from Git and must not be
redistributed from this repository.

## Provenance

- Dataset: DeepLoc 1.0 training/test protein dataset
- Original project page reported by the paper:
  <http://www.cbs.dtu.dk/services/DeepLoc/data.php>
- Current DTU DeepLoc service: <https://services.healthtech.dtu.dk/services/DeepLoc-2.1/>
- Citation: Almagro Armenteros JJ, Sønderby CK, Sønderby SK, Nielsen H, Winther O.
  *DeepLoc: prediction of protein subcellular localization using deep learning.*
  Bioinformatics. 2017;33(21):3387–3395.
  <https://doi.org/10.1093/bioinformatics/btx431>
- Source described by the paper: experimentally annotated eukaryotic proteins
  from UniProt release 2016_04, filtered and mapped to ten locations plus the
  membrane/soluble label.
- Recorded local acquisition date: 2026-07-20, based on the local file creation
  and modification timestamp. The original download event was not logged
  separately.
- Local SHA-256:
  `ea5b052662dd580d56fab3bf7ebb43bbc215feb96308d45b66a96675874c1f2b`
- Local records: 14,004 FASTA entries

## License and redistribution

No explicit dataset license was identified on the accessible source and paper
pages during the 2026-08-10 documentation review. The article's publication
license must not be assumed to license the accompanying dataset. Therefore the
raw FASTA remains Git-ignored; users must obtain it from the source and verify
their permitted use independently.

## Integrity rules

- Never manually edit the raw FASTA.
- Keep raw and processed sequence data out of Git.
- Run `scripts/prepare_data.py` to create the length-limited ESM-2 dataset.
- Run `scripts/train_classical_baselines.py` directly on the raw FASTA for the
  full-length baseline.
