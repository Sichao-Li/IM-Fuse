# Changelog

## 1.0.0 - 2026-07-19

- Released the validated raw-target workflows for average voltage and
  volumetric capacity.
- Retained deterministic, non-overlapping 80/10/10 split assignments for five
  shared seeds (`0`-`4`) over 8,088 unique model-ready discharge IDs, derived
  from the 10,114-row cleaned source table.
- Included composition, RDF, CGCNN-style structure, early/intermediate/late
  fusion, RF/XGBoost, and pretrained ALIGNN+RF workflows.
- Included modality-dropout, chemistry-aware OOD, subgroup, attribution,
  interaction, and deletion-faithfulness analyses.
- Added a single public `imfuse` command surface, tracked-data checksums, and a
  release preflight check.
