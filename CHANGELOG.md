# Changelog

## 1.1.0 - 2026-08-07

- Renamed the Python package from `battery_fusion` to `imfuse` so the import
  name matches the project and the `imfuse` command line. Update imports from
  `battery_fusion.<module>` to `imfuse.<module>`; the CLI is unchanged.
- Curated a lazy top-level API: feature builders, encoders and fusion models,
  dataset/collate utilities, the training loop, and the modality-dropout
  probe now import directly from `imfuse`.
- Added `examples/quickstart.py`, a CPU-only, in-memory end-to-end example
  (features → intermediate fusion training → modality-dropout probe) that
  does not require the publication data contract.
- Added a GitHub Actions test workflow and README badges.
- Removed superseded internal code (~630 lines): the unreleased error-bar
  variants of the deletion-curve and permutation plotting pipelines, an
  orphaned faithfulness plotting helper, and a duplicate split-manifest
  loader; consolidated duplicated device-transfer helpers in the
  modality-dropout module onto `tri_adapter`/`move_graph_tuple`.
- Updated citation metadata for the accepted article in *Cell Reports
  Physical Science* and recorded the Zenodo software DOI
  (10.5281/zenodo.21440948).

## 1.0.0 - 2026-07-19

- Released the validated raw-target workflows for average voltage and
  volumetric capacity.
- Retained deterministic, non-overlapping 80/10/10 split assignments for five
  shared seeds (`0`-`4`) over 8,088 unique model-ready discharge IDs, derived
  from the 10,123-row source table.
- Included composition, RDF, CGCNN-style structure, early/intermediate/late
  fusion, RF/XGBoost, and pretrained ALIGNN+RF workflows.
- Included modality-dropout, chemistry-aware OOD, subgroup, attribution,
  interaction, and deletion-faithfulness analyses.
- Added a single public `imfuse` command surface, tracked-data checksums, and a
  release preflight check.
