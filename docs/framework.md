# Framework Overview

IM-Fuse separates the project into reusable layers.

## Data

- `battery_fusion.data`: label normalization, split creation, preprocessing.
- `battery_fusion.features`: composition, RDF, and structure feature builders.
- `battery_fusion.fusion.datasets`: aligned multimodal datasets.

## Models

- `battery_fusion.models`: single-modality MLP, RDF sequence encoder, and graph
  baseline components.
- `battery_fusion.fusion`: early, intermediate, late, dual, and tri-modal fusion
  model definitions.
- `battery_fusion.training`: shared metrics, target transforms, train/evaluate
  loops, and prediction export.

## Experiments

- `battery_fusion.experiments.publication`: random-split publication matrix.
- `battery_fusion.experiments.modality_dropout`: inference-time missing-modality robustness.
- `battery_fusion.experiments.anion_holdout`: chemistry-aware OOD holdout.
- `battery_fusion.experiments.subgroups`: anion-family and working-ion audits.
- `battery_fusion.experiments.classical_baselines`: RF/XGBoost baselines.
- `battery_fusion.experiments.alignn_pretrained_baseline`: pretrained ALIGNN
  readout + RF baseline.

## Interpretation

- `battery_fusion.explain`: perturbation/permutation attribution, feature
  interaction summaries, faithfulness/deletion curves, and plotting utilities.

The `imfuse` CLI is a thin dispatcher over these modules.
