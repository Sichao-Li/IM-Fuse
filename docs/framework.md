# Framework Overview

IM-Fuse separates the project into reusable layers.

## Data

- `battery_fusion.data`: label normalization, split creation, preprocessing.
- `battery_fusion.features`: composition, RDF, and structure feature builders.
- `battery_fusion.fusion.feature_store`: sample-aligned modality cache loading.

## Models

- `battery_fusion.models.lstm`: retained RDF sequence baseline.
- `battery_fusion.fusion.cgcnn_multimodal`: composition, RDF, and CGCNN-style
  encoders plus early and intermediate fusion models.
- `battery_fusion.experiments.publication`: unimodal training, validation-based
  early stopping, validation-fitted late fusion, and prediction export.
- `battery_fusion.training`: regression metrics and optional target transforms.

## Experiments

- `battery_fusion.experiments.publication`: random-split publication matrix.
- `battery_fusion.experiments.modality_dropout`: inference-time missing-modality robustness.
- `battery_fusion.experiments.ood_splits`: composition-cluster and working-ion OOD split generation.
- `battery_fusion.experiments.subgroups`: anion-family and working-ion audits.
- `battery_fusion.experiments.classical_baselines`: RF/XGBoost baselines.
- `battery_fusion.experiments.alignn_pretrained_baseline`: pretrained ALIGNN
  readout + RF baseline.

## Interpretation

- `battery_fusion.explain`: perturbation/permutation attribution, feature
  interaction summaries, faithfulness/deletion curves, and plotting utilities.

The `imfuse` CLI is a thin dispatcher over these modules.
