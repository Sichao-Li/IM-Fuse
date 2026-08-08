# Framework Overview

IM-Fuse separates the project into reusable layers. The names most users need
are re-exported at the top level (`from imfuse import ...`); see
[examples/quickstart.py](../examples/quickstart.py) for the complete
feature-build → train → probe path on new data.

## Data

- `imfuse.data`: label normalization, split creation, preprocessing.
- `imfuse.features`: composition, RDF, and structure feature builders.
- `imfuse.fusion.feature_store`: sample-aligned modality cache loading.

## Models

- `imfuse.models.lstm`: retained RDF sequence baseline.
- `imfuse.fusion.cgcnn_multimodal`: composition, RDF, and CGCNN-style
  encoders plus early and intermediate fusion models.
- `imfuse.experiments.publication`: unimodal training, validation-based
  early stopping, validation-fitted late fusion, and prediction export.
- `imfuse.training`: regression metrics and optional target transforms.

## Experiments

- `imfuse.experiments.publication`: random-split publication matrix.
- `imfuse.experiments.modality_dropout`: inference-time missing-modality robustness.
- `imfuse.experiments.ood_splits`: composition-cluster and working-ion OOD split generation.
- `imfuse.experiments.subgroups`: anion-family and working-ion audits.
- `imfuse.experiments.classical_baselines`: RF/XGBoost baselines.
- `imfuse.experiments.alignn_pretrained_baseline`: pretrained ALIGNN
  readout + RF baseline.

## Interpretation

- `imfuse.explain`: perturbation/permutation attribution, feature
  interaction summaries, faithfulness/deletion curves, and plotting utilities.

The `imfuse` CLI is a thin dispatcher over these modules.
