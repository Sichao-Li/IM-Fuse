# Extend IM-Fuse

Use this checklist when adding a model, modality, or evaluation protocol.

## Add A New Modality

1. Implement the feature builder in `src/battery_fusion/features/`.
2. Save one tensor/object per sample in the same cache style as existing
   modalities.
3. Register the modality in `ProcessedFeatureStore` and the publication dataset
   used by `src/battery_fusion/experiments/publication.py`.
4. Add a small feature-builder and alignment test under `tests/`.

## Add A New Model

1. Put single-modality models in `src/battery_fusion/models/`.
2. Put fusion models in `src/battery_fusion/fusion/`.
3. Reuse the validated fit/evaluate helpers in
   `battery_fusion.experiments.publication` for early stopping, metrics, and
   prediction export.
4. Add the model to the publication matrix only if it is part of the retained
   study.

## Add A New Evaluation

1. Implement core logic under `src/battery_fusion/experiments/`.
2. Save metrics as CSV and predictions as CSV.
3. Add a small command in `src/battery_fusion/cli.py`.
4. Add a lightweight unit test that does not require full training.

## Add An Interpretation Method

1. Implement reusable logic under `src/battery_fusion/explain/`.
2. Keep numerical outputs separate from plotting outputs.
3. Report whether the method audits prediction sensitivity, feature
   contribution, or model interaction. Avoid causal claims unless the method
   supports them.
