# Extend IM-Fuse

Use this checklist when adding a model, modality, or evaluation protocol.

## Run IM-Fuse On Your Own Data

Start from [examples/quickstart.py](../examples/quickstart.py). It shows the
minimal in-memory path with no publication data contract: build the three
modality features with `formula_vector`, `build_rdf_vector`, and
`build_crystal_graph`; wrap them in a small in-memory feature store; assemble
batches with `PublicationTriDataset` and `collate_tri`; train with
`train_torch_model`; and probe modality reliance with
`masked_modality_forward`. All of these names import directly from `imfuse`.

Two things the library expects you to supply:

- an `atom_init.json` mapping atomic numbers to embedding vectors (any
  consistent width works; the quickstart writes a one-hot version);
- a target value per sample for your property of interest.

## Add A New Modality

1. Implement the feature builder in `src/imfuse/features/`.
2. Save one tensor/object per sample in the same cache style as existing
   modalities.
3. Register the modality in `ProcessedFeatureStore` and the publication dataset
   used by `src/imfuse/experiments/publication.py`.
4. Add a small feature-builder and alignment test under `tests/`.

## Add A New Model

1. Put single-modality models in `src/imfuse/models/`.
2. Put fusion models in `src/imfuse/fusion/`.
3. Reuse the validated fit/evaluate helpers in
   `imfuse.experiments.publication` for early stopping, metrics, and
   prediction export.
4. Add the model to the publication matrix only if it is part of the retained
   study.

## Add A New Evaluation

1. Implement core logic under `src/imfuse/experiments/`.
2. Save metrics as CSV and predictions as CSV.
3. Add a small command in `src/imfuse/cli.py`.
4. Add a lightweight unit test that does not require full training.

## Add An Interpretation Method

1. Implement reusable logic under `src/imfuse/explain/`.
2. Keep numerical outputs separate from plotting outputs.
3. Report whether the method audits prediction sensitivity, feature
   contribution, or model interaction. Avoid causal claims unless the method
   supports them.
