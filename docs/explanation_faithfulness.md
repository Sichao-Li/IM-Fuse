# Interpretability And Faithfulness

This document describes the retained public-facing interpretability workflow
for the final mid-tri fusion model. The goal is to audit model sensitivity and
feature contribution. These experiments do not claim causal physical
mechanisms.

All commands use the public `imfuse` command. Implementation modules live under
`src/battery_fusion/explain/`.

Install the interpretation dependency with:

```bash
python -m pip install -e ".[interpretability]"
```

## Method Summary

### Composition GRS/FIS Importance

Composition importance ports the original GRS/FIS logic into a reusable command.
It produces local sample-level scores for the composition vector and an
aggregated global summary.

### Importance-Guided Deletion Faithfulness

Deletion validation reloads the same trained fusion checkpoint, masks features
in three orders, and measures prediction change:

- `top`: highest absolute importance first
- `random`: random feature order with fixed seed
- `bottom`: lowest absolute importance first

Useful columns include:

- `prediction_delta`: `abs(y_pred_original - y_pred_ablated)`
- `error_delta`: `abs(y_true - y_pred_ablated) - abs(y_true - y_pred_original)`
- `prediction_delta_auc_mean`: area under the deletion curve
- `prediction_auc_minus_random`: AUC difference from random deletion

### Standard Permutation Importance

Permutation importance is the faster global method used for composition, RDF,
and whole-structure sensitivity.

- Composition is permuted one feature at a time.
- RDF is permuted by contiguous radial windows. With `--group_size 10`, 400 RDF
  bins become 40 radial-window features.
- Whole-structure permutation shuffles complete structures across samples. This
  validates structure-modality contribution, but does not provide atom-level
  feature importance.

### Structure Atom Ablation

Atom ablation masks one atom feature vector at a time, measures prediction
change, and aggregates results by element and local atom index. The element
summary is the recommended structure feature-importance table.

## Reproduction Commands

Composition importance:

```bash
imfuse explain-composition \
  --target_col capacity_vol \
  --seed 0 \
  --split test \
  --sample_index 0 \
  --max_samples 100 \
  --device cpu \
  --output_dir results/explanations/composition_importance
```

Composition faithfulness:

```bash
imfuse explain-faithfulness \
  --importance_csv results/explanations/composition_importance/composition_importance_capacity_vol_mid_tri_rdf_tabular_structure_seed_0_test_index_0_n_100.csv \
  --target_col capacity_vol \
  --seed 0 \
  --split test \
  --modality composition \
  --device cpu \
  --output_dir results/explanation_validation/capacity_vol/composition \
  --overwrite
```

Deletion-curve figure suite:

```bash
imfuse explain-deletion \
  --target_col capacity_vol \
  --input_root results/explanation_validation/capacity_vol \
  --output_dir figures/explanation_validation \
  --device cpu
```

Single-modality permutation:

```bash
imfuse explain-permutation-single \
  --target_col capacity_vol \
  --seed 0 \
  --split test \
  --modality rdf \
  --sample_index 0 \
  --max_samples 100 \
  --group_size 10 \
  --repeats 5 \
  --device cpu \
  --output_dir results/explanation_validation/capacity_vol/rdf_permutation \
  --overwrite
```

Five-seed 3-modality permutation matrix:

```bash
imfuse explain-permutation \
  --target_col capacity_vol \
  --seeds 0 1 2 3 4 \
  --split test \
  --max_samples 0 \
  --repeats 20 \
  --rdf_group_size 10 \
  --output_dir results/explanation_validation/permutation_matrix \
  --device cpu \
  --overwrite
```

Structure atom ablation:

```bash
imfuse explain-structure \
  --target_col capacity_vol \
  --seed 0 \
  --split test \
  --sample_index 0 \
  --max_samples 100 \
  --device cpu \
  --output_dir results/explanation_validation/permutation_matrix/capacity_vol/seed_0/structure_atom_ablation \
  --overwrite
```

Interpretability summary plots:

```bash
imfuse plot-interpretability \
  --target_col capacity_vol \
  --input_root results/explanation_validation/permutation_matrix/capacity_vol \
  --output_dir figures/explanation_validation/summary
```

## Interpretation Guidance

Use cautious wording:

- `feature contribution`
- `model sensitivity`
- `faithfulness to the trained fusion model`
- `radial-window sensitivity`
- `atom-feature ablation`

Avoid stronger claims:

- `causal mechanism`
- `physical removal effect`
- `guaranteed chemical driver`

Recommended manuscript statement:

> We validate feature importance as a model-faithfulness audit. Importance-ranked
> deletion and standard permutation/ablation tests show that the trained fusion
> model is sensitive to composition entries, RDF radial windows, and atom-level
> structure features, with contributions varying across feature groups.
