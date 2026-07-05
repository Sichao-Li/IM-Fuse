# Reproducibility Guide

All commands assume the clean repo root:

```bash
conda activate battery
cd battery-fusion-public
pip install -e ".[classical]"
```

The public release keeps one command surface: `imfuse`. The Python modules under
`src/battery_fusion/experiments/` are implementation details and are not meant to
be called directly by new users.

The retained manuscript scope uses raw targets only, seeds `0 1 2 3 4`, and the
two targets `average_voltage` and `capacity_vol`.

## One-Command Final Rerun

```bash
DEVICE=mps bash scripts/reproduce_publication.sh
```

Use `DEVICE=cuda` on a CUDA server. Useful overrides:

```bash
DEVICE=cuda \
PYTHON=python \
ALIGNN_PYTHON=.venv-alignn/bin/python \
bash scripts/reproduce_publication.sh
```

The runner executes the random-split neural matrix, RF/XGBoost baselines,
pretrained ALIGNN+RF, Experiment B modality dropout, Experiment C halide
holdout, Experiment D subgroup analysis, summary tables, Cell Reports-style
figures, and train/test parity plots.

## Core Commands

Random-split uni/dual/tri neural matrix:

```bash
imfuse train \
  --processed_root data/processed/legacy_rdf_split_seed_42 \
  --raw_data data/raw/mp_total.csv \
  --target_col average_voltage \
  --split_dir data/splits/publication/average_voltage \
  --output_dir results/final_publication/average_voltage/random_split \
  --assignment_output results/final_publication/average_voltage/random_split/anion_family_assignments.csv \
  --experiment_name final_publication_random \
  --predictions_root results/predictions \
  --seeds 0 1 2 3 4 \
  --epochs 1000 \
  --rdf_epochs 1000 \
  --tabular_epochs 1000 \
  --structure_epochs 1000 \
  --tri_epochs 1000 \
  --early_stopping_patience 100 \
  --batch_size 256 \
  --device mps \
  --skip_split_creation \
  --overwrite
```

Classical composition baselines:

```bash
imfuse baseline-classical \
  --target_col average_voltage \
  --split_dir data/splits/publication/average_voltage \
  --output_dir results/final_publication/average_voltage/classical_baselines \
  --experiment_name final_publication_classical_random \
  --predictions_root results/predictions \
  --seeds 0 1 2 3 4 \
  --vocabulary_csv data/raw/mp_total.csv \
  --vocabulary_formula_col formula_discharge \
  --include_xgboost \
  --overwrite
```

Pretrained ALIGNN readout + RF baseline:

```bash
imfuse baseline-alignn \
  --target_col average_voltage \
  --split_dir data/splits/publication/average_voltage \
  --cif_dir data/raw/cifs \
  --output_dir results/final_publication/average_voltage/alignn_pretrained_rf \
  --experiment_name final_publication_alignn_pretrained \
  --predictions_root results/predictions \
  --alignn_python .venv-alignn/bin/python \
  --pretrained_models mp_e_form_alignn \
  --feature_mode readout \
  --feature_cache_dir results/final_publication/alignn_pretrained_features \
  --seeds 0 1 2 3 4 \
  --n_estimators 500 \
  --overwrite
```

Repeat the same commands with `capacity_vol` paths for the capacity task.

## Experiment B: Modality Dropout

```bash
imfuse dropout \
  --target_name average_voltage \
  --processed_root data/processed/legacy_rdf_split_seed_42 \
  --checkpoint_dir results/final_publication/average_voltage/random_split/checkpoints/mid_tri_rdf_tabular_structure \
  --split_dir data/splits/publication/average_voltage \
  --output_dir results/final_publication/average_voltage/modality_dropout_mid_tri \
  --metadata results/final_publication/average_voltage/random_split/anion_family_assignments.csv \
  --experiment_name final_publication_modality_dropout \
  --predictions_root results/predictions \
  --device mps \
  --seeds 0 1 2 3 4 \
  --overwrite
```

## Experiment C: Halide Holdout

Create or verify splits:

```bash
imfuse split \
  --input_data data/raw/mp_total.csv \
  --sample_id_col id_discharge \
  --formula_col formula_discharge \
  --target_col average_voltage \
  --working_ion_col working_ion \
  --heldout_family halide \
  --output_dir data/splits/publication_anion_holdout/average_voltage/halide \
  --assignment_output results/final_publication/average_voltage/random_split/anion_family_assignments.csv \
  --seeds 0 1 2 3 4 \
  --min_test_samples 100 \
  --overwrite
```

Run the retained holdout neural models:

```bash
imfuse holdout \
  --processed_root data/processed/legacy_rdf_split_seed_42 \
  --split_dir data/splits/publication_anion_holdout/average_voltage/halide \
  --models composition graph composition_graph full_fusion \
  --output_dir results/final_publication/average_voltage/anion_holdout_halide \
  --seeds 0 1 2 3 4 \
  --fusion mid \
  --epochs 1000 \
  --batch_size 256 \
  --learning_rate 0.0005 \
  --device mps \
  --early_stopping_patience 100 \
  --target_transform none \
  --predictions_root results/predictions/final_publication_anion_holdout/average_voltage \
  --overwrite
```

Run RF/XGBoost and ALIGNN+RF on the same holdout split by reusing
`imfuse baseline-classical` and `imfuse baseline-alignn` with
`--split_dir data/splits/publication_anion_holdout/average_voltage/halide`.

## Experiment D: Subgroup Analysis

Fusion models:

```bash
imfuse subgroups \
  --predictions_dir results/predictions/final_publication_random/average_voltage \
  --metadata results/final_publication/average_voltage/random_split/anion_family_assignments.csv \
  --target_col average_voltage \
  --output_dir results/final_publication/average_voltage/subgroup_analysis \
  --min_group_size 30 \
  --split test \
  --overwrite
```

Use the same command for classical and ALIGNN+RF predictions by changing
`--predictions_dir` and `--output_dir`.

## Tables, Figures, And Parity Plots

```bash
imfuse tables \
  --results_root results/final_publication \
  --output_dir results/final_publication \
  --overwrite

imfuse figures \
  --results_root results/final_publication \
  --output_dir figures/final_publication/cell_reports \
  --data_output_dir results/final_publication/cell_reports_figure_data \
  --overwrite

imfuse parity \
  --predictions_root results/predictions \
  --output_dir figures/final_publication/parity_plots \
  --summary_output results/final_publication/parity_plot_summary.csv \
  --splits train test \
  --overwrite
```

## Explanation Validation

Five-seed permutation matrix:

```bash
imfuse explain-permutation \
  --target_col average_voltage \
  --seeds 0 1 2 3 4 \
  --split test \
  --max_samples 0 \
  --repeats 20 \
  --rdf_group_size 10 \
  --output_dir results/explanation_validation/permutation_matrix \
  --device cpu \
  --overwrite
```

Single-job atom ablation:

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

Deletion-curve figure suite:

```bash
imfuse explain-deletion \
  --target_col capacity_vol \
  --input_root results/explanation_validation/capacity_vol \
  --output_dir figures/explanation_validation \
  --device cpu
```

## Verification

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
