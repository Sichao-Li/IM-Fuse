# Reproducibility Guide

All commands assume the clean repo root:

```bash
conda activate battery
cd battery-fusion-public
pip install -e ".[classical,interpretability]"
```

The public release keeps one command surface: `imfuse`. The Python modules under
`src/battery_fusion/experiments/` are implementation details and are not meant to
be called directly by new users.

The retained manuscript scope uses raw targets only, seeds `0 1 2 3 4`, and the
two targets `average_voltage` and `capacity_vol`.

Before a full rerun, validate the tracked splits and separately distributed
artifacts:

```bash
imfuse check --strict-artifacts
```

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

The runner does not overwrite existing outputs by default. For a deliberate
replacement rerun, set `OVERWRITE=1`.

The runner executes the random-split neural matrix, RF/XGBoost baselines,
pretrained ALIGNN+RF, Experiment B modality dropout, Experiment C OOD audits,
Experiment D subgroup analysis, summary tables, retained manuscript figures,
and train/test parity plots.

## Core Commands

Random-split uni/dual/tri neural matrix:

```bash
imfuse train \
  --processed_root data/processed/publication \
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
  --processed_root data/processed/publication \
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

## Experiment C: OOD Audits

The retained OOD audit has two protocols:

- composition-cluster holdout: KMeans in composition-count feature space,
  default `K=3`, no target labels used for clustering;
- working-ion holdout: Na holdout and Mg/Ca/Zn multivalent holdout.

Run all retained OOD scenarios for both targets:

```bash
DEVICE=mps bash scripts/run_ood_publication.sh
```

By default the OOD script uses seed `0`, matching the manuscript OOD table. To
run more seeds:

```bash
OOD_SEEDS="0 1 2 3 4" DEVICE=mps bash scripts/run_ood_publication.sh
```

Create only the split files:

```bash
imfuse split-ood composition-cluster \
  --input_data data/raw/mp_total.csv \
  --sample_id_col id_discharge \
  --formula_col formula_discharge \
  --target_col average_voltage \
  --working_ion_col working_ion \
  --output_dir data/splits/publication_ood/composition_cluster_holdout/average_voltage/k_3 \
  --seeds 0 \
  --n_clusters 3 \
  --min_test_size 50 \
  --overwrite

imfuse split-ood working-ion \
  --input_data data/raw/mp_total.csv \
  --sample_id_col id_discharge \
  --formula_col formula_discharge \
  --target_col average_voltage \
  --working_ion_col working_ion \
  --heldout_ions Mg Ca Zn \
  --output_dir data/splits/publication_ood/working_ion_holdout/average_voltage/Mg_Ca_Zn \
  --seeds 0 \
  --min_test_size 50 \
  --overwrite
```

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
  --ood_results_root results/final_publication_ood \
  --output_dir results/final_publication \
  --overwrite

imfuse figures \
  --results_root results/final_publication \
  --output_dir figures/final_publication/main \
  --data_output_dir results/final_publication/figure_data \
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
imfuse check
PYTHONPATH=src python -m unittest discover -s tests
```
