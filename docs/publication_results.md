# Publication Results

This document is the public-facing index for the finalized raw-target result
set. The reproducible scope is two targets, five random seeds, the multimodal
fusion family, classical composition baselines, pretrained ALIGNN + RF,
modality-dropout robustness, halide holdout, subgroup analysis, and
explanation-validation audits.

The CSV and figure files listed here are generated outputs, not committed
artifacts. Recreate them with `scripts/reproduce_publication.sh` or the
individual commands in `docs/reproducibility.md`. Exploratory runs outside the
raw-target scope are not part of the public result narrative.

Authoritative roots:

- Metrics/configs: `results/final_publication/`
- Predictions: `results/predictions/final_publication_*`
- Main figures: `figures/final_publication/`
- Explanation validation: `results/explanation_validation/` and
  `figures/explanation_validation/`

Generated clean summary tables:

- `results/final_publication/publication_random_split_summary.csv`
- `results/final_publication/publication_experiment_b_modality_dropout_summary.csv`
- `results/final_publication/publication_experiment_c_halide_holdout_summary.csv`
- `results/final_publication/publication_experiment_d_subgroup_summary.csv`

`publication_random_split_summary.csv` keeps the unprefixed `MAE`, `RMSE`, and
`R2` columns as test-set metrics for backward compatibility and also includes
split-prefixed `train_*`, `val_*`, and `test_*` metric columns for training,
validation, and test diagnostics.

Cell Reports-style composite figures:

- `figures/final_publication/cell_reports/figure_b_modality_dropout_delta_mae.pdf`
- `figures/final_publication/cell_reports/figure_c_halide_holdout_random_vs_ood.pdf`
- `figures/final_publication/cell_reports/figure_d_subgroup_delta_mae_heatmap.pdf`
- `figures/final_publication/cell_reports/figure_d2_fusion_subgroup_win_rate.pdf`
- `figures/final_publication/cell_reports/figure_d3_fusion_baseline_win_rate.pdf`
- Plot-data CSVs: `results/final_publication/cell_reports_figure_data/`

Parity plots:

- Train/test parity figures: `figures/final_publication/parity_plots/{target}/{split}/`
- Parity metric summary: `results/final_publication/parity_plot_summary.csv`

## average_voltage

### Random Split

| model | source | modalities | seeds | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| RDF sequence encoder | neural/fusion | rdf | 5 | 0.760 +/- 0.048 | 0.335 +/- 0.086 |
| Composition neural | neural/fusion | composition | 5 | 0.472 +/- 0.030 | 0.650 +/- 0.167 |
| CGCNN-style graph | neural/fusion | graph | 5 | 0.475 +/- 0.040 | 0.654 +/- 0.179 |
| RDF + composition late fusion | neural/fusion | rdf+composition | 5 | 0.470 +/- 0.032 | 0.655 +/- 0.167 |
| RDF + graph late fusion | neural/fusion | rdf+graph | 5 | 0.472 +/- 0.042 | 0.657 +/- 0.177 |
| Composition + graph late fusion | neural/fusion | composition+graph | 5 | 0.439 +/- 0.034 | 0.679 +/- 0.181 |
| Tri-fusion early | neural/fusion | rdf+composition+graph | 5 | 0.506 +/- 0.062 | 0.624 +/- 0.164 |
| Tri-fusion mid | neural/fusion | rdf+composition+graph | 5 | 0.454 +/- 0.033 | 0.669 +/- 0.174 |
| Tri-fusion late | neural/fusion | rdf+composition+graph | 5 | 0.439 +/- 0.034 | 0.679 +/- 0.180 |
| Random forest composition | classical composition baseline | composition_counts | 5 | 0.482 +/- 0.028 | 0.620 +/- 0.156 |
| XGBoost composition | classical composition baseline | composition_counts | 5 | 0.538 +/- 0.025 | 0.607 +/- 0.140 |
| ALIGNN pretrained + RF | pretrained structure baseline | alignn_pretrained_structure | 5 | 0.464 +/- 0.031 | 0.643 +/- 0.166 |

### Experiment B: Inference-Time Modality Dropout

| condition | available modalities | seeds | MAE | R2 |
| --- | --- | --- | --- | --- |
| full | tabular+structure+rdf | 5 | 0.454 +/- 0.033 | 0.669 +/- 0.174 |
| drop_composition | structure+rdf | 5 | 0.556 +/- 0.046 | 0.601 +/- 0.165 |
| drop_graph | tabular+rdf | 5 | 1.044 +/- 0.217 | -0.025 +/- 0.472 |
| drop_rdf | tabular+structure | 5 | 0.594 +/- 0.125 | 0.583 +/- 0.217 |
| composition_only_fallback | tabular | 5 | 1.530 +/- 0.702 | -1.130 +/- 2.105 |
| graph_only_fallback | structure | 5 | 0.658 +/- 0.079 | 0.520 +/- 0.171 |
| rdf_only_fallback | rdf | 5 | 1.353 +/- 0.265 | -0.521 +/- 0.579 |

### Experiment C: Halide Anion-Family Holdout

| model | source | modalities | seeds | n_test | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| Composition | neural/fusion | composition | 5 | 1050 | 2.685 +/- 0.302 | -3.102 +/- 0.768 |
| Random forest composition | classical composition baseline | composition_counts | 5 | 1050 | 2.447 +/- 0.041 | -3.014 +/- 0.091 |
| XGBoost composition | classical composition baseline | composition_counts | 5 | 1050 | 2.330 +/- 0.021 | -2.564 +/- 0.070 |
| Graph | neural/fusion | graph | 5 | 1050 | 2.240 +/- 0.490 | -2.236 +/- 1.366 |
| Composition + graph | neural/fusion | composition+graph | 5 | 1050 | 2.653 +/- 0.454 | -3.136 +/- 1.079 |
| Full fusion | neural/fusion | composition+graph+rdf | 5 | 1050 | 2.814 +/- 0.336 | -3.540 +/- 0.879 |
| ALIGNN pretrained + RF | pretrained structure baseline | alignn_pretrained_structure | 5 | 1050 | 1.321 +/- 0.019 | -0.419 +/- 0.021 |

### Experiment D: Subgroup Analysis

| source | rows | path |
| --- | --- | --- |
| fusion/random models | 585 | `results/final_publication/average_voltage/subgroup_analysis/subgroup_metrics.csv` |
| RF/XGBoost composition baselines | 130 | `results/final_publication/average_voltage/classical_subgroup_analysis/subgroup_metrics.csv` |
| ALIGNN pretrained + RF | 65 | `results/final_publication/average_voltage/alignn_pretrained_subgroup_analysis/subgroup_metrics.csv` |

## capacity_vol

### Random Split

| model | source | modalities | seeds | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| RDF sequence encoder | neural/fusion | rdf | 5 | 213.655 +/- 8.091 | 0.274 +/- 0.176 |
| Composition neural | neural/fusion | composition | 5 | 207.462 +/- 14.076 | 0.144 +/- 0.735 |
| CGCNN-style graph | neural/fusion | graph | 5 | 199.253 +/- 13.108 | 0.545 +/- 0.148 |
| RDF + composition late fusion | neural/fusion | rdf+composition | 5 | 203.853 +/- 12.519 | 0.224 +/- 0.551 |
| RDF + graph late fusion | neural/fusion | rdf+graph | 5 | 198.145 +/- 13.326 | 0.552 +/- 0.153 |
| Composition + graph late fusion | neural/fusion | composition+graph | 5 | 197.970 +/- 13.286 | 0.515 +/- 0.223 |
| Tri-fusion early | neural/fusion | rdf+composition+graph | 5 | 198.055 +/- 12.073 | 0.505 +/- 0.128 |
| Tri-fusion mid | neural/fusion | rdf+composition+graph | 5 | 185.071 +/- 13.300 | 0.511 +/- 0.114 |
| Tri-fusion late | neural/fusion | rdf+composition+graph | 5 | 197.578 +/- 13.383 | 0.519 +/- 0.227 |
| Random forest composition | classical composition baseline | composition_counts | 5 | 182.260 +/- 7.303 | 0.485 +/- 0.059 |
| XGBoost composition | classical composition baseline | composition_counts | 5 | 179.204 +/- 9.654 | 0.562 +/- 0.087 |
| ALIGNN pretrained + RF | pretrained structure baseline | alignn_pretrained_structure | 5 | 193.732 +/- 9.553 | 0.506 +/- 0.102 |

### Experiment B: Inference-Time Modality Dropout

| condition | available modalities | seeds | MAE | R2 |
| --- | --- | --- | --- | --- |
| full | tabular+structure+rdf | 5 | 185.071 +/- 13.300 | 0.511 +/- 0.114 |
| drop_composition | structure+rdf | 5 | 384.012 +/- 126.802 | -0.780 +/- 1.223 |
| drop_graph | tabular+rdf | 5 | 248.433 +/- 19.394 | 0.198 +/- 0.171 |
| drop_rdf | tabular+structure | 5 | 202.093 +/- 9.292 | 0.452 +/- 0.129 |
| composition_only_fallback | tabular | 5 | 292.664 +/- 21.508 | -0.083 +/- 0.289 |
| graph_only_fallback | structure | 5 | 369.566 +/- 93.675 | -0.787 +/- 1.044 |
| rdf_only_fallback | rdf | 5 | 351.994 +/- 86.750 | -0.479 +/- 0.578 |

### Experiment C: Halide Anion-Family Holdout

| model | source | modalities | seeds | n_test | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| Composition | neural/fusion | composition | 5 | 1050 | 672.733 +/- 153.070 | -16.409 +/- 6.820 |
| Random forest composition | classical composition baseline | composition_counts | 5 | 1050 | 347.734 +/- 17.902 | -3.272 +/- 0.548 |
| XGBoost composition | classical composition baseline | composition_counts | 5 | 1050 | 335.224 +/- 24.641 | -2.480 +/- 0.541 |
| Graph | neural/fusion | graph | 5 | 1050 | 503.280 +/- 284.833 | -7.250 +/- 7.320 |
| Composition + graph | neural/fusion | composition+graph | 5 | 1050 | 231.841 +/- 78.521 | -0.879 +/- 1.484 |
| Full fusion | neural/fusion | composition+graph+rdf | 5 | 1050 | 183.120 +/- 6.959 | -0.075 +/- 0.109 |
| ALIGNN pretrained + RF | pretrained structure baseline | alignn_pretrained_structure | 5 | 1050 | 206.000 +/- 3.221 | -0.104 +/- 0.043 |

### Experiment D: Subgroup Analysis

| source | rows | path |
| --- | --- | --- |
| fusion/random models | 585 | `results/final_publication/capacity_vol/subgroup_analysis/subgroup_metrics.csv` |
| RF/XGBoost composition baselines | 130 | `results/final_publication/capacity_vol/classical_subgroup_analysis/subgroup_metrics.csv` |
| ALIGNN pretrained + RF | 65 | `results/final_publication/capacity_vol/alignn_pretrained_subgroup_analysis/subgroup_metrics.csv` |

## Explanation Validation

Explanation-validation outputs are retained separately because they audit
trained-model sensitivity rather than predictive performance. The public set
includes five-seed permutation matrices, permutation-derived deletion curves,
and atom-level structure ablation/deletion curves.

- Matrix root: `results/explanation_validation/permutation_matrix/`
- Deletion-curve metrics: `results/explanation_validation/permutation_deletion_curves/`,
  `results/explanation_validation/structure_atom_deletion_curves/`
- Figures: `figures/explanation_validation/`

See `docs/explanation_faithfulness.md` for commands and interpretation
guidance.
