# Publication Result Outputs

This repo does not commit generated `results/` or `figures/` artifacts. The
files below are produced by `scripts/reproduce_publication.sh` and are the
public result interface for the manuscript.

## Main Tables

| Output | Purpose |
| --- | --- |
| `results/final_publication/publication_random_split_summary.csv` | Random-split train/validation/test metrics for unimodal, dual-fusion, tri-fusion, RF, XGBoost, and pretrained ALIGNN+RF models. |
| `results/final_publication/publication_experiment_b_modality_dropout_summary.csv` | Inference-time modality-dropout metrics for the retained full-fusion model. |
| `results/final_publication/publication_experiment_c_ood_summary.csv` | Composition-cluster and working-ion OOD metrics from `results/final_publication_ood/`. |
| `results/final_publication/publication_experiment_d_subgroup_summary.csv` | Test-set subgroup metrics by anion family and working ion. |
| `results/final_publication/publication_manifest.csv` | Index of generated public-scope metrics and figures. |

## Main Figures

| Output | Purpose |
| --- | --- |
| `figures/final_publication/main/figure_b_modality_dropout_delta_mae.pdf` | MAE increase after inference-time modality dropout. |
| `figures/final_publication/main/figure_d_subgroup_delta_mae_heatmap.pdf` | Subgroup MAE deltas relative to the best non-fusion comparator. |
| `figures/final_publication/main/figure_d2_fusion_subgroup_win_rate.pdf` | Fusion win rate against unimodal branches. |
| `figures/final_publication/main/figure_d3_fusion_baseline_win_rate.pdf` | Fusion win rate against neural, classical, and pretrained baselines. |
| `figures/final_publication/parity_plots/` | Train/test parity plots by target, split, and model. |

## OOD Result Root

`scripts/run_ood_publication.sh` writes composition-cluster and working-ion
holdout outputs under:

```text
results/final_publication_ood/{target}/composition_cluster_holdout/k_3/cluster_{0,1,2}/
results/final_publication_ood/{target}/working_ion_holdout/{Na,Mg_Ca_Zn}/
```

Each scenario contains:

- `neural/publication_metrics.csv`
- `classical_baselines/classical_baseline_metrics.csv`
- `alignn_pretrained_rf/alignn_pretrained_rf_metrics.csv`

Use `imfuse tables --ood_results_root results/final_publication_ood` to collect
these files into `publication_experiment_c_ood_summary.csv`.

## Explanation Outputs

Explanation-validation outputs are generated separately because they audit model
sensitivity rather than predictive performance:

- matrices: `results/explanation_validation/permutation_matrix/`
- deletion-curve metrics: `results/explanation_validation/permutation_deletion_curves/`
- structure atom ablation: `results/explanation_validation/structure_atom_deletion_curves/`
- figures: `figures/explanation_validation/`

See `docs/explanation_faithfulness.md` for commands and interpretation notes.
