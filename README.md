# IM-Fuse

**Interpretable Multimodal Fusion for Battery Materials**

IM-Fuse is a compact research framework for training, evaluating, and auditing
multimodal battery-material predictors. It aligns three representations:

- composition/tabular descriptors,
- CGCNN-style crystal-graph features,
- RDF/radial-distribution descriptors.

The public release supports two manuscript targets, `average_voltage` and
`capacity_vol`, and keeps the narrative deliberately cautious: fusion is used
to audit modality contribution, missing-modality robustness, chemistry-aware
generalization, subgroup-dependent complementarity, and attribution
faithfulness.

## Install

```bash
conda activate battery
cd battery-fusion-public
pip install -e ".[classical]"
```

The package installs the `imfuse` command:

```bash
imfuse --help
```

Optional pretrained ALIGNN + RF baselines use a separate ALIGNN environment;
see `docs/reproduce.md` or `docs/reproducibility.md`.

## Quick Start

Run a command directly:

```bash
imfuse prepare-data --help
imfuse train --help
imfuse dropout --help
imfuse split-ood --help
imfuse subgroups --help
imfuse explain-permutation --help
```

Or reproduce the retained raw-target manuscript pipeline:

```bash
DEVICE=mps bash scripts/reproduce_publication.sh
```

Use `DEVICE=cuda` on a CUDA server.

## Public Layout

```text
configs/                    # public defaults
data/sample_order/          # compact one-row-per-sample ID order table
data/splits/                # fixed publication split CSVs
docs/                       # data, framework, reproduction, and output notes
scripts/                    # end-to-end reproduction scripts
src/battery_fusion/         # reusable framework code
tests/                      # lightweight unit tests
```

Large artifacts are intentionally excluded from Git: raw CIFs, `mp_total.csv`,
processed tensor caches, checkpoints, raw predictions, generated `results/`,
generated `figures/`, pretrained readout caches, local environments, logs, and
exploratory trial outputs.

## Data Contract

For full reruns, place external artifacts at:

```text
data/raw/mp_total.csv
data/raw/cifs/*.cif
data/raw/atom_init.json
data/processed/publication/
```

To rebuild those artifacts from an external `mp_total.csv` and CIF release, see
`docs/data_preparation.md`. The compact sample-order table and split CSVs are included;
see `docs/data_manifest.md` for retained artifact details.

## Command Map

```text
imfuse prepare-data          stage/download mp_total + CIFs, labels, splits, caches
imfuse preprocess           build aligned modality caches
imfuse random-split         create deterministic random split manifests
imfuse train                run the neural fusion publication matrix
imfuse baseline-classical   run RF/XGBoost composition baselines
imfuse baseline-alignn      run pretrained ALIGNN readout + RF baseline
imfuse dropout              run inference-time modality dropout
imfuse split-ood            create composition-cluster or working-ion OOD splits
imfuse subgroups            compute anion/working-ion subgroup metrics
imfuse tables               build summary tables
imfuse figures              build publication figures
imfuse parity               build train/test parity plots
imfuse explain-*            run attribution and faithfulness audits
```

`imfuse` is the only public command surface. The implementation lives under
`src/battery_fusion/`, including reproducibility modules in
`src/battery_fusion/experiments/`.

## Generated Outputs

The reproduction scripts generate summary tables and figures under:

- `results/final_publication/publication_random_split_summary.csv`
- `results/final_publication/publication_experiment_b_modality_dropout_summary.csv`
- `results/final_publication/publication_experiment_c_ood_summary.csv`
- `results/final_publication/publication_experiment_d_subgroup_summary.csv`
- `results/final_publication_ood/`
- `figures/final_publication/`

These output folders are ignored by Git so users can reproduce them locally.
`docs/publication_results.md` documents the expected output structure.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

The public tests are smoke tests for reusable behavior, not a full manuscript
regression suite. See `tests/README.md`.

## Extend

Most extensions should touch only one layer:

- add a descriptor in `src/battery_fusion/features/`,
- add a model in `src/battery_fusion/models/` or `src/battery_fusion/fusion/`,
- add an evaluation protocol in `src/battery_fusion/experiments/`,
- expose it through `src/battery_fusion/cli.py`.
