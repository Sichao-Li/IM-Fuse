# Reproduce The Manuscript Experiments

This is the short path. The full command log and options are in
`docs/reproducibility.md`.

## 1. Prepare Artifacts

Place external data at:

```text
data/raw/mp_total.csv
data/raw/cifs/*.cif
data/raw/atom_init.json
data/processed/legacy_rdf_split_seed_42/
```

To rebuild the raw inputs and processed modality caches from an external
`mp_total.csv` and CIF source, run `imfuse prepare-data` first. See
`docs/data_preparation.md` for GitHub release, local folder, URL-template, and
Materials Project examples.

The included split files define the retained seeds `0 1 2 3 4`.

## 2. Install

```bash
conda activate battery
pip install -e ".[classical]"
```

For pretrained ALIGNN + RF, create the optional ALIGNN environment described in
`docs/reproducibility.md`.

## 3. Run

```bash
DEVICE=mps bash scripts/reproduce_publication.sh
```

Use `DEVICE=cuda` on a CUDA server. Set `RUN_ALIGNN_PRETRAINED=0` to skip the
optional pretrained ALIGNN baseline.

To run only the OOD audit table:

```bash
DEVICE=mps bash scripts/run_ood_publication.sh
```

## 4. Outputs

Main summaries are written to:

```text
results/final_publication/publication_random_split_summary.csv
results/final_publication/publication_experiment_b_modality_dropout_summary.csv
results/final_publication/publication_experiment_c_ood_summary.csv
results/final_publication/publication_experiment_d_subgroup_summary.csv
results/final_publication_ood/
```

Publication figures are written to:

```text
figures/final_publication/
```
