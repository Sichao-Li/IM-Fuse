# Architecture

The clean rebuild separates raw data, target labels, split identity, cached
features, model training, and publication analysis.

## Data Flow

```text
data/raw/mp_total.csv + data/raw/cifs/*.cif
  -> target metadata grouped by id_discharge, keeping the last row
  -> target-specific split CSVs
  -> data/processed/legacy_rdf_split_seed_42/{rdf,tabular,structure}
  -> results/final_publication/{target}/...
  -> results/predictions/final_publication_*
  -> figures/final_publication/{target}/...
  -> results/explanation_validation/...
  -> figures/explanation_validation/...
```

## Modalities

- `tabular`: composition descriptors
- `structure`: CGCNN-style crystal graph tensors
- `rdf`: fixed-length RDF descriptors

## Reproducibility Rules

- The target value is loaded from `data/raw/mp_total.csv` for each experiment,
  not from cached feature files.
- Labels are grouped by `id_discharge`, keeping the last source row.
- Random publication splits use the same sample pool for every seed but assign
  different train/validation/test IDs.
- Holdout splits keep the held-out anion family out of train and validation.
- Training, metrics, configs, checkpoints, and predictions are saved under
  target-specific publication folders.

## Scope

Current scope is model training, evaluation, reviewer-requested robustness/OOD
experiments, and model-sensitivity explanation validation for publication.
