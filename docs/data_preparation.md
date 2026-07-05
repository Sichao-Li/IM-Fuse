# Data Preparation From `mp_total.csv`

This page describes the public, reproducible path from an external
`mp_total.csv` table and CIF structures to the three IM-Fuse modalities:
composition, CGCNN-style structure graphs, and RDF vectors.

The preparation code is intentionally small and explicit. It does not assume a
private local folder. Inputs can come from a GitHub release, Zenodo, an
institutional mirror, a local CIF directory, or Materials Project if the user
has an API key.

## Expected Inputs

At minimum, the source table needs:

| Meaning | Default column | Override |
| --- | --- | --- |
| sample ID | `id_discharge` | `--id-col` |
| discharge formula | `formula_discharge` | `--formula-col` |
| target value | e.g. `average_voltage`, `capacity_vol` | `--target-col` |
| working ion | `working_ion` | `--working-ion-col` |

CIF files should resolve to one file per retained sample ID:

```text
{id_discharge}.cif
```

If the CIF source uses a nested folder or mixed `.cif`/`.CIF` suffixes, the
preparation command copies matching files into:

```text
data/raw/cifs/{id_discharge}.cif
```

## One-Command Preparation

For a GitHub or release-hosted source:

```bash
export MP_TOTAL_URL="https://github.com/<org>/<repo>/releases/download/<tag>/mp_total.csv"
export CIF_ARCHIVE_URL="https://github.com/<org>/<repo>/releases/download/<tag>/cifs.zip"
export ATOM_INIT_URL="https://github.com/<org>/<repo>/releases/download/<tag>/atom_init.json"

imfuse prepare-data \
  --target-col average_voltage \
  --mp-total-url "$MP_TOTAL_URL" \
  --cif-archive-url "$CIF_ARCHIVE_URL" \
  --atom-init-url "$ATOM_INIT_URL" \
  --seeds 0 1 2 3 4 \
  --preprocess \
  --overwrite
```

Repeat for `capacity_vol`:

```bash
imfuse prepare-data \
  --target-col capacity_vol \
  --mp-total-url "$MP_TOTAL_URL" \
  --cif-archive-url "$CIF_ARCHIVE_URL" \
  --atom-init-url "$ATOM_INIT_URL" \
  --seeds 0 1 2 3 4 \
  --preprocess \
  --overwrite
```

For local files:

```bash
imfuse prepare-data \
  --target-col average_voltage \
  --mp-total /path/to/mp_total.csv \
  --cif-dir /path/to/cif_directory \
  --atom-init /path/to/atom_init.json \
  --seeds 0 1 2 3 4 \
  --preprocess
```

For a per-sample CIF URL pattern:

```bash
imfuse prepare-data \
  --target-col average_voltage \
  --mp-total-url "$MP_TOTAL_URL" \
  --cif-url-template "https://raw.githubusercontent.com/<org>/<repo>/<ref>/cifs/{sample_id}.cif" \
  --seeds 0 1 2 3 4 \
  --preprocess
```

For Materials Project download, install the optional data extra and provide the
column containing MP material IDs:

```bash
pip install -e ".[data]"

imfuse prepare-data \
  --target-col average_voltage \
  --mp-total /path/to/mp_total.csv \
  --mp-api-key "$MP_API_KEY" \
  --mp-id-col material_id \
  --seeds 0 1 2 3 4 \
  --preprocess
```

If `id_discharge` itself is the MP material ID, use:

```bash
--mp-id-col id_discharge
```

## What The Command Writes

For each target, the command writes:

```text
data/raw/mp_total.csv
data/raw/cifs/{id_discharge}.cif
data/raw/atom_init.json
data/labels/{target_col}_labels_keep_last.csv
data/manifests/{target_col}_cif_coverage.csv
data/manifests/{target_col}_data_foundation_manifest.json
data/splits/random/{target_col}/random_{target_col}_seed_{seed}.json
data/processed/random_{target_col}_seed_{seed}/
```

The label table has the standardized columns used by the framework:

```text
id_discharge,target,formula_discharge,working_ion
```

The command keeps the last row for each `id_discharge`, matching the manuscript
pipeline. Rows with missing/non-numeric target values are dropped and counted in
the manifest.

## Modality Processing

With `--preprocess`, IM-Fuse builds:

- `tabular`: raw formula element-count vectors from `formula_discharge`;
- `structure`: CGCNN-style graph tensors from CIF files and `atom_init.json`;
- `rdf`: fixed-length `rdfpy` RDF vectors from the same CIF structures.

Without `--preprocess`, the command only stages raw inputs, labels, CIF coverage,
and split manifests. This is useful for auditing external data before running
the slower RDF/graph conversion.

## Reproducing Publication Inputs

The public repository does not commit large raw artifacts. To reproduce the
manuscript from scratch:

1. Publish or download `mp_total.csv`, a CIF archive, and `atom_init.json`.
2. Run `imfuse prepare-data` for `average_voltage` and `capacity_vol`.
3. Run the retained manuscript matrix:

```bash
DEVICE=mps bash scripts/reproduce_publication.sh
```

Use `DEVICE=cuda` on a CUDA server.

## Adapting To A New Dataset

For a new battery dataset, prepare a CSV with one row per observation or one row
per state before deduplication. Then map its columns:

```bash
imfuse prepare-data \
  --target-col my_property \
  --id-col sample_id \
  --formula-col discharged_formula \
  --working-ion-col ion \
  --mp-total my_dataset.csv \
  --cif-dir my_cifs \
  --seeds 0 1 2 3 4 \
  --preprocess
```

After preparation, train on the generated split:

```bash
imfuse train --help
```

For extensions, add a new descriptor under `src/battery_fusion/features/` and
wire it into the preprocessing command rather than editing the training scripts.
