# IM-Fuse

**Interpretable Multimodal Fusion for Battery Materials**

IM-Fuse is the reproducible implementation accompanying *Multimodal Fusion of
Complementary Material Representations for Battery Property Prediction with
Generalization Stability and Interpretability*. It aligns three material
representations:

- composition element-count descriptors;
- CGCNN-style crystal graphs;
- radial distribution function (RDF) vectors.

The release predicts average voltage (V) and volumetric capacity (mAh cm^-3)
with single-modality, pairwise-fusion, and three-modality models. It also
provides modality-dropout, chemistry-aware OOD, subgroup, attribution,
interaction, and deletion-faithfulness audits. These analyses describe model
reliance and subgroup-dependent complementarity; they do not establish causal
physical mechanisms or universal fusion superiority.

## Install

With Conda:

```bash
conda env create -f environment.yml
conda activate im-fuse
```

Or install into an existing Python 3.10+ environment:

```bash
python -m pip install -e ".[classical,interpretability]"
```

Pretrained ALIGNN+RF uses a separate ALIGNN/DGL environment; see
[docs/reproducibility.md](docs/reproducibility.md).

## Validate The Checkout

```bash
imfuse check
PYTHONPATH=src python -m unittest discover -s tests
```

`imfuse check` verifies package versions, checksums, split schema,
disjointness, deterministic membership, seed coverage, and cross-target
alignment. Missing large external artifacts are reported as warnings. After
downloading them, require the complete data contract with:

```bash
imfuse check --strict-artifacts
```

## Data

Git tracks the exact five-seed split assignments and their checksums. The
model-ready intersection contains 8,088 unique discharge IDs, with
6,470/808/810 train/validation/test samples per seed. Both targets use the same
IDs and membership. The source table contains 10,123 rows before repeated
discharge IDs are resolved; see
[data/README.md](data/README.md) for the two-stage contract.

Full reruns additionally require:

```text
data/raw/mp_total.csv
data/raw/cifs/*.cif
data/raw/atom_init.json
data/processed/publication/
```

These large or source-licensed artifacts are distributed separately from Git.
See [data/README.md](data/README.md),
[docs/data_preparation.md](docs/data_preparation.md), and
[docs/data_and_code_availability.md](docs/data_and_code_availability.md).

## Reproduce

Once the full data contract passes validation:

```bash
DEVICE=mps bash scripts/reproduce_publication.sh
```

Use `DEVICE=cuda` on a CUDA server. The runner executes the retained raw-target
pipeline for seeds 0-4: neural uni/dual/tri models, RF/XGBoost, pretrained
ALIGNN+RF, modality dropout, OOD evaluation, subgroup analysis, summary tables,
and manuscript figures. Full training is computationally intensive and runtime
depends on hardware. Existing outputs are preserved by default; set
`OVERWRITE=1` only for a deliberate replacement rerun.

The main generated tables are:

```text
results/final_publication/publication_random_split_summary.csv
results/final_publication/publication_experiment_b_modality_dropout_summary.csv
results/final_publication/publication_experiment_c_ood_summary.csv
results/final_publication/publication_experiment_d_subgroup_summary.csv
```

Generated results, figures, predictions, and checkpoints are ignored by Git.
The output contract is documented in
[docs/publication_results.md](docs/publication_results.md).

## Command Surface

```text
imfuse check                 validate the release environment and data contract
imfuse prepare-data          stage raw inputs, labels, splits, and modality caches
imfuse train                 run the neural publication matrix
imfuse baseline-classical    run RF/XGBoost composition baselines
imfuse baseline-alignn       run pretrained ALIGNN readout + RF
imfuse dropout               evaluate inference-time modality dropout
imfuse split-ood             create composition-cluster/working-ion holdouts
imfuse subgroups             evaluate anion-family/working-ion subgroups
imfuse tables                collect publication summary tables
imfuse figures               generate publication figures
imfuse explain-*             run attribution and faithfulness audits
```

Run `imfuse --help` or `imfuse <command> --help` for options.

## Adapt IM-Fuse

Reusable code lives under `src/battery_fusion/`. Add descriptors in
`features/`, models in `models/` or `fusion/`, and evaluation protocols in
`experiments/`. The expected extension points and data format are described in
[docs/extend.md](docs/extend.md) and [docs/framework.md](docs/framework.md).

## License

Software and documentation are released under the
[BSD 3-Clause License](LICENSE). Data-derived metadata tracked under `data/`
is released under [CC BY 4.0](data/LICENSE). Third-party source records, CIFs,
and descriptors retain their original providers' terms and are not relicensed
by IM-Fuse.

## Citation And Release

Citation metadata are available in [CITATION.cff](CITATION.cff). The final
software and data DOIs will be added after Zenodo archival. Source-data and
artifact availability are described in
[docs/data_and_code_availability.md](docs/data_and_code_availability.md).
