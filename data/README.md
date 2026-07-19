# Released Data Contract

Git tracks only the compact identifiers and exact publication split tables.
Raw Materials Project-derived records, CIF files, atom descriptors, and
generated tensor caches are intentionally distributed separately because they
are larger and may retain source-specific terms.

## Tracked Files

| Path | Contents |
| --- | --- |
| `sample_order/sample_order_keep_last.csv` | Ordered, unique `id_discharge` values after keeping the last source row per ID. |
| `splits/publication/{target}/seed_{0..4}/{train,val,test}.csv` | Exact model-ready 80/10/10 assignments and target-specific metadata. |
| `splits/publication/{target}/seed_{0..4}/split_config.json` | Seed, target, ratios, and split counts. |
| `checksums.sha256` | SHA-256 checksums for every tracked scientific data file. |

The model-ready intersection contains 8,088 unique discharge IDs. Each seed
contains 6,470 training, 808 validation, and 810 test samples. Membership is
identical across the two targets, and the partitions are disjoint.

The manuscript's 10,114 count refers to the cleaned row-level source table
before discharge-ID deduplication. Repeated `id_discharge` values are expected
at that stage. The released learning contract then keeps the last source row
per ID and intersects samples available in all three modalities, producing the
8,088 unique IDs represented by the tracked splits. These counts describe
different stages and should not be used interchangeably.

## Split Table Fields

| Field | Type | Unit | Description |
| --- | --- | --- | --- |
| `sample_id` | string | none | Stable discharge-material identifier. |
| `formula` | string | none | Discharged composition used for composition descriptors and chemistry grouping. |
| `working_ion` | string | none | Working-ion label from the source record. |
| `anion_family` | string | none | Approximate rule-based chemistry group used for evaluation. |
| `target` | float | V or mAh cm^-3 | Average voltage for `average_voltage`; volumetric capacity for `capacity_vol`. |

## Provenance And Processing

The external cleaned `mp_total.csv` contains 10,114 rows and may contain
repeated `id_discharge` values. The publication pipeline retains the last row
for each ID, intersects IDs available in composition, CIF-derived RDF, and
graph representations, and then applies the tracked seed-specific assignments.
Composition clusters are fitted without target values. Target transformations
are disabled in the retained release.

The exact source-cleaning operation that produces the 10,114-row table belongs
to the upstream data release. IM-Fuse starts from that cleaned table and does
not infer or silently reapply undocumented physical-outlier rules.

Run `imfuse check` to verify checksums, schema, split disjointness, seed
coverage, deterministic membership, and cross-target alignment. Add
`--strict-artifacts` when raw inputs and processed caches have been downloaded.

Source-data provenance and redistribution terms must accompany the external
data archive. Materials Project records remain subject to the source provider's
terms; this repository does not relicense those source records.
