# Public Test Scope

The public test suite is intentionally small. It checks reusable package
behavior that should remain stable for users who adapt IM-Fuse to new battery
datasets:

- data staging and split creation,
- chemistry group assignment,
- composition/RDF/structure feature builders,
- fusion dataset collation,
- model and training smoke tests,
- classical composition baselines,
- OOD split creation, subgroup metrics, and modality-dropout utilities,
- core attribution and faithfulness helpers.

It does not try to regression-test every manuscript figure, summary table, or
full training run. Those outputs are reproduced through the documented `imfuse`
commands and the end-to-end scripts.
