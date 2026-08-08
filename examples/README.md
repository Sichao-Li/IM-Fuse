# Examples

- [`quickstart.py`](quickstart.py) — the minimal in-memory path for running
  IM-Fuse on new data: build the three modality features for a set of
  structures, train the intermediate-fusion model with the library training
  loop, and probe modality reliance with inference-time dropout. CPU-only, no
  downloads, no publication data contract; runs in a few minutes.

```bash
python examples/quickstart.py
```

The example predicts crystal density for synthetic rocksalt-type structures.
Density depends on both composition (atomic masses) and geometry (cell
volume), so it is a convenient sanity check that all three modalities carry
signal: expect test R^2 above 0.9 with all modalities present and a clearly
higher error when the structure modality is dropped.

To adapt it to real data, replace `build_toy_dataset` with a loop over your
own structures and targets, and widen `atom_init.json` to cover your elements
(or reuse the published CGCNN `atom_init.json`). The rest of the pipeline is
unchanged. See [docs/extend.md](../docs/extend.md) for the extension
checklist.
