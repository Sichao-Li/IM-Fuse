# Data And Code Availability

- **Data:** Stable sample identifiers, target-specific metadata, exact split
  assignments for seeds 0-4, and checksums are included in this repository.
  The source battery dataset is publicly available at
  <https://github.com/ZoraZhuang/battery_datasets>. The larger `mp_total.csv`,
  CIF collection, atom initializer, and processed modality caches are staged
  from that source via `imfuse prepare-data`; see
  [data_preparation.md](data_preparation.md). Externally sourced Materials
  Project records remain subject to their original terms.
- **Code:** IM-Fuse is available at <https://github.com/Sichao-Li/IM-Fuse>
  and archived on Zenodo under DOI
  [10.5281/zenodo.21440948](https://doi.org/10.5281/zenodo.21440948). The
  tagged archive is the version of record.
- **Additional information:** Installation, data preparation, full experiment
  commands, output locations, and adaptation guidance are provided in this
  repository. Pretrained ALIGNN is an optional external dependency and is
  identified separately from models trained within IM-Fuse.

Associated article:

> Li, S., Zhu, T., Deng, W., Zhuang, Z., Xu, X., Wei, Y., Barnard, A. S., &
> Butler, K. T. Multimodal fusion of complementary material representations
> for generalizable and interpretable property prediction. *Cell Reports
> Physical Science* (2026).
