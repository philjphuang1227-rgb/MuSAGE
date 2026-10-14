# Data provenance

## Included training data

The repository contains the SKEMPI v2.0 source table, the WT structures used by the prepared benchmark samples, ready-to-train sample tables, and fixed fold assignments under [`datasets/data/`](../datasets/README.md). The preparation report records the original source and row counts with cluster-specific paths removed. The source is also available from the [official SKEMPI v2.0 database](https://life.bsc.es/pid/skempi2/database/index).

S4169, S1131, and M1707 source fold CSVs and prepared sample/fold tables are included under `datasets/data/`; their upstream project is [DSSA-PPI](https://github.com/Agitw/DSSA-PPI). Fixed split files and homology metadata are included so the benchmark definitions can be inspected and reused.

## Included external validation data

AB-Bind and AbDesign v1/v2 source tables, the required WT structures, processed sample tables, and deduplication audits are included under `datasets/external_data/`. Upstream sources:

- [AB-Bind-Database](https://github.com/sarahsirin/AB-Bind-Database)
- [AbDesign](https://staging.naturalantibody.com/ab-design/)
- [AbDesign v2](https://staging.naturalantibody.com/ab-design2/)

These are frozen external evaluations; their labels were not used for model fitting or model selection. Dataset terms continue to be governed by the respective upstream sources. The AbDesign tables contain affinity-ratio-derived proxy labels; see `datasets/external_data/EXTERNAL_DATA_MANIFEST.json`.

## Pretrained feature models

- **ESM-2:** t33 650M UR50D via [fair-esm](https://github.com/facebookresearch/esm).
- **ESM-IF1:** GVP4 t16 142M via [fair-esm](https://github.com/facebookresearch/esm).
- **GearNet-Edge:** Multiview Contrast checkpoint from [GearNet](https://github.com/DeepGraphLearning/GearNet); the project links its weights on [Zenodo](https://zenodo.org/records/7723075).

The upstream model weights are not redistributed. Feature-generation commands identify the expected model variants and write generated caches under `data/`, which is ignored by Git.

## Paths and checksums

Sample-table `pdb_path` values are repository-relative, such as `datasets/data/raw/pdb/1A4Y.pdb`; no cluster-specific absolute path is required. `datasets/SHA256SUMS` covers the included dataset files.
