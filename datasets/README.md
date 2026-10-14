# Datasets

This directory is reserved for the training and external-validation inputs used by MuSAGE. The benchmark files are being synchronized from the project cluster; this checkout currently contains setup instructions only.

## Benchmarks used by the release code

- **SKEMPI v2.0** — primary benchmark for the released complex-level ten-fold experiment. The raw affinity table and cleaned PDB structures are available from the [official SKEMPI download page](https://life.bsc.es/pid/skempi2/database/index).
- **S4169, S1131, and M1707** — benchmark names supported by `scripts/prepare_data.py` for training/validation data preparation. Their exact source files, split assignments, and provenance will be documented here when synchronized from the cluster.

## Reproduction workflow

Follow [the reproducibility guide](../docs/REPRODUCIBILITY.md) for preprocessing, fixed folds, feature generation, training, and evaluation. Keep generated graph and embedding caches outside version control; they can be recreated from the benchmark inputs and pretrained upstream models.
