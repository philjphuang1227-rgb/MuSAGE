# MuSAGE datasets

This directory follows the data-first organization used by the [DSSA-PPI repository](https://github.com/Agitw/DSSA-PPI): clone the repository, create the Conda environment, use the included benchmark inputs, then train and validate with the scripts in `run_scripts/` and `validation/`.

All table and structure files listed below are included in this repository. Generated graph/embedding caches and model outputs are not included. `SHA256SUMS` records the exact published dataset files.

## Directory layout

```text
datasets/
├── data/
│   ├── raw/
│   │   ├── SKEMPI2.csv
│   │   ├── benchmark_folds/{S4169,S1131,M1707}/
│   │   └── pdb/                         # WT structures and mapping files
│   └── processed/{SKEMPI,S4169,S1131,M1707}/
└── external_data/
    ├── AB-Bind-Database/                # source table and WT structures
    ├── AbDesign_v1/                     # source table and required WT structures
    ├── AbDesign_v2/                     # source affinity table
    └── processed/{AB-Bind,AbDesign_v1,AbDesign_v2}/
```

## Training and benchmark datasets

- **SKEMPI v2.0** — `data/raw/SKEMPI2.csv`, cleaned WT PDB structures, processed sample tables, ready-to-train rows, and the fixed ten-fold complex split used by the released experiment. The prepared table contains 6,798 ready samples. The raw table and structures can also be obtained from the [official SKEMPI v2.0 site](https://life.bsc.es/pid/skempi2/database/index).
- **S4169, S1131, M1707** — public DSSA-PPI benchmark fold files under `data/raw/benchmark_folds/`, processed sample tables, available random folds, and strict-homology split metadata. Source repository: [DSSA-PPI](https://github.com/Agitw/DSSA-PPI). The sample tables contain 4,160, 1,129, and 1,707 rows respectively.

The exact complex-level SKEMPI split for the final model is `data/processed/SKEMPI/random_complex_10fold_seed20260916/`. Its fold CSVs assign each sample to `train` or `val`; each complex remains in one fold. The separate `random_mutation_*` and `strict_homology_*` folders are alternate benchmark protocols, not the final SKEMPI split.

## External validation datasets

All three sets are evaluation-only. Their labels were not used for training or model selection.

- **AB-Bind** — source CSV, source WT structures, cleaned samples (851 ready rows), and deduplication audit. Upstream: [AB-Bind-Database](https://github.com/sarahsirin/AB-Bind-Database).
- **AbDesign v1** — source mutation table, the WT structures used by the released sample table, 654 ready samples, and deduplication audit. Upstream: [AbDesign](https://staging.naturalantibody.com/ab-design/).
- **AbDesign v2** — source affinity table, 5GGS-based WT structure from the included AbDesign v1 structures, 387 ready samples, and deduplication audit. Upstream: [AbDesign v2](https://staging.naturalantibody.com/ab-design2/).

The AbDesign labels are affinity-ratio-derived proxies as described in `external_data/EXTERNAL_DATA_MANIFEST.json`; they are not the same measurement type as SKEMPI or AB-Bind. Consult each upstream source for its data-use terms.

## Reproduction steps

1. Clone MuSAGE and create the environment using the commands in the root README.
2. The dataset files are already present in this folder; no separate data download is needed for the released SKEMPI training table or listed validation sets.
3. Run `bash run_scripts/prepare_skempicl_features.sh` from the repository root to stage the fixed SKEMPI table and fold files. Then follow the feature-generation commands in `docs/REPRODUCIBILITY.md` to create local caches.
4. Run the training/evaluation commands in [the reproduction guide](../docs/REPRODUCIBILITY.md).
5. Verify file integrity from the repository root:

   ```bash
   sha256sum -c datasets/SHA256SUMS
   ```

The cache builders download/load upstream pretrained ESM checkpoints and require the GearNet-Edge checkpoint separately. These third-party model weights and generated feature caches are intentionally not bundled here.


