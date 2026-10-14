#!/usr/bin/env bash
set -euo pipefail

# Stage the released sample table and the exact complex-level folds.
mkdir -p data/processed/SKEMPI/random_complex_10fold_seed20260916
cp datasets/data/processed/SKEMPI/samples_ready.csv data/processed/SKEMPI/samples_ready.csv
cp -R datasets/data/processed/SKEMPI/random_complex_10fold_seed20260916/. \
  data/processed/SKEMPI/random_complex_10fold_seed20260916/
