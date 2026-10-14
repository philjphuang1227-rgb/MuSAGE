#!/usr/bin/env bash
set -euo pipefail

: "${ESM_IF1_CHECKPOINT:?Set ESM_IF1_CHECKPOINT to the official ESM-IF1 GVP4 t16 checkpoint}"
: "${GEARNET_CHECKPOINT:?Set GEARNET_CHECKPOINT to upstream mc_gearnet_edge.pth}"

mkdir -p data/processed/SKEMPI data/esm_cache data/esm_if1_cache/sites data/gearnet_cache/sites

python scripts/prepare_skempi_full.py \
  --csv data/raw/SKEMPI2.csv --pdb-dir data/raw/pdb \
  --output-dir data/processed/SKEMPI
python scripts/filter_ready_samples.py \
  --input data/processed/SKEMPI/samples.csv \
  --output data/processed/SKEMPI/samples_ready.csv
python scripts/cache_structure_graphs.py \
  --samples data/processed/SKEMPI/samples_ready.csv \
  --output data/processed/SKEMPI/structure_graphs_r12.pt \
  --radius 12 --edge-radius 10
python scripts/cache_esm_features.py \
  --samples data/processed/SKEMPI/samples_ready.csv \
  --output data/esm_cache/SKEMPI_esm2_t33_650M_UR50D.npz \
  --model esm2_t33_650M_UR50D --layer 33
python scripts/cache_local_delta.py \
  --samples data/processed/SKEMPI/samples_ready.csv \
  --output data/esm_cache/SKEMPI_local_delta_r12_s5.npz \
  --model esm2_t33_650M_UR50D --layer 33 --radius 12 --sigma 5
python scripts/cache_local_geometry.py \
  --graphs data/processed/SKEMPI/structure_graphs_r12.pt \
  --output data/processed/SKEMPI/local_geometry_r12.npz
python scripts/cache_esm_if1_features.py \
  --samples data/processed/SKEMPI/samples_ready.csv \
  --checkpoint "$ESM_IF1_CHECKPOINT" \
  --output data/esm_if1_cache/SKEMPI_esm_if1_mutation_sites.npz \
  --site-cache data/esm_if1_cache/sites
python scripts/cache_gearnet_features.py \
  --samples data/processed/SKEMPI/samples_ready.csv \
  --checkpoint "$GEARNET_CHECKPOINT" \
  --output data/gearnet_cache/SKEMPI_gearnet_mc_edge_sites.npz \
  --site-cache data/gearnet_cache/sites
python scripts/build_random_folds.py \
  --samples data/processed/SKEMPI/samples_ready.csv \
  --output-dir data/processed/SKEMPI/random_complex_10fold_seed20260916 \
  --num-folds 10 --seed 20260916 --split-level complex-level
