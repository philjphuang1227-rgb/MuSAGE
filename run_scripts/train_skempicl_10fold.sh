#!/usr/bin/env bash
set -euo pipefail

SAMPLES=data/processed/SKEMPI/samples_ready.csv
GRAPHS=data/processed/SKEMPI/structure_graphs_r12.pt
FOLDS=data/processed/SKEMPI/random_complex_10fold_seed20260916
ESM=data/esm_cache/SKEMPI_esm2_t33_650M_UR50D.npz
LOCAL_DELTA=data/esm_cache/SKEMPI_local_delta_r12_s5.npz
GEOMETRY=data/processed/SKEMPI/local_geometry_r12.npz
ESM_IF1=data/esm_if1_cache/SKEMPI_esm_if1_mutation_sites.npz
GEARNET=data/gearnet_cache/SKEMPI_gearnet_mc_edge_sites.npz

for seed in 20260911 20260912 20260913 20260914 20260915; do
  python scripts/train_graph_model.py \
    --graphs "$GRAPHS" --samples "$SAMPLES" --fold-dir "$FOLDS" \
    --esm-features "$ESM" --local-delta-features "$LOCAL_DELTA" \
    --geometry-features "$GEOMETRY" \
    --pretrained-structure-features "$ESM_IF1" \
    --auxiliary-pretrained-structure-features "$GEARNET" \
    --mode residual_gated --hidden 128 --layers 4 --dropout 0.1 \
    --batch-size 16 --accumulation-steps 1 \
    --learning-rate 3e-4 --weight-decay 1e-4 \
    --max-epochs 100 --patience 12 --min-refit-epochs 8 \
    --antisymmetric --pairwise-epistasis --higher-order-epistasis \
    --num-folds 10 --seed "$seed" \
    --output-dir "outputs/SKEMPI_CL/seed${seed}"
done
