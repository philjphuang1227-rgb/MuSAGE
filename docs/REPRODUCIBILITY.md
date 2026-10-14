# Reproduce the SKEMPI_CL model

This guide reproduces the released five-seed, ten-fold complex-level protocol. It assumes a Linux environment with a CUDA-capable GPU. The release environment targets Python 3.10, PyTorch 2.x, and CUDA 12.1; training uses one GPU, batch size 16, and gradient clipping at 5.

## 1. Install

```bash
conda env create -f environment.yml
conda activate musage
pip install -e .
```

To regenerate GearNet features, also install TorchDrug 0.2.1 and the upstream dependencies described by [GearNet](https://github.com/DeepGraphLearning/GearNet). Checkpoint `mc_gearnet_edge.pth` is available from the [upstream Zenodo record](https://zenodo.org/records/7723075). The core training script consumes cached features and does not require TorchDrug after feature generation.

## 2. Prepare SKEMPI samples

Download SKEMPI v2.0 and cleaned structures as described in [DATA_PROVENANCE.md](DATA_PROVENANCE.md). Then run:

```bash
python scripts/prepare_skempi_full.py \
  --csv data/raw/SKEMPI2.csv \
  --pdb-dir data/raw/pdb \
  --output-dir data/processed/SKEMPI

python scripts/filter_ready_samples.py \
  --input data/processed/SKEMPI/samples.csv \
  --output data/processed/SKEMPI/samples_ready.csv
```

The CSV label convention is `ΔΔG = RT ln(Kd_WT / Kd_mut)` in kcal/mol. The structure parser verifies the WT residue at each mutation site. The ready-sample file must have unique `sample_id` values, nonempty `complex_id`, numeric `ddg`, valid `mutation_sites_json`, and valid PDB paths.

## 3. Build graph and frozen feature inputs

The commands below use the same settings and feature modalities as the released checkpoints. ESM-2 weights are downloaded through fair-esm. Provide local upstream checkpoint paths for ESM-IF1 and GearNet-Edge.

```bash
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
  --checkpoint /path/to/esm_if1_gvp4_t16_142M_UR50.pt \
  --output data/esm_if1_cache/SKEMPI_esm_if1_mutation_sites.npz \
  --site-cache data/esm_if1_cache/sites

python scripts/cache_gearnet_features.py \
  --samples data/processed/SKEMPI/samples_ready.csv \
  --checkpoint /path/to/mc_gearnet_edge.pth \
  --output data/gearnet_cache/SKEMPI_gearnet_mc_edge_sites.npz \
  --site-cache data/gearnet_cache/sites
```

Build the shared complex-level ten-fold assignment:

```bash
python scripts/build_random_folds.py \
  --samples data/processed/SKEMPI/samples_ready.csv \
  --output-dir data/processed/SKEMPI/random_complex_10fold_seed20260916 \
  --num-folds 10 --seed 20260916 --split-level complex-level
```

## 4. Train the five seeds

The released architecture is residual-gated, with a 128-unit, four-layer EGNN; dropout 0.1; AdamW (`lr=3e-4`, `weight_decay=1e-4`); batch size 16; Huber loss with δ=1; maximum 100 epochs; validation early stopping patience 12; minimum refit 8 epochs; and gradient norm clipping 5. Exact antisymmetry, pairwise epistasis, and higher-order epistasis are enabled. Other auxiliary losses/features are disabled.

Run one fold first to confirm the external datasets and cached feature files are in place:

```bash
python scripts/train_graph_model.py \
  --graphs data/processed/SKEMPI/structure_graphs_r12.pt \
  --samples data/processed/SKEMPI/samples_ready.csv \
  --fold-dir data/processed/SKEMPI/random_complex_10fold_seed20260916 \
  --esm-features data/esm_cache/SKEMPI_esm2_t33_650M_UR50D.npz \
  --local-delta-features data/esm_cache/SKEMPI_local_delta_r12_s5.npz \
  --geometry-features data/processed/SKEMPI/local_geometry_r12.npz \
  --pretrained-structure-features data/esm_if1_cache/SKEMPI_esm_if1_mutation_sites.npz \
  --auxiliary-pretrained-structure-features data/gearnet_cache/SKEMPI_gearnet_mc_edge_sites.npz \
  --mode residual_gated --hidden 128 --layers 4 --dropout 0.1 \
  --batch-size 16 --learning-rate 3e-4 --weight-decay 1e-4 \
  --max-epochs 100 --patience 12 --min-refit-epochs 8 \
  --antisymmetric --pairwise-epistasis --higher-order-epistasis \
  --num-folds 10 --fold 1 --seed 20260911 \
  --output-dir outputs/SKEMPI_CL/seed20260911
```

The full protocol is run with:

```bash
bash run_scripts/train_skempicl_10fold.sh
```

Each seed output includes `fold_1/` through `fold_10/`, with `model.pt` and held-out `predictions.csv`, plus a seed-level `metrics.json`. Five seeds produce 50 checkpoints and 50 fold prediction files.

## 5. Aggregate held-out predictions and metrics

```bash
bash run_scripts/aggregate_skempicl_oof.sh
```

The aggregation script verifies identical sample IDs, fold IDs, and labels across seeds, then averages the five seed predictions for each held-out sample and computes fold-level and pooled metrics. `scripts/score_predictions.py` can score individual prediction CSVs; `scripts/summarize_multiseed.py` summarizes seed-level reports.

## 6. Checkpoint/result archive

The Zenodo archive contains the released 50 fold checkpoints, original out-of-fold prediction CSVs, seed metrics, mutation-stratified summary, configuration, and checksum manifest. It does not contain SKEMPI CSV/PDB inputs or generated ESM/geometry caches. Follow `SHA256SUMS` in the archive to verify every contained file after extraction.

For a held-out SKEMPI example in fold *k*, use `fold_k/model.pt` from each of the five seeds. The all-fold checkpoint set is not a single full-data deployment model. A production model for new samples requires training on an explicitly declared training set and preserving the same feature-generation pipeline.
