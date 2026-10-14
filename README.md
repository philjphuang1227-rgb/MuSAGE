<p align="center">
  <img src="assets/musage_architecture.png" alt="MuSAGE architecture" width="100%">
</p>

# MuSAGE

MuSAGE is a mutation-aware sequence–structure model for predicting changes in protein–protein binding affinity from mutations.

## Getting started with this repository

### Main requirements

- Python 3.10
- PyTorch 2.x with CUDA 12.1 for GPU training (CPU is supported for small jobs)
- NumPy, SciPy, scikit-learn, and fair-esm
- TorchDrug 0.2.1 when regenerating GearNet-Edge features

### Installation

```bash
git clone https://github.com/philjphuang1227-rgb/MuSAGE.git
cd MuSAGE
conda env create -f environment.yml
conda activate musage
pip install -e .
```

## Download data

The `datasets/` directory is reserved for the training and external-validation inputs used by the released experiments. See [datasets/README.md](datasets/README.md) for benchmark sources, the current data availability, and preparation details.

The current training protocol uses SKEMPI v2.0 with complex-level ten-fold cross-validation. Follow [the reproducibility guide](docs/REPRODUCIBILITY.md) to prepare structures, features, and folds.

## Training and evaluation

Run the five-seed, ten-fold SKEMPI complex-level experiment:

```bash
bash run_scripts/train_skempicl_10fold.sh
```

Aggregate the five seed-specific out-of-fold predictions:

```bash
bash run_scripts/aggregate_skempicl_oof.sh
```

The complete commands, feature-generation steps, fold handling, and evaluation protocol are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Model weights and results

The released fold checkpoints and computed results are packaged separately from the source code. The Zenodo record link will be added after the upload is complete. Use the held-out prediction files or the checkpoint matching each sample's held-out fold when reproducing benchmark metrics.

## Architecture

The author-provided architecture diagram is included unchanged at [assets/musage_architecture.png](assets/musage_architecture.png). Implementation details and tensor conventions are described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

No project license has been selected. Dataset and upstream model terms remain those of their respective sources.
