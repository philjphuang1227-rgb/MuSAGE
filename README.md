# MuSAGE

<p align="center">
  <img src="assets/musage_architecture.png" alt="MuSAGE architecture" width="100%">
</p>

MuSAGE predicts mutation-associated changes in protein–protein binding affinity from sequence and structure inputs.

## Getting started

### Requirements

- Python 3.10
- PyTorch 2.x; CUDA 12.1 is recommended for GPU training
- NumPy, SciPy, scikit-learn, fair-esm
- TorchDrug 0.2.1 for GearNet-Edge feature generation

### Installation

```bash
git clone https://github.com/philjphuang1227-rgb/MuSAGE.git
cd MuSAGE
conda env create -f environment.yml
conda activate musage
pip install -e .
```

## Datasets

Training benchmark inputs, fixed split files, and the external-validation inputs are included under [`datasets/`](datasets/README.md), following the data-first layout used by DSSA-PPI. The included tables and structures are ready to use; their checksums and upstream sources are listed in the dataset guide.

To stage the released SKEMPI sample table and fold assignments for the feature/training scripts:

```bash
bash run_scripts/prepare_skempicl_features.sh
```

The feature-generation commands then build graph and pretrained sequence/structure features locally. The staging script does not download or redistribute upstream pretrained model weights. See [data provenance](docs/DATA_PROVENANCE.md) and [reproduction steps](docs/REPRODUCIBILITY.md).

## Training and evaluation

Run the five-seed, ten-fold SKEMPI complex-level training protocol:

```bash
bash run_scripts/train_skempicl_10fold.sh
```

Aggregate fold-matched out-of-fold predictions for the five seeds:

```bash
bash run_scripts/aggregate_skempicl_oof.sh
```

External validation inputs and scoring notes are in [`validation/`](validation/README.md). The external sets are frozen evaluation sets and were not used for training or model selection.

## Model weights and results

The 50 final SKEMPI fold checkpoints, fold predictions, and run summaries are packaged for Zenodo. The Zenodo link will be added here after the draft upload is complete. Checkpoint identities and integrity instructions are documented in [`zenodo/README.md`](zenodo/README.md).

## Architecture

The author-provided architecture figure is included unchanged at [assets/musage_architecture.png](assets/musage_architecture.png). See [architecture notes](docs/ARCHITECTURE.md) and [reproduction steps](docs/REPRODUCIBILITY.md) for model inputs and training details.

## License

A project license has not been selected. Dataset and upstream model terms remain those of their respective sources.

