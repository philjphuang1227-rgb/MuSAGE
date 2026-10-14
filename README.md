<p align="center">
  <img src="assets/musage_architecture.png" alt="MuSAGE architecture" width="100%">
</p>

# MuSAGE

**Mutation-Aware Sequence–Structure Antisymmetric Gated Estimator** for predicting mutation-induced changes in protein–protein binding free energy (ΔΔG).

This repository contains the paper-aligned model implementation and the steps to reproduce its SKEMPI v2.0 complex-level cross-validation results. The final paper model name is MuSAGE; the earlier development repository remains available as `Philjphuang/AntiMut`.

## Paper

Juping Huang, Zhenjia Gan, Wenzhen Li, Tianyun Liu, Lu Xie, and John Z. H. Zhang. **MuSAGE: Predicting Mutation Induced Binding Free Energy Changes through Gated Sequence-Structure Integration and Antisymmetric Learning.**

The model reports ΔΔG in kcal mol⁻¹ using `RT ln(Kd_WT / Kd_mut)`. Under this sign convention, positive values indicate stronger mutant binding (a lower mutant Kd).

## Main requirements

- Python 3.10
- PyTorch 2.x and CUDA 12.1 for GPU training (CPU is supported for small jobs)
- NumPy, SciPy, scikit-learn, and fair-esm
- TorchDrug 0.2.1 only when regenerating GearNet-Edge features

## Install

```bash
git clone https://github.com/philjphuang1227-rgb/MuSAGE.git
cd MuSAGE
conda env create -f environment.yml
conda activate musage
pip install -e .
```

## Data and pretrained features

Download the SKEMPI v2.0 CSV and cleaned complex structures from the [official SKEMPI download page](https://life.bsc.es/pid/skempi2/database/index). Place the CSV at `data/raw/SKEMPI2.csv` and PDB files in `data/raw/pdb/`. Do not commit benchmark files or generated feature caches; see [data provenance and setup](docs/DATA_PROVENANCE.md).

Prepare samples, graphs, folds, and frozen features by following [the reproducibility guide](docs/REPRODUCIBILITY.md). The GearNet-Edge and ESM-IF1 feature builders require their upstream pretrained checkpoints. The model itself consumes cached features and does not redistribute third-party weights.

## Train and evaluate

```bash
bash run_scripts/train_skempicl_10fold.sh
```

The script trains five seeds (`20260911`–`20260915`) over the same ten complex-level folds and writes fold checkpoints, held-out predictions, and metrics under `outputs/SKEMPI_CL/`. To run one fold first or rebuild all preprocessing features, see [the complete commands](docs/REPRODUCIBILITY.md).

To aggregate the five seed-specific out-of-fold predictions (matching samples by ID and fold):

```bash
bash run_scripts/aggregate_skempicl_oof.sh
```

## Final checkpoints and computed results

The 50 final fold checkpoints (5 seeds × 10 folds), 50 held-out prediction files, five metric reports, mutation-stratified results, configuration, and SHA-256 manifest are packaged in the Zenodo model/results archive (record link will be added after the upload draft is created). After downloading:

```bash
sha256sum -c MuSAGE_SKEMPI_CL_weights_and_results.zip.sha256
unzip MuSAGE_SKEMPI_CL_weights_and_results.zip
```

The checkpoints are cross-validation fold models. To reproduce benchmark predictions, use the held-out predictions or the checkpoint for the corresponding fold from each seed; averaging all 50 checkpoints for every validation sample would mix models trained on different folds. For a new deployment dataset, train with a declared split appropriate to that dataset.

## Architecture

See [the model architecture and tensor conventions](docs/ARCHITECTURE.md). The author-supplied architecture image is included unchanged at [assets/musage_architecture.png](assets/musage_architecture.png).

## License

No project license has been selected yet. The repository and model weights are therefore not granted an open-source reuse license at this time. Dataset and upstream model terms remain those of their respective sources.

## Citation

Please cite the MuSAGE manuscript above and cite the Zenodo record linked in this README when using the released checkpoints or prediction outputs.
