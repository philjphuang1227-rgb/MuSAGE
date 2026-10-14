# MuSAGE model card

## Intended use

Research prediction of mutation-induced binding free-energy changes for protein–protein complexes represented by a local residue graph and frozen sequence/structure features. This release provides the source code, datasets, fixed folds, and configuration needed to generate features and train MuSAGE locally.

## Model details

- Name: MuSAGE (Mutation-Aware Sequence–Structure Antisymmetric Gated Estimator)
- Task: regression of binding ΔΔG in kcal/mol
- Inputs: WT complex PDB-derived graph, mutation set, ESM-2, local ESM delta, geometry, ESM-IF1, and GearNet-Edge features
- Outputs: signed ΔΔG estimate, as recorded in the corresponding dataset units
- Training protocol: 5 seeds × 10 complex-level folds; training generates 50 local fold checkpoints
- Release contents: code, datasets, configurations, and architecture figure; trained MuSAGE weights and raw experiment outputs are not distributed

## Evaluation

The training scripts generate per-fold held-out predictions and per-seed metrics. Follow [the reproduction guide](REPRODUCIBILITY.md) to train the models and aggregate these outputs. Use the matching fold model for each held-out sample so that the evaluated sample is excluded from that model's training data.

## Limitations

Predictions depend on SKEMPI label quality, mutation-to-PDB mapping, feature extraction versions, and the scope of the training benchmark. For comparisons, use the same dataset-specific target direction and units. Repeated training can produce numerical differences across hardware and software versions. Frozen upstream model weights remain governed by their original terms. A project license for the MuSAGE implementation has not been selected.

