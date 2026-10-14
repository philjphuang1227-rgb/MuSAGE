# MuSAGE model card

## Intended use

Research prediction of mutation-induced binding free-energy changes for protein–protein complexes represented by the released residue graph and frozen sequence/structure features. The supplied checkpoints are for reproducing SKEMPI v2.0 complex-level cross-validation.

## Model details

- Name: MuSAGE (Mutation-Aware Sequence–Structure Antisymmetric Gated Estimator)
- Task: regression of binding ΔΔG in kcal/mol
- Inputs: WT complex PDB-derived graph, mutation set, ESM-2, local ESM delta, geometry, ESM-IF1, and GearNet-Edge features
- Outputs: signed ΔΔG estimate, using `RT ln(Kd_WT / Kd_mut)`
- Checkpoints: 50 fold models, 5 seeds × 10 complex-level folds, stored in the linked Zenodo archive

## Evaluation

The archive includes per-fold held-out predictions, per-seed metrics, and the mutation-stratified aggregate. Use the matching fold model for each held-out sample. The folds are not independent deployment models for arbitrary samples.

## Limitations

Predictions depend on SKEMPI label quality, mutation-to-PDB mapping, feature extraction versions, and the scope of the training benchmark. The model's sign convention must be checked before comparing with tools that report the opposite ΔΔG sign. Frozen upstream model weights remain governed by their original terms. This release does not include a license grant for the MuSAGE implementation or checkpoints.
