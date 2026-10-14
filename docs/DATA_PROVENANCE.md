# Data and pretrained-model provenance

## SKEMPI v2.0

The primary benchmark is SKEMPI v2.0. Its mutation/affinity table and cleaned PDB structures are available from the [official SKEMPI v2.0 download page](https://life.bsc.es/pid/skempi2/database/index). The source table is semicolon-delimited and includes fields such as `Mutation(s)_cleaned` or `Mutation(s)_PDB`, `Affinity_wt`, `Affinity_mut`, and optional `Temperature`.

The release preparation code parses complex structures, checks each wild-type amino acid against its PDB residue, records stable sample IDs, and derives the training target. Training and external-validation benchmark inputs are being synchronized from the project cluster into [`datasets/`](../datasets/README.md). Until that sync is complete, use each benchmark's upstream source and the commands in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## ESM-2

The model uses the ESM-2 t33 650M UR50D model via [fair-esm](https://github.com/facebookresearch/esm). The feature scripts download/load the model as required. Record its version and layer (33) when regenerating features. ESM model weights are upstream artifacts and are not included in the MuSAGE repository or Zenodo archive.

## ESM-IF1

The structural sequence prior is ESM-IF1 GVP4 t16 142M, from the same [fair-esm project](https://github.com/facebookresearch/esm). The feature-generation script takes the official checkpoint path explicitly and caches residue/site features. Do not replace this checkpoint with a different ESM-IF1 variant when reproducing the released run.

## GearNet-Edge

The auxiliary structural prior uses the official GearNet-Edge Multiview Contrast checkpoint. The upstream [GearNet project](https://github.com/DeepGraphLearning/GearNet) links the [pretrained weights on Zenodo](https://zenodo.org/records/7723075), including `mc_gearnet_edge.pth`. The feature builder records the upstream model commit `28bf75f5dfb6cd7b9bf4394d153a87f69bddf1a6`. Install the matching TorchDrug/GearNet dependencies separately; these third-party weights and sources are not redistributed by MuSAGE.

## Generated inputs

Graph and feature caches are derived data and remain local under `data/processed/`, `data/esm_cache/`, `data/esm_if1_cache/`, and `data/gearnet_cache/`. They are excluded from GitHub and the Zenodo model/results archive. Regeneration commands and expected schemas are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
