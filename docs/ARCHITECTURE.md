# MuSAGE architecture

![MuSAGE model overview](../assets/musage_architecture.png)

This is the author-supplied architecture image, included unchanged. No redraw or image edits were applied. The feature builders and training code are the source of truth for model implementation details.

MuSAGE combines frozen sequence/structure priors with a trainable mutation-aware graph model:

1. **Interface graph:** Cα residues within 12 Å of a mutation site form a residue graph. Node features encode amino-acid identity, partner identity, and mutation state; edges use a 10 Å cutoff and radial distance encoding.
2. **Structure branch:** a four-layer E(3)-equivariant graph network with 128 hidden units produces the structural state.
3. **Frozen priors:** ESM-2 t33 sequence mutation-site features (1280 dimensions), local ESM-2 Δ features, 59-dimensional local geometry descriptors, ESM-IF1 site embeddings (512 dimensions), and GearNet-Edge site embeddings (3072 dimensions) are projected and fused into the structural representation.
4. **Mutation interaction:** a pairwise epistasis MLP and four-head mutation-set self-attention add context for multi-site variants. Pairwise and higher-order terms are zero for a single-site variant.
5. **Sequence–structure gate:** a learned channel-wise gate combines sequence and structural states.
6. **Antisymmetric readout:** one shared scoring function is evaluated for forward and reversed mutation states; the output is `0.5 × (f(WT→MT) − f(MT→WT))`. This enforces sign reversal when mutation direction is swapped, subject to deterministic evaluation.

The feature builders and training code are the source of truth for tensor alignment and shapes. Every cache is keyed by stable sample IDs and mutation sites must map to the same residue indexing in the prepared PDB, graph, and sequence features.

## Scope of released weights

The Zenodo package contains the 50 fold-specific SKEMPI_CL checkpoints. They are cross-validation models, not one model trained on the complete benchmark. For each benchmark sample, use the fold-specific model from each of the five seeds and average those five predictions. Do not average all 50 models for every validation sample because that would include models trained on that sample's fold.
