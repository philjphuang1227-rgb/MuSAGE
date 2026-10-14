# External validation

The repository includes frozen external-evaluation inputs for AB-Bind, AbDesign v1, and AbDesign v2 under `datasets/external_data/`. Each set contains its source measurement table, processed sample rows, audit report, and required WT PDB files. The sample tables contain repository-relative `pdb_path` entries.

These data were not used for training or model selection. Labels come from distinct source assays and transformations; see `datasets/external_data/EXTERNAL_DATA_MANIFEST.json` before comparing metrics.

To score a model's prediction CSV against one of the released sample tables, use:

```bash
python scripts/score_predictions.py --help
```

Generate features from the selected sample CSV with the feature builders in `scripts/`, then pass the prediction file and matching labels to the scorer. Do not mix these external samples into the SKEMPI cross-validation folds.
