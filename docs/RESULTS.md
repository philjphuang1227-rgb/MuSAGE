# SKEMPI_CL evaluation summary

The values below summarize the original five-seed, ten-fold complex-level evaluation as the arithmetic mean and sample standard deviation across five seed-specific ten-fold means. Reproduce the evaluation by generating features, training the five seeds, and aggregating their held-out predictions with [the reproduction guide](REPRODUCIBILITY.md). Original checkpoint and raw result files are not distributed.

| Metric | Mean across seeds | Sample SD across seeds |
|---|---:|---:|
| Pearson r | 0.6572 | 0.0077 |
| Spearman ρ | 0.5740 | 0.0088 |
| RMSE (kcal/mol) | 1.5496 | 0.0152 |
| MAE (kcal/mol) | 1.0805 | 0.0063 |
| AUROC | 0.7509 | 0.0024 |

For each seed, the table first averages its ten fold metrics. It then reports the mean and sample SD of the five seed means. The training and aggregation scripts write your new fold-level and pooled held-out metrics under `outputs/SKEMPI_CL/`. For comparisons across datasets, follow the target direction and units defined by the source dataset, and compare metrics calculated with the same aggregation method.

