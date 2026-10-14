# Released SKEMPI_CL results

The values below are the arithmetic mean and sample standard deviation across five seed-specific ten-fold means. The Zenodo archive contains every seed/fold metric and all held-out prediction rows. This is the original five-seed, ten-fold complex-level evaluation; it is not a full-data deployment score.

| Metric | Mean across seeds | Sample SD across seeds |
|---|---:|---:|
| Pearson r | 0.6572 | 0.0077 |
| Spearman ρ | 0.5740 | 0.0088 |
| RMSE (kcal/mol) | 1.5496 | 0.0152 |
| MAE (kcal/mol) | 1.0805 | 0.0063 |
| AUROC | 0.7509 | 0.0024 |

For each seed, the table first averages its ten fold metrics. It then reports the mean and sample SD of the five seed means. Pooled held-out metrics and mutation-count strata are available in the Zenodo archive. For comparisons across datasets, follow the target direction and units defined by the source dataset.

