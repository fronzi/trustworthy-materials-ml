# Decision log

| Decision | Reason | Effect on interpretation |
|---|---|---|
| Use the official Matbench outer split | Allows a known-answer check against published dummy results | Comparable random-split numbers |
| Keep hyperparameters fixed | The aim is reliable evaluation, not leaderboard tuning | Scores are not state of the art |
| Add five-cluster hold-out validation | Random folds can contain close chemical relatives | Gives a harder estimate with high fold-to-fold variance |
| Report raw, absolute conformal and normalised conformal intervals | Shows what calibration changes | No method is presented as universally best |
| Fixed the Lindblad readout sign after a physics check | Analytic and numerical signals disagreed | Corrected before results were generated |
| Increased simulated shots from 120 to 200 after the first run | Four adaptive trials did not reach the longest time budget | All 300 paired trials now reach all budgets |
| Added paired bootstrap ratios for quantum RMSE | Mean RMSE curves alone hide trial-to-trial uncertainty | Inconclusive 200 µs result is visible |
