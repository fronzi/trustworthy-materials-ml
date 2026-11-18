# Research plan (written before results were generated)

## Decision question
When can a materials ML model be trusted to prioritise experiments or calculations, and how should
uncertainty change a laboratory's next decision?

## Pre-specified hypotheses
| ID | Hypothesis | Test | Would be falsified by |
|---|---|---|---|
| H1 | Random splits overstate performance on new chemistries | MAE: leave-one-cluster-out vs Matbench 5-fold, all models | LOCO MAE not worse for most non-dummy models |
| H2 | Conformal calibration restores nominal coverage on random splits | 68/90/95% coverage | Coverage more than 3 points from nominal |
| H3 | Marginal conformal guarantees weaken under chemical shift | LOCO coverage vs nominal | LOCO coverage stays within 3 points of nominal |
| H4 | Greedy or UCB acquisition finds high-gap materials faster than random | Target recall, 10 paired seeds | 95% bootstrap interval of the paired difference includes zero |
| H5 | Uncertainty sampling lowers test MAE faster than random | Final test MAE, 10 paired seeds | 95% interval includes zero or favours random |
| H6 | Adaptive Ramsey control lowers frequency error for equal sensing time | Interpolated RMSE, 300 paired-seed trials | No reduction at the common time budget |

## Rules
- Datasets: Matbench v0.1 `expt_gap` and `steels`. Both are reported.
- Every model is run on every split. Results that weaken a hypothesis stay in the tables.
- Dummy-regressor MAE on `expt_gap` must reproduce the published Matbench value within 0.05 eV.
  This validates the data and split pipeline. Nothing is fitted to pass it.
- Quantum results are simulations. Device parameters are assumptions, not measurements.

## Milestones and risks
| Milestone | Exit criterion |
|---|---|
| Reproducible data and descriptors | Dummy benchmark passes |
| Model benchmark | Random and LOCO tables complete |
| Trust layer | Coverage and rank-correlation tables complete |
| Decision layer | Active-learning campaigns complete |
| Quantum extension | Master-equation and analytic checks pass |

| Risk | Mitigation |
|---|---|
| Composition duplicates inflate random-split scores | LOCO is reported beside the Matbench split |
| Conformal coverage misread as conditional | Reported separately under cluster shift |
| Toy quantum model is over-interpreted | Scope and missing noise sources are listed in the README |
