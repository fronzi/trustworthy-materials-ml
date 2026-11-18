"""Render README.md from committed result tables so reported numbers cannot drift."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

R = Path("results")
MODELS = {
    "dummy": "Dummy mean",
    "ridge": "Ridge",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
    "mlp_ensemble": "MLP ensemble",
    "kan_ensemble": "KAN ensemble",
}
PUBLISHED_DUMMY, PUBLISHED_RF, PUBLISHED_MODNET = 1.1435, 0.4461, 0.3327


def table(frame: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in frame.to_numpy()]
    return "\n".join([head, rule, *rows])


b = pd.read_csv(R / "benchmark_summary.csv")
rows = []
for dataset, unit, digits in (("expt_gap", "eV", 3), ("steels", "MPa", 1)):
    part = b.loc[b["dataset"].eq(dataset)].set_index(["split", "model"])
    for model, name in MODELS.items():
        random, loco = part.loc[("random_5fold", model)], part.loc[("loco_5cluster", model)]
        rows.append(
            {
                "Task": f"{dataset} ({unit})",
                "Model": name,
                "Random 5-fold MAE": f"{random['mae_mean']:.{digits}f} ± {random['mae_std']:.{digits}f}",
                "Leave-one-cluster-out MAE": f"{loco['mae_mean']:.{digits}f} ± {loco['mae_std']:.{digits}f}",
                "Ratio": f"{loco['mae_mean'] / random['mae_mean']:.2f}",
            }
        )
bench = pd.DataFrame(rows)
gap = b.loc[b["dataset"].eq("expt_gap")].set_index(["split", "model"])
dummy = gap.loc[("random_5fold", "dummy"), "mae_mean"]
best_model = gap.loc["random_5fold"].drop(index="dummy")["mae_mean"].idxmin()
best = gap.loc[("random_5fold", best_model), "mae_mean"]
learned = [m for m in MODELS if m != "dummy"]
ratio_all = b.pivot_table(index=["dataset", "model"], columns="split", values="mae_mean")
ratio_all = (ratio_all["loco_5cluster"] / ratio_all["random_5fold"]).reset_index(name="ratio")
ratio_all = ratio_all.loc[ratio_all["model"].isin(learned)]

u = pd.read_csv(R / "uncertainty_summary.csv")
view = u.loc[u["dataset"].eq("expt_gap") & u["nominal"].eq(0.9)]
cov_rows = []
for model in ("random_forest", "mlp_ensemble", "kan_ensemble"):
    item = {"Model": MODELS[model]}
    for split, label in (("random_5fold", "Random"), ("loco_5cluster", "Cluster shift")):
        for method, short in (("raw_gaussian", "raw"), ("normalised_conformal", "conformal")):
            mask = view["model"].eq(model) & view["split"].eq(split) & view["method"].eq(method)
            value = view.loc[mask].iloc[0]
            item[f"{label}: {short}"] = f"{value['coverage_mean']:.1%}"
    cov_rows.append(item)
conformal = u.loc[u["method"].str.contains("conformal")].copy()
conformal["gap"] = (conformal["coverage_mean"] - conformal["nominal"]).abs()
random_gap = conformal.loc[conformal["split"].eq("random_5fold"), "gap"]
loco_gap = conformal.loc[conformal["split"].eq("loco_5cluster"), "gap"]

al = pd.read_csv(R / "active_learning_tests.csv").set_index(["strategy", "metric"])
curves = pd.read_csv(R / "active_learning_curves.csv")
final = (
    curves.loc[curves["n_labelled"].eq(curves["n_labelled"].max())]
    .groupby("strategy")[["target_recall", "test_mae"]]
    .mean()
)
ucb, unc = al.loc[("ucb", "target_recall")], al.loc[("uncertainty", "test_mae")]

q = pd.read_csv(R / "quantum_sensing_summary.csv")
qt = pd.read_csv(R / "quantum_sensing_tests.csv")
last = qt.iloc[-1]
adaptive = q.loc[q["strategy"].eq("adaptive")]
q_rows = qt.assign(
    **{
        "Sensing time (µs)": qt["budget_us"].astype(int),
        "Adaptive / fixed RMSE": qt["adaptive_to_fixed_rmse_ratio"].map("{:.2f}".format),
        "95% bootstrap CI": [
            f"{a:.2f} to {b_:.2f}" for a, b_ in zip(qt["ci95_low"], qt["ci95_high"], strict=True)
        ],
    }
)[["Sensing time (µs)", "Adaptive / fixed RMSE", "95% bootstrap CI"]]
imp = pd.read_csv(R / "kan_feature_importance.csv").head(3)["feature"].tolist()

README = f"""# Trustworthy Materials ML

[![CI](https://github.com/fronzi/trustworthy-materials-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/fronzi/trustworthy-materials-ml/actions/workflows/ci.yml)

**When should a scientist trust a model enough to act on it?** This project tests that question on real
materials data, then applies the same decision logic to a simulated quantum sensor.

**Main finding.** A random train-test split makes every learned model look better than it does on
unfamiliar chemistry: MAE rises by a factor of {ratio_all["ratio"].min():.2f} to {ratio_all["ratio"].max():.2f} under leave-one-cluster-out testing.
Conformal calibration repairs uncertainty on random splits, but not reliably under that shift.

![Random split versus chemical shift](results/figures/fig1_benchmark_shift.png)

## Results

### Prediction under distribution shift

{table(bench)}

Error terms are standard deviations across folds. Cluster-out folds have different difficulty, so their spread is large.

**Validation (nothing fitted).** The dummy model reproduces the published Matbench `expt_gap` result:
**{dummy:.4f} eV here versus {PUBLISHED_DUMMY:.4f} eV on the [leaderboard](https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_expt_gap/)**.

**Context.** The best model here is {MODELS[best_model]} at {best:.3f} eV. That is close to the leaderboard's
RF-SCM/Magpie entry ({PUBLISHED_RF:.4f} eV) and clearly behind MODNet v0.1.12 ({PUBLISHED_MODNET:.4f} eV). This repository
does not claim a state-of-the-art predictor. Its contribution is the evaluation and decision framework.

### Calibrated uncertainty: 90% intervals on `expt_gap`

{table(pd.DataFrame(cov_rows))}

<p align="center"><img src="results/figures/fig2_uncertainty_calibration.png" width="720" alt="Coverage of uncertainty intervals"></p>

### Active learning: finding the top 5% band gaps

After 1,000 labels, UCB found **{final.loc["ucb", "target_recall"]:.0%}** of the targets and random selection found **{final.loc["random", "target_recall"]:.0%}**
(paired difference {ucb["mean_difference"]:.2f}, 95% CI {ucb["ci95_low"]:.2f} to {ucb["ci95_high"]:.2f}, 10 seeds). There is a trade-off: every
targeted strategy had a *worse* general test MAE than random sampling.

<p align="center"><img src="results/figures/fig3_active_learning.png" width="720" alt="Active learning curves"></p>

### Quantum sensing extension (simulation)

An NV-centre-style Ramsey magnetometer chooses each interrogation time and phase by expected information
gain per unit time. It is compared with a fixed cycle over the same settings, using 300 paired trials.

{table(q_rows)}

Adaptive 90% credible intervals covered {adaptive["credible_90_coverage"].min():.1%} to {adaptive["credible_90_coverage"].max():.1%} of true values.
At {int(last["budget_us"])} µs the adaptive RMSE is {last["adaptive_to_fixed_rmse_ratio"]:.0%} of the fixed-cycle RMSE.
The 200 µs interval includes 1, so that time point alone is inconclusive.

<p align="center"><img src="results/figures/fig4_quantum_sensing.png" width="720" alt="Quantum sensing performance"></p>

### Pre-registered hypotheses ([plan](docs/RESEARCH_PLAN.md))

| ID | Outcome | Evidence |
|---|---|---|
| H1: random splits overstate performance | **Supported** | Cluster-out MAE is higher for all learned models in both tasks |
| H2: conformal restores nominal coverage on random splits | **Mostly supported** | Mean gap {random_gap.mean() * 100:.1f} points; the largest, {random_gap.max() * 100:.1f} points in the small `steels` task, exceeds the 3-point criterion |
| H3: guarantees weaken under chemical shift | **Supported** | Mean coverage gap {loco_gap.mean() * 100:.1f} points; largest {loco_gap.max() * 100:.1f} points |
| H4: greedy or UCB finds targets faster than random | **Supported** | UCB interval is entirely above zero; greedy result is in `active_learning_tests.csv` |
| H5: uncertainty sampling lowers test MAE faster | **Rejected** | MAE was higher by {unc["mean_difference"]:.3f} eV (95% CI {unc["ci95_low"]:.3f} to {unc["ci95_high"]:.3f}) |
| H6: adaptive sensing lowers error | **Supported at 5 of 6 budgets** | Bootstrap interval excludes a ratio of 1 except at 200 µs |

## Skills demonstrated

| Target role | Essential skills shown here | Highly desirable extensions shown here |
|---|---|---|
| **Applied / AI Research Scientist** | PyTorch, ensembles, experimental design, baselines, hypothesis testing | From-scratch KAN-style layer, conformal prediction, active learning, research under distribution shift |
| **Materials Modelling Scientist** | Composition descriptors, pymatgen, Matbench, property prediction | Chemical-cluster validation, uncertainty for screening, descriptor interpretation |
| **Computational Scientist / Scientific Software Specialist** | Installable package, CLI, tests, typed code, reproducible runs | Property-based tests, analytic and Lindblad cross-checks, parallel experiments, CI |
| **Quantum Technology Scientist** | Spin Hamiltonians, Ramsey sequences, decoherence, QuTiP | Bayesian adaptive sensing, Fisher information, calibration of posterior intervals |
| **R&D Scientist / Research Manager** | Decision framing, milestones, risks, transparent reporting | [Research plan](docs/RESEARCH_PLAN.md), [decision log](docs/DECISIONS.md), negative results kept in view |
| **Machine Learning Engineer** | Modular pipelines, deterministic seeds, automated checks | Cached artefacts, run manifest, model-agnostic uncertainty interface |

## Method notes

- **Features:** {len(pd.read_csv(R / "kan_feature_importance.csv"))} transparent descriptors from elemental properties and stoichiometry.
- **Splits:** official Matbench 5-fold seed, plus five K-means chemistry clusters held out in turn.
- **Uncertainty:** forest and ensemble spread, Gaussian intervals, absolute and normalised split conformal prediction.
- **KAN:** a compact Gaussian-basis Kolmogorov-Arnold-style network. Top descriptors by edge magnitude are {", ".join(f"`{x}`" for x in imp)}. They are model associations, not causal mechanisms.
- **Quantum checks:** analytic Ramsey signal against a QuTiP Lindblad solution, and aligned-field NV transitions against `D ± γB`.

## Run it

```bash
make install
make test
.venv/bin/trustmat all --jobs 4   # downloads public data and reproduces all results
```

## Assumptions and limits

- Composition-only models cannot distinguish polymorphs.
- Leave-one-cluster-out depends on the chosen clustering and has only five folds.
- Active learning is retrospective, so "experiments" reveal labels already in the dataset.
- The quantum study is a **simulation** with assumed T2* = 10 µs, visibility 0.8 and 2 µs overhead.
  It omits hyperfine structure, drift, pulse errors, spin-bath memory and hardware validation.
- The Cramér-Rao curve is a local reference for repeated fixed-time Ramsey, not a universal bound.

**What would change the conclusions:** structure-aware models, alternative chemistry groupings, prospective
measurements, and experimental NV data with a realistic noise model.

## References

- Dunn et al., *npj Computational Materials* (2020), [10.1038/s41524-020-00406-3](https://doi.org/10.1038/s41524-020-00406-3). Matbench.
- Zhuo et al., *J. Phys. Chem. Lett.* (2018), [10.1021/acs.jpclett.8b00124](https://doi.org/10.1021/acs.jpclett.8b00124). Band-gap data.
- Ward et al., *npj Computational Materials* (2016), [10.1038/npjcompumats.2016.28](https://doi.org/10.1038/npjcompumats.2016.28). Composition features.
- Meredig et al., *Mol. Syst. Des. Eng.* (2018), [10.1039/C8ME00012C](https://doi.org/10.1039/C8ME00012C). Cluster-based validation.
- Romano et al. (2019), [arXiv:1905.03222](https://arxiv.org/abs/1905.03222). Conformalised quantile regression.
- Lakshminarayanan et al. (2016), [arXiv:1612.01474](https://arxiv.org/abs/1612.01474). Deep ensembles.
- Liu et al. (2024), [arXiv:2404.19756](https://arxiv.org/abs/2404.19756). KAN.
- Doherty et al., *Physics Reports* (2013), [10.1016/j.physrep.2013.02.001](https://doi.org/10.1016/j.physrep.2013.02.001). NV centre.
- Degen et al., *Rev. Mod. Phys.* (2017), [10.1103/RevModPhys.89.035002](https://doi.org/10.1103/RevModPhys.89.035002). Quantum sensing.

Companion project: [nem-battery-lab](https://github.com/fronzi/nem-battery-lab).
Developed with AI coding assistance. Licensed under MIT.
"""
Path("README.md").write_text(README, encoding="utf-8")
