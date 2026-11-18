"""End-to-end experiment orchestration with cached, resumable result tables."""

from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .active import STRATEGIES, run_campaign
from .data import DATASETS, load_dataset
from .evaluation import (
    benchmark_task,
    cluster_labels,
    cluster_splits,
    random_splits,
    summarise,
    uncertainty_task,
)
from .features import feature_names, featurize
from .models import MODEL_NAMES, UNCERTAINTY_MODELS, make_model
from .quantum import SensorConfig, sensitivity_limit, simulate_trial


def _matrix(dataset: str, data_dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    frame = load_dataset(dataset, data_dir)
    return (
        featurize(frame["composition"]).to_numpy(),
        frame["target"].to_numpy(),
        frame.attrs["sha256"],
    )


def _splits(X: np.ndarray) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    return {
        "random_5fold": random_splits(len(X)),
        "loco_5cluster": cluster_splits(cluster_labels(X)),
    }


def run_benchmarks(root: Path, jobs: int) -> None:
    rows, uq_jobs, tasks = [], [], []
    hashes = {}
    for dataset in DATASETS:
        X, y, hashes[dataset] = _matrix(dataset, root / "data")
        for split, folds in _splits(X).items():
            for fold, (train, test) in enumerate(folds):
                for model in MODEL_NAMES:
                    tasks.append(
                        joblib.delayed(benchmark_task)(
                            dataset, split, fold, model, X, y, train, test
                        )
                    )
                for model in UNCERTAINTY_MODELS:
                    uq_jobs.append(
                        joblib.delayed(uncertainty_task)(
                            dataset, split, fold, model, X, y, train, test
                        )
                    )
    rows = joblib.Parallel(n_jobs=jobs, verbose=5)(tasks)
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "results" / "benchmark_folds.csv", index=False)
    summarise(frame, ["dataset", "split", "model"], ["mae", "rmse", "r2", "max_error"]).to_csv(
        root / "results" / "benchmark_summary.csv", index=False
    )
    uq = pd.DataFrame(
        [row for chunk in joblib.Parallel(n_jobs=jobs, verbose=5)(uq_jobs) for row in chunk]
    )
    uq.to_csv(root / "results" / "uncertainty_folds.csv", index=False)
    summarise(
        uq,
        ["dataset", "split", "model", "nominal", "method"],
        ["coverage", "mean_width", "spearman_sigma_error"],
    ).to_csv(root / "results" / "uncertainty_summary.csv", index=False)
    (root / "results" / "dataset_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")


def run_active(root: Path, jobs: int, seeds: int = 10) -> None:
    X, y, _ = _matrix("expt_gap", root / "data")
    frames = joblib.Parallel(n_jobs=jobs, verbose=5)(
        joblib.delayed(run_campaign)(X, y, strategy, seed)
        for strategy in STRATEGIES
        for seed in range(seeds)
    )
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(root / "results" / "active_learning_curves.csv", index=False)
    final = frame.loc[frame["n_labelled"].eq(frame["n_labelled"].max())]
    base = final.loc[final["strategy"].eq("random")].set_index("seed")
    rng = np.random.default_rng(0)
    rows = []
    for strategy in STRATEGIES[1:]:
        current = final.loc[final["strategy"].eq(strategy)].set_index("seed")
        for metric in ("target_recall", "test_mae"):
            difference = (current[metric] - base[metric]).to_numpy()
            draws = rng.choice(difference, (5000, len(difference))).mean(axis=1)
            rows.append(
                {
                    "strategy": strategy,
                    "baseline": "random",
                    "metric": metric,
                    "mean_difference": difference.mean(),
                    "ci95_low": np.quantile(draws, 0.025),
                    "ci95_high": np.quantile(draws, 0.975),
                    "seeds": len(difference),
                }
            )
    pd.DataFrame(rows).to_csv(root / "results" / "active_learning_tests.csv", index=False)


def run_kan_interpretation(root: Path) -> None:
    X, y, _ = _matrix("expt_gap", root / "data")
    model = make_model("kan_ensemble", 0).fit(X, y)
    importance = model.kan_feature_importance(X)
    frame = pd.DataFrame({"feature": feature_names(), "importance": importance})
    frame["share"] = frame["importance"] / frame["importance"].sum()
    frame.sort_values("importance", ascending=False).to_csv(
        root / "results" / "kan_feature_importance.csv", index=False
    )


def run_quantum(root: Path, jobs: int, trials: int = 300) -> None:
    config = SensorConfig()
    frames = joblib.Parallel(n_jobs=jobs, verbose=5, batch_size=8)(
        joblib.delayed(simulate_trial)(strategy, seed, config)
        for strategy in ("fixed_cycle", "adaptive")
        for seed in range(trials)
    )
    frame = pd.concat(frames, ignore_index=True)
    # Compare at identical elapsed sensing times by last-observation carry-forward.
    budgets = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    rows = []
    for (strategy, seed), group in frame.groupby(["strategy", "seed"]):
        positions = np.searchsorted(group["elapsed_us"].to_numpy(), budgets, side="right") - 1
        for budget, position in zip(budgets, positions, strict=True):
            if position >= 0:
                row = group.iloc[position]
                rows.append(
                    {
                        "strategy": strategy,
                        "seed": seed,
                        "budget_us": budget,
                        "squared_error": row["squared_error"],
                        "posterior_variance": row["posterior_variance"],
                        "inside_90": row["inside_90"],
                        "complete": group["elapsed_us"].iloc[-1] >= budget,
                    }
                )
    at_budget = pd.DataFrame(rows)
    at_budget = at_budget.loc[at_budget["complete"]]
    summary = (
        at_budget.groupby(["strategy", "budget_us"])
        .agg(
            trials=("seed", "nunique"),
            rmse_mhz=("squared_error", lambda x: float(np.sqrt(x.mean()))),
            mean_posterior_sd_mhz=("posterior_variance", lambda x: float(np.sqrt(x.mean()))),
            credible_90_coverage=("inside_90", "mean"),
        )
        .reset_index()
    )
    tau, rate = sensitivity_limit(config.t2_us, config.visibility, config.overhead_us)
    summary["fixed_tau_crb_rmse_mhz"] = 1 / np.sqrt(rate * summary["budget_us"])
    summary["crb_optimal_tau_us"] = tau
    summary.to_csv(root / "results" / "quantum_sensing_summary.csv", index=False)


def write_manifest(root: Path) -> None:
    names = ["numpy", "pandas", "scikit-learn", "torch", "lightgbm", "pymatgen", "qutip"]
    payload = {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in names},
        "matbench_split_seed": 18012019,
    }
    (root / "results" / "run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
