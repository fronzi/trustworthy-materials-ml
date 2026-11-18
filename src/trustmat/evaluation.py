"""Benchmarks, leave-one-cluster-out validation and calibrated uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .models import make_model

MATBENCH_SEED = 18012019


def random_splits(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The published Matbench v0.1 outer split."""
    return list(KFold(5, shuffle=True, random_state=MATBENCH_SEED).split(np.zeros(n)))


def cluster_labels(X: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    data = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    return KMeans(k, random_state=seed, n_init=10).fit_predict(data)


def cluster_splits(labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.flatnonzero(labels != c), np.flatnonzero(labels == c)) for c in np.unique(labels)]


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = y - prediction
    total = np.sum((y - y.mean()) ** 2)
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_error": float(np.abs(error).max()),
        "r2": float(1 - np.sum(error**2) / total) if total else float("nan"),
    }


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.inf) if rank > n else float(np.sort(scores)[rank - 1])


def benchmark_task(
    dataset: str, split: str, fold: int, model_name: str, X, y, train, test, seed: int = 0
) -> dict:
    model = make_model(model_name, seed + fold).fit(X[train], y[train])
    prediction = model.predict(X[test])
    return {
        "dataset": dataset, "split": split, "fold": fold, "model": model_name,
        "n_train": len(train), "n_test": len(test), **regression_metrics(y[test], prediction),
    }  # fmt: skip


def uncertainty_task(
    dataset: str, split: str, fold: int, model_name: str, X, y, train, test, seed: int = 0
) -> list[dict]:
    fit, calibration = train_test_split(train, test_size=0.2, random_state=seed + fold)
    model = make_model(model_name, seed + fold).fit(X[fit], y[fit])
    mean_cal, std_cal = model.predict_dist(X[calibration])
    mean_test, std_test = model.predict_dist(X[test])
    scale = np.median(std_cal) * 0.05 + 1e-9
    rows = []
    for nominal in (0.68, 0.90, 0.95):
        alpha = 1 - nominal
        z = stats.norm.ppf(1 - alpha / 2)
        raw = np.abs(y[test] - mean_test) <= z * std_test
        for method, cal_scale, test_scale in (
            ("absolute_conformal", np.ones_like(std_cal), np.ones_like(std_test)),
            ("normalised_conformal", std_cal + scale, std_test + scale),
        ):
            q = conformal_quantile(np.abs(y[calibration] - mean_cal) / cal_scale, alpha)
            half_width = q * test_scale
            rows.append(
                {
                    "dataset": dataset, "split": split, "fold": fold, "model": model_name,
                    "nominal": nominal, "method": method,
                    "coverage": float((np.abs(y[test] - mean_test) <= half_width).mean()),
                    "mean_width": float((2 * half_width).mean()),
                }
            )  # fmt: skip
        rows.append(
            {
                "dataset": dataset, "split": split, "fold": fold, "model": model_name,
                "nominal": nominal, "method": "raw_gaussian", "coverage": float(raw.mean()),
                "mean_width": float((2 * z * std_test).mean()),
            }
        )  # fmt: skip
    rank = stats.spearmanr(std_test, np.abs(y[test] - mean_test)).statistic
    for row in rows:
        row["spearman_sigma_error"] = float(rank)
    return rows


def summarise(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys)[values].agg(["mean", "std"])
    grouped.columns = [f"{name}_{stat}" for name, stat in grouped.columns.to_flat_index()]
    return grouped.reset_index()
