"""Retrospective active-learning campaign on a held-out candidate pool."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .models import Forest

STRATEGIES = ("random", "uncertainty", "greedy", "ucb")


def acquisition(strategy: str, mean: np.ndarray, std: np.ndarray, rng) -> np.ndarray:
    if strategy == "random":
        return rng.random(len(mean))
    if strategy == "uncertainty":
        return std
    if strategy == "greedy":
        return mean
    if strategy == "ucb":
        return mean + std
    raise KeyError(strategy)


def run_campaign(
    X: np.ndarray, y: np.ndarray, strategy: str, seed: int,
    initial: int = 100, batch: int = 50, rounds: int = 18, top_fraction: float = 0.05,
) -> pd.DataFrame:  # fmt: skip
    rng = np.random.default_rng(seed)
    pool_index, test_index = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
    cutoff = np.quantile(y[pool_index], 1 - top_fraction)
    targets = set(pool_index[y[pool_index] >= cutoff].tolist())
    order = rng.permutation(pool_index)
    labelled, pool = list(order[:initial]), list(order[initial:])
    rows = []
    for step in range(rounds + 1):
        model = Forest(seed=seed + step, trees=100).fit(X[labelled], y[labelled])
        rows.append(
            {
                "strategy": strategy,
                "seed": seed,
                "n_labelled": len(labelled),
                "test_mae": float(np.abs(y[test_index] - model.predict(X[test_index])).mean()),
                "target_recall": len(targets.intersection(labelled)) / len(targets),
            }
        )
        if step == rounds:
            break
        mean, std = model.predict_dist(X[pool])
        score = acquisition(strategy, mean, std, rng)
        chosen = np.argsort(-score, kind="stable")[:batch]
        selected = {pool[i] for i in chosen}
        labelled.extend(selected)
        pool = [item for item in pool if item not in selected]
    return pd.DataFrame(rows)
