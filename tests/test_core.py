from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from trustmat.active import acquisition, run_campaign
from trustmat.evaluation import (
    cluster_splits,
    conformal_quantile,
    random_splits,
    regression_metrics,
)
from trustmat.features import feature_names, featurize_formula
from trustmat.models import KAN, Forest, NeuralEnsemble, make_model
from trustmat.quantum import (
    GAMMA_E_GHZ_PER_T,
    ZERO_FIELD_SPLITTING_GHZ,
    SensorConfig,
    fisher_per_shot,
    nv_transition_frequencies,
    ramsey_master_equation,
    ramsey_probability,
    simulate_trial,
)


def test_composition_features_are_scale_invariant() -> None:
    assert featurize_formula("Fe2O3") == pytest.approx(featurize_formula("Fe4O6"))
    assert len(featurize_formula("NaCl")) == len(feature_names())


def test_elemental_material_has_zero_spread() -> None:
    values = dict(zip(feature_names(), featurize_formula("Si"), strict=True))
    assert values["atomic_number_mean"] == 14
    assert values["atomic_number_std"] == 0
    assert values["max_fraction"] == 1


def test_matbench_split_partitions_all_samples() -> None:
    folds = random_splits(101)
    assert sorted(np.concatenate([test for _, test in folds]).tolist()) == list(range(101))


def test_cluster_split_never_leaks() -> None:
    labels = np.array([0, 0, 1, 1, 2])
    for train, test in cluster_splits(labels):
        assert not set(labels[train]).intersection(labels[test])


@given(st.integers(19, 500))
def test_conformal_rank_is_finite_sample_valid(n: int) -> None:
    scores = np.arange(1, n + 1, dtype=float)
    assert conformal_quantile(scores, 0.1) == np.ceil((n + 1) * 0.9)


def test_conformal_is_infinite_when_calibration_is_too_small() -> None:
    assert conformal_quantile(np.array([1.0, 2.0]), 0.05) == np.inf


def test_conformal_coverage_on_exchangeable_data() -> None:
    rng = np.random.default_rng(0)
    hits = []
    for _ in range(300):
        draws = np.abs(rng.normal(size=201))
        hits.append(draws[-1] <= conformal_quantile(draws[:-1], 0.1))
    assert np.mean(hits) == pytest.approx(0.9, abs=0.04)


def test_metrics_known_values() -> None:
    result = regression_metrics(np.array([1.0, 3.0]), np.array([2.0, 2.0]))
    assert result["mae"] == 1 and result["rmse"] == 1 and result["r2"] == 0


def test_models_learn_a_simple_signal() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(220, 4))
    y = 3 + np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    for model in (
        Forest(trees=30),
        NeuralEnsemble("mlp", members=2, epochs=25),
        NeuralEnsemble("kan", members=2, epochs=25),
    ):
        fitted = model.fit(X[:170], y[:170])
        mean, std = fitted.predict_dist(X[170:])
        assert np.abs(mean - y[170:]).mean() < np.abs(y[170:] - y[:170].mean()).mean()
        assert (std >= 0).all()


def test_kan_shape_and_importance() -> None:
    import torch

    assert KAN(5)(torch.zeros(3, 5)).shape == (3,)
    X = np.random.default_rng(0).normal(size=(40, 3))
    model = make_model("kan_ensemble").__class__("kan", members=1, epochs=1).fit(X, X[:, 0])
    assert model.kan_feature_importance(X).shape == (3,)


def test_acquisition_rules() -> None:
    mean, std = np.array([1.0, 2.0]), np.array([3.0, 0.0])
    rng = np.random.default_rng(0)
    assert acquisition("uncertainty", mean, std, rng).argmax() == 0
    assert acquisition("greedy", mean, std, rng).argmax() == 1
    assert acquisition("ucb", mean, std, rng).argmax() == 0


def test_campaign_labels_without_replacement() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = X[:, 0]
    result = run_campaign(X, y, "ucb", 0, initial=10, batch=5, rounds=2)
    assert result["n_labelled"].tolist() == [10, 15, 20]
    assert result["target_recall"].is_monotonic_increasing


def test_nv_levels_match_exact_aligned_field_solution() -> None:
    field = 2e-3
    low, high = nv_transition_frequencies(field)
    assert low == pytest.approx(ZERO_FIELD_SPLITTING_GHZ - GAMMA_E_GHZ_PER_T * field)
    assert high == pytest.approx(ZERO_FIELD_SPLITTING_GHZ + GAMMA_E_GHZ_PER_T * field)


@pytest.mark.parametrize(("f", "tau", "phase"), [(0.1, 3.0, 0.4), (-0.25, 7.0, 1.2), (0.0, 1.0, 0)])
def test_ramsey_analytic_matches_lindblad(f: float, tau: float, phase: float) -> None:
    analytic = float(ramsey_probability(f, tau, phase, 10.0))
    assert ramsey_master_equation(f, tau, phase, 10.0) == pytest.approx(analytic, abs=1e-5)


def test_fisher_vanishes_without_sensing_time() -> None:
    assert fisher_per_shot(0.0, 10.0) == 0


def test_bayesian_trial_is_consistent() -> None:
    config = SensorConfig(grid_points=401, shots=20)
    trial = simulate_trial("adaptive", 1, config)
    assert trial["elapsed_us"].is_monotonic_increasing
    assert trial["posterior_variance"].iloc[-1] < (2 * config.frequency_limit_mhz) ** 2 / 12
