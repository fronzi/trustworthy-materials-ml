"""NV-centre spin physics and adaptive Bayesian Ramsey magnetometry (simulation only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import qutip as qt
from scipy import constants

ZERO_FIELD_SPLITTING_GHZ = 2.87  # Doherty et al., Phys. Rep. 528, 1 (2013)
GAMMA_E_GHZ_PER_T = constants.physical_constants["electron gyromag. ratio in MHz/T"][0] / 1000


def nv_transition_frequencies(field_tesla: float) -> tuple[float, float]:
    """Ground-state ms=0 to ms=+/-1 frequencies (GHz) for a field along the NV axis."""
    sz = qt.jmat(1, "z")
    hamiltonian = ZERO_FIELD_SPLITTING_GHZ * sz * sz + GAMMA_E_GHZ_PER_T * field_tesla * sz
    energies = hamiltonian.diag()
    ground = energies[1]
    low, high = sorted((float(energies[0] - ground), float(energies[2] - ground)))
    return low, high


def ramsey_probability(
    frequency_mhz: np.ndarray | float, tau_us: np.ndarray | float, phase: np.ndarray | float,
    t2_us: float, visibility: float = 1.0,
) -> np.ndarray:  # fmt: skip
    """Probability of outcome 1 after a Ramsey sequence with exponential dephasing."""
    argument = 2 * np.pi * np.asarray(frequency_mhz) * np.asarray(tau_us) + np.asarray(phase)
    decay = np.exp(-np.asarray(tau_us) / t2_us)
    return 0.5 * (1 + visibility * decay * np.cos(argument))


def ramsey_master_equation(
    frequency_mhz: float, tau_us: float, phase: float, t2_us: float
) -> float:
    """Lindblad check of the analytic Ramsey signal with ideal pi/2 pulses."""
    h = 0.5 * 2 * np.pi * frequency_mhz * qt.sigmaz()
    collapse = [np.sqrt(1 / (2 * t2_us)) * qt.sigmaz()]
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    result = qt.mesolve(h, qt.ket2dm(plus), [0, tau_us], collapse, options={"store_states": True})
    readout = (qt.basis(2, 0) + np.exp(-1j * phase) * qt.basis(2, 1)).unit()
    return float(qt.expect(qt.ket2dm(readout), result.states[-1]))


def fisher_per_shot(tau_us: float, t2_us: float, visibility: float = 1.0) -> float:
    """Maximum classical Fisher information on frequency (1/MHz^2) at the best phase."""
    return float((2 * np.pi * tau_us * visibility * np.exp(-tau_us / t2_us)) ** 2)


def sensitivity_limit(t2_us: float, visibility: float, overhead_us: float) -> tuple[float, float]:
    """Best fixed-tau repeated-Ramsey Fisher information per microsecond of total time."""
    grid = np.linspace(t2_us / 1000, 3 * t2_us, 6000)
    rates = np.array([fisher_per_shot(t, t2_us, visibility) for t in grid]) / (grid + overhead_us)
    best = int(rates.argmax())
    return float(grid[best]), float(rates[best])


@dataclass(frozen=True)
class SensorConfig:
    t2_us: float = 10.0
    visibility: float = 0.8
    overhead_us: float = 2.0
    frequency_limit_mhz: float = 0.5
    grid_points: int = 1601
    shots: int = 200


def _entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def choose_setting(
    posterior: np.ndarray, grid: np.ndarray, taus: np.ndarray, phases: np.ndarray, c: SensorConfig
) -> tuple[float, float]:
    """Maximise expected information gain per unit experiment time."""
    best, setting = -np.inf, (float(taus[0]), float(phases[0]))
    for tau in taus:
        for phase in phases:
            p = ramsey_probability(grid, tau, phase, c.t2_us, c.visibility)
            mean = float(posterior @ p)
            gain = float(_entropy(np.array(mean)) - posterior @ _entropy(p))
            value = gain / (tau + c.overhead_us)
            if value > best:
                best, setting = value, (float(tau), float(phase))
    return setting


def simulate_trial(strategy: str, seed: int, c: SensorConfig = SensorConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grid = np.linspace(-c.frequency_limit_mhz, c.frequency_limit_mhz, c.grid_points)
    truth = float(rng.uniform(-c.frequency_limit_mhz, c.frequency_limit_mhz))
    posterior = np.full(c.grid_points, 1 / c.grid_points)
    taus = np.geomspace(0.25, c.t2_us, 8)
    phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    elapsed = 0.0
    rows = []
    for shot in range(c.shots):
        if strategy == "adaptive":
            tau, phase = choose_setting(posterior, grid, taus, phases, c)
        elif strategy == "fixed_cycle":
            tau = float(taus[shot % len(taus)])
            phase = float(phases[shot % len(phases)])
        else:
            raise KeyError(strategy)
        probability = ramsey_probability(truth, tau, phase, c.t2_us, c.visibility)
        outcome = rng.random() < probability
        likelihood = ramsey_probability(grid, tau, phase, c.t2_us, c.visibility)
        posterior *= likelihood if outcome else 1 - likelihood
        posterior /= posterior.sum()
        elapsed += tau + c.overhead_us
        mean = float(posterior @ grid)
        variance = float(posterior @ (grid - mean) ** 2)
        cdf = np.cumsum(posterior)
        low, high = np.interp([0.05, 0.95], cdf, grid)
        rows.append(
            {
                "strategy": strategy, "seed": seed, "shot": shot + 1, "elapsed_us": elapsed,
                "squared_error": (mean - truth) ** 2, "posterior_variance": variance,
                "inside_90": bool(low <= truth <= high),
            }
        )  # fmt: skip
    return pd.DataFrame(rows)
