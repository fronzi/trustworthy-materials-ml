"""Transparent composition descriptors built from elemental properties."""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

PROPERTIES = (
    "atomic_number", "atomic_mass", "electronegativity", "atomic_radius", "row", "group",
    "mendeleev_number", "valence_s", "valence_p", "valence_d", "valence_f",
)  # fmt: skip
STATISTICS = ("mean", "std", "min", "max", "range")


@cache
def element_vector(symbol: str) -> tuple[float, ...]:
    element = Element(symbol)
    shells = element.full_electronic_structure
    outer = max(n for n, _, _ in shells)
    valence = {"s": 0.0, "p": 0.0, "d": 0.0, "f": 0.0}
    target_shell = {"s": outer, "p": outer, "d": outer - 1, "f": outer - 2}
    for n, orbital, electrons in shells:
        if n == target_shell[orbital]:
            valence[orbital] += float(electrons)
    radius = element.atomic_radius
    return (
        float(element.Z),
        float(element.atomic_mass),
        float(element.X),
        float(radius) if radius is not None else np.nan,
        float(element.row),
        float(element.group),
        float(element.mendeleev_no) if element.mendeleev_no is not None else np.nan,
        valence["s"],
        valence["p"],
        valence["d"],
        valence["f"],
    )


def feature_names() -> list[str]:
    names = [f"{prop}_{stat}" for prop in PROPERTIES for stat in STATISTICS]
    return [*names, "n_elements", "fraction_l2", "fraction_l3", "fraction_l5", "max_fraction"]


def featurize_formula(formula: str) -> np.ndarray:
    fractions = Composition(formula).fractional_composition.get_el_amt_dict()
    symbols = list(fractions)
    weights = np.array([fractions[s] for s in symbols], dtype=float)
    table = np.array([element_vector(s) for s in symbols], dtype=float)
    values: list[float] = []
    for column in table.T:
        present = ~np.isnan(column)
        if not present.any():
            values.extend([np.nan] * len(STATISTICS))
            continue
        x, w = column[present], weights[present] / weights[present].sum()
        mean = float(w @ x)
        minimum, maximum = float(x.min()), float(x.max())
        values.extend(
            [mean, float(np.sqrt(w @ (x - mean) ** 2)), minimum, maximum, maximum - minimum]
        )
    norms = [float(np.sum(weights**p) ** (1 / p)) for p in (2, 3, 5)]
    return np.array([*values, float(len(symbols)), *norms, float(weights.max())])


def featurize(compositions: pd.Series) -> pd.DataFrame:
    matrix = np.vstack([featurize_formula(value) for value in compositions])
    return pd.DataFrame(matrix, columns=feature_names(), index=compositions.index)
