"""Versioned access to public Matbench v0.1 regression tasks."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

DATASETS = {
    "expt_gap": ("matbench_expt_gap", "gap expt", "eV", True),
    "steels": ("matbench_steels", "yield strength", "MPa", True),
}
URL = "https://ml.materialsproject.org/projects/{name}.json.gz"


def load_dataset(key: str, cache: Path = Path("data")) -> pd.DataFrame:
    name, target, _, _ = DATASETS[key]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{name}.json.gz"
    if not path.exists():
        response = requests.get(
            URL.format(name=name), headers={"User-Agent": "trustmat"}, timeout=120
        )
        response.raise_for_status()
        path.write_bytes(response.content)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    frame = pd.DataFrame(payload["data"], columns=payload["columns"])
    frame = frame.rename(columns={"composition": "composition", target: "target"})
    if frame[["composition", "target"]].isna().any().any():
        raise ValueError(f"{key} contains missing values")
    frame.attrs["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return frame[["composition", "target"]]
