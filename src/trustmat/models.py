"""Model zoo with one interface: fit(X, y), predict(X), predict_dist(X)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

MODEL_NAMES = ("dummy", "ridge", "random_forest", "lightgbm", "mlp_ensemble", "kan_ensemble")
UNCERTAINTY_MODELS = ("random_forest", "mlp_ensemble", "kan_ensemble")


class RBFKANLayer(nn.Module):
    """Kolmogorov-Arnold style layer with learnable univariate Gaussian-basis edge functions."""

    def __init__(self, n_in: int, n_out: int, n_basis: int = 8, limit: float = 3.0) -> None:
        super().__init__()
        self.register_buffer("centres", torch.linspace(-limit, limit, n_basis))
        self.width = 2 * limit / (n_basis - 1)
        self.weights = nn.Parameter(0.05 * torch.randn(n_in, n_out, n_basis))

    def edge_values(self, x: torch.Tensor) -> torch.Tensor:
        distance = (x.unsqueeze(-1) - self.centres) / self.width
        basis = torch.exp(-0.5 * distance**2)
        return torch.einsum("nib,ijb->nij", basis, self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.edge_values(x).sum(dim=1)


class KAN(nn.Module):
    def __init__(self, n_in: int, hidden: int = 24) -> None:
        super().__init__()
        self.first = RBFKANLayer(n_in, hidden)
        self.second = RBFKANLayer(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(3.0 * torch.tanh(self.first(x))).squeeze(-1)


def _mlp(n_in: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_in, 256), nn.SiLU(), nn.Dropout(0.05),
        nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1), nn.Flatten(0),
    )  # fmt: skip


@dataclass
class NeuralEnsemble:
    kind: str
    members: int
    epochs: int
    seed: int = 0
    lr: float = 2e-3
    non_negative: bool = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> NeuralEnsemble:
        torch.set_num_threads(1)
        self.imputer_ = SimpleImputer(strategy="median").fit(X)
        self.scaler_ = StandardScaler().fit(self.imputer_.transform(X))
        data = np.clip(self.scaler_.transform(self.imputer_.transform(X)), -3, 3)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() or 1.0)
        target = (y - self.y_mean_) / self.y_std_
        x_tensor = torch.tensor(data, dtype=torch.float32)
        y_tensor = torch.tensor(target, dtype=torch.float32)
        self.nets_: list[Any] = []
        for member in range(self.members):
            seed = self.seed + 1009 * member
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(data))
            n_valid = max(1, len(data) // 10)
            valid, train = order[:n_valid], order[n_valid:]
            net = _mlp(data.shape[1]) if self.kind == "mlp" else KAN(data.shape[1])
            optimiser = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, self.epochs)
            best_loss = np.inf
            best_state: dict[str, Any] = {}
            for _ in range(self.epochs):
                net.train()
                shuffled = rng.permutation(train)
                for start in range(0, len(shuffled), 128):
                    batch = torch.as_tensor(shuffled[start : start + 128])
                    optimiser.zero_grad()
                    loss = nn.functional.huber_loss(net(x_tensor[batch]), y_tensor[batch])
                    loss.backward()
                    optimiser.step()
                scheduler.step()
                net.eval()
                with torch.no_grad():
                    score = float(nn.functional.l1_loss(net(x_tensor[valid]), y_tensor[valid]))
                if score < best_loss:
                    best_loss = score
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
            net.load_state_dict(best_state)
            net.eval()
            self.nets_.append(net)
        return self

    def _matrix(self, X: np.ndarray) -> np.ndarray:
        data = np.clip(self.scaler_.transform(self.imputer_.transform(X)), -3, 3)
        tensor = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            values = torch.stack([net(tensor) for net in self.nets_]).numpy()
        result = values * self.y_std_ + self.y_mean_
        return np.maximum(result, 0.0) if self.non_negative else result

    def predict_dist(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self._matrix(X)
        return values.mean(axis=0), values.std(axis=0, ddof=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_dist(X)[0]

    def kan_feature_importance(self, X: np.ndarray) -> np.ndarray:
        if self.kind != "kan":
            raise ValueError("Feature importance is defined for KAN ensembles only")
        data = np.clip(self.scaler_.transform(self.imputer_.transform(X)), -3, 3)
        tensor = torch.tensor(data, dtype=torch.float32)
        scores = []
        with torch.no_grad():
            for net in self.nets_:
                values = net.first.edge_values(tensor).abs().mean(dim=0).sum(dim=1)
                scores.append(values.numpy())
        return np.mean(scores, axis=0)


class Forest:
    def __init__(self, seed: int = 0, trees: int = 300, non_negative: bool = True) -> None:
        self.model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=trees, max_features=0.33, random_state=seed, n_jobs=1
            ),
        )
        self.non_negative = non_negative

    def fit(self, X: np.ndarray, y: np.ndarray) -> Forest:
        self.model.fit(X, y)
        return self

    def predict_dist(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data = self.model[0].transform(X)
        values = np.stack([tree.predict(data) for tree in self.model[-1].estimators_])
        return values.mean(axis=0), values.std(axis=0, ddof=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_dist(X)[0]


class Point:
    def __init__(self, estimator, non_negative: bool = True) -> None:
        self.estimator, self.non_negative = estimator, non_negative

    def fit(self, X: np.ndarray, y: np.ndarray) -> Point:
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        prediction = self.estimator.predict(X)
        return np.maximum(prediction, 0.0) if self.non_negative else prediction


def make_model(name: str, seed: int = 0):
    if name == "dummy":
        return Point(DummyRegressor(strategy="mean"))
    if name == "ridge":
        return Point(
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
        )
    if name == "random_forest":
        return Forest(seed)
    if name == "lightgbm":
        return Point(
            lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.5, min_child_samples=5,
                random_state=seed, n_jobs=1, verbose=-1,
            )
        )  # fmt: skip
    if name == "mlp_ensemble":
        return NeuralEnsemble("mlp", members=5, epochs=120, seed=seed)
    if name == "kan_ensemble":
        return NeuralEnsemble("kan", members=3, epochs=80, seed=seed)
    raise KeyError(name)
