"""Publication-quality result figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import MODEL_NAMES, UNCERTAINTY_MODELS

LABEL = {
    "dummy": "Dummy", "ridge": "Ridge", "random_forest": "Random forest",
    "lightgbm": "LightGBM", "mlp_ensemble": "MLP ensemble", "kan_ensemble": "KAN ensemble",
}  # fmt: skip
BLUE, TEAL, RED, GREY, VIOLET = "#2563eb", "#0f766e", "#dc2626", "#6b7280", "#7c3aed"


def _save(fig, root: Path, name: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(root / "results" / "figures" / f"{name}.{suffix}", dpi=200)
    plt.close(fig)


def benchmark_figure(root: Path) -> None:
    table = pd.read_csv(root / "results" / "benchmark_summary.csv")
    units = {"expt_gap": "eV", "steels": "MPa"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, dataset in zip(axes, ("expt_gap", "steels"), strict=True):
        frame = table.loc[table["dataset"].eq(dataset)]
        x = np.arange(len(MODEL_NAMES))
        for offset, split, colour, label in (
            (-0.2, "random_5fold", BLUE, "Matbench 5-fold"),
            (0.2, "loco_5cluster", RED, "Leave one cluster out"),
        ):
            values = frame.loc[frame["split"].eq(split)].set_index("model").loc[list(MODEL_NAMES)]
            ax.bar(
                x + offset, values["mae_mean"], 0.38, yerr=values["mae_std"], color=colour,
                label=label, error_kw={"lw": 0.8},
            )  # fmt: skip
        ax.set_xticks(x, [LABEL[m] for m in MODEL_NAMES], rotation=35, ha="right")
        ax.set_ylabel(f"Mean absolute error ({units[dataset]})")
        ax.set_title(dataset, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, root, "fig1_benchmark_shift")


def uncertainty_figure(root: Path) -> None:
    table = pd.read_csv(root / "results" / "uncertainty_summary.csv")
    frame = table.loc[table["dataset"].eq("expt_gap") & table["nominal"].eq(0.9)]
    methods = [
        ("raw_gaussian", GREY, "Raw Gaussian"),
        ("absolute_conformal", BLUE, "Absolute conformal"),
        ("normalised_conformal", TEAL, "Normalised conformal"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3), sharey=True)
    for ax, split, title in zip(
        axes, ("random_5fold", "loco_5cluster"), ("Random split", "Chemical shift"), strict=True
    ):
        x = np.arange(len(UNCERTAINTY_MODELS))
        for i, (method, colour, label) in enumerate(methods):
            values = (
                frame.loc[frame["split"].eq(split) & frame["method"].eq(method)]
                .set_index("model")
                .loc[list(UNCERTAINTY_MODELS)]
            )
            ax.bar(
                x + (i - 1) * 0.26, values["coverage_mean"] * 100, 0.25,
                yerr=values["coverage_std"] * 100, color=colour, label=label,
                error_kw={"lw": 0.8},
            )  # fmt: skip
        ax.axhline(90, color="#111827", ls="--", lw=1)
        ax.set_xticks(x, [LABEL[m] for m in UNCERTAINTY_MODELS], rotation=15)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Coverage of 90% interval (%)")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    _save(fig, root, "fig2_uncertainty_calibration")


def active_figure(root: Path) -> None:
    curves = pd.read_csv(root / "results" / "active_learning_curves.csv")
    colours = {"random": GREY, "uncertainty": BLUE, "greedy": RED, "ucb": TEAL}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3))
    for ax, metric, label in (
        (axes[0], "target_recall", "Top-5% band gaps found (recall)"),
        (axes[1], "test_mae", "Held-out MAE (eV)"),
    ):
        for strategy_key, group in curves.groupby("strategy"):
            strategy = str(strategy_key)
            stats = group.groupby("n_labelled")[metric].agg(["mean", "sem"])
            ax.plot(stats.index, stats["mean"], color=colours[strategy], label=strategy)
            ax.fill_between(
                stats.index, stats["mean"] - 1.96 * stats["sem"],
                stats["mean"] + 1.96 * stats["sem"], color=colours[strategy], alpha=0.15, lw=0,
            )  # fmt: skip
        ax.set_xlabel("Labelled materials")
        ax.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, root, "fig3_active_learning")


def quantum_figure(root: Path) -> None:
    table = pd.read_csv(root / "results" / "quantum_sensing_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3))
    for strategy, colour, label in (
        ("fixed_cycle", GREY, "Fixed cycle"),
        ("adaptive", VIOLET, "Adaptive"),
    ):
        frame = table.loc[table["strategy"].eq(strategy)]
        axes[0].loglog(
            frame["budget_us"], frame["rmse_mhz"] * 1000, "o-", color=colour, label=label
        )
        axes[1].plot(
            frame["budget_us"], frame["credible_90_coverage"] * 100, "o-", color=colour, label=label
        )
    reference = table.drop_duplicates("budget_us")
    axes[0].loglog(
        reference["budget_us"], reference["fixed_tau_crb_rmse_mhz"] * 1000, "k--",
        label="Fixed-τ local Cramér-Rao reference",
    )  # fmt: skip
    axes[0].set_xlabel("Total sensing time (µs)")
    axes[0].set_ylabel("Frequency RMSE (kHz)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].axhline(90, color="#111827", ls="--", lw=1)
    axes[1].set_ylim(70, 100)
    axes[1].set_xlabel("Total sensing time (µs)")
    axes[1].set_ylabel("90% credible-interval coverage (%)")
    _save(fig, root, "fig4_quantum_sensing")


def importance_figure(root: Path) -> None:
    table = pd.read_csv(root / "results" / "kan_feature_importance.csv").head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.barh(table["feature"], table["share"] * 100, color=TEAL)
    ax.set_xlabel("Share of total KAN edge magnitude (%)")
    ax.set_ylabel("Composition descriptor")
    _save(fig, root, "fig5_kan_descriptors")


def build_report(root: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.25, "figure.constrained_layout.use": True,
            "pdf.fonttype": 42,
        }
    )  # fmt: skip
    benchmark_figure(root)
    uncertainty_figure(root)
    active_figure(root)
    quantum_figure(root)
    importance_figure(root)
