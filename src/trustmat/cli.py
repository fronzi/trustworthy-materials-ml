"""Command line entry point: ``trustmat all`` reproduces the published results."""

from __future__ import annotations

from pathlib import Path

import typer

from .pipeline import (
    run_active,
    run_benchmarks,
    run_kan_interpretation,
    run_quantum,
    write_manifest,
)
from .report import build_report

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _ready(root: Path) -> Path:
    (root / "results" / "figures").mkdir(parents=True, exist_ok=True)
    return root


@app.command()
def benchmark(root: Path = Path("."), jobs: int = 4) -> None:
    """Run model benchmarks and uncertainty calibration."""
    run_benchmarks(_ready(root), jobs)


@app.command()
def active(root: Path = Path("."), jobs: int = 4) -> None:
    """Run retrospective active-learning campaigns."""
    run_active(_ready(root), jobs)


@app.command()
def quantum(root: Path = Path("."), jobs: int = 4) -> None:
    """Run adaptive quantum-sensing simulations."""
    run_quantum(_ready(root), jobs)


@app.command()
def report(root: Path = Path(".")) -> None:
    """Fit the interpretation model and render all figures."""
    run_kan_interpretation(_ready(root))
    write_manifest(root)
    build_report(root)


@app.command("all")
def run_all(root: Path = Path("."), jobs: int = 4) -> None:
    """Run the complete experiment."""
    benchmark(root, jobs)
    active(root, jobs)
    quantum(root, jobs)
    report(root)
