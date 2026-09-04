"""Public API for the benchmark plot visualizer."""

from .plot_benchmarks import (
    BenchmarkResults,
    BenchmarkRow,
    load_results,
    plot_benchmarks,
    plot_comparison,
    plot_scaling,
)

__all__ = [
    "BenchmarkResults",
    "BenchmarkRow",
    "load_results",
    "plot_benchmarks",
    "plot_comparison",
    "plot_scaling",
]
