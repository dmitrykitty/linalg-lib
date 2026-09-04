"""Rebuild the published matmul loop-order figures from committed benchmark data."""

from __future__ import annotations

import re
from pathlib import Path

from .plot_benchmarks import BenchmarkResults, load_results


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPOSITORY_ROOT / "docs/data/matmul-loop-orders-2026-08-30.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "docs/images"
SQUARE_SHAPES = ((64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512))
RECTANGULAR_SHAPES = ((1024, 1024, 64), (1024, 64, 1024), (64, 1024, 1024))
LOOP_ORDERS = ("ijk", "ikj", "jik", "jki", "kij", "kji")
BAR_ORDERS = ("ikj", "kij", "ijk", "jik", "jki", "kji")
COLORS = {
    "ijk": "#e69f00",
    "ikj": "#0072b2",
    "jik": "#d55e00",
    "jki": "#cc79a7",
    "kij": "#56b4e9",
    "kji": "#7a5195",
}
MARKERS = {"ijk": "o", "ikj": "s", "jik": "^", "jki": "D", "kij": "P", "kji": "X"}


def _matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Matplotlib is required. Run: python3 -m pip install -r "
            "tools/plot_visualizer/requirements.txt"
        ) from error
    return plt, FuncFormatter


def _loop_order(family: str) -> str:
    match = re.fullmatch(r"MatMul([A-Za-z]{3})Kernel", family)
    if match is None:
        raise ValueError(f"unexpected matmul benchmark family: {family}")
    return match.group(1).lower()


def _lookup(results: BenchmarkResults) -> dict[tuple[str, tuple[int, int, int]], dict[str, float]]:
    values = {}
    for row in results.rows:
        if len(row.arguments) != 3:
            continue
        shape = tuple(int(argument) for argument in row.arguments)
        values[(_loop_order(row.family), shape)] = row.values
    expected = {
        (order, shape)
        for order in LOOP_ORDERS
        for shape in (*SQUARE_SHAPES, *RECTANGULAR_SHAPES)
    }
    missing = expected - values.keys()
    if missing:
        raise ValueError(f"benchmark data is incomplete: {sorted(missing)}")
    return values


def _format_time(value: float, _: float | None = None) -> str:
    if value < 1e-3:
        return f"{value * 1e6:g} µs"
    if value < 1:
        return f"{value * 1e3:g} ms"
    return f"{value:g} s"


def plot_square_summary(
    results: BenchmarkResults,
    output_path: str | Path,
) -> Path:
    """Plot median CPU time and throughput for the four square sizes."""

    plt, FuncFormatter = _matplotlib()
    values = _lookup(results)
    positions = list(range(len(SQUARE_SHAPES)))
    labels = [str(shape[0]) for shape in SQUARE_SHAPES]

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, (time_axes, rate_axes) = plt.subplots(1, 2, figsize=(13.5, 5.8))
        for order in LOOP_ORDERS:
            times = [values[(order, shape)]["cpu_time"] for shape in SQUARE_SHAPES]
            rates = [values[(order, shape)]["FLOP/s"] / 1e9 for shape in SQUARE_SHAPES]
            style = {
                "color": COLORS[order],
                "marker": MARKERS[order],
                "linewidth": 2.2,
                "markersize": 6,
                "label": order,
            }
            time_axes.plot(positions, times, **style)
            rate_axes.plot(positions, rates, **style)

        time_axes.set_yscale("log")
        time_axes.yaxis.set_major_formatter(FuncFormatter(_format_time))
        time_axes.set_title("Median CPU time (log scale)")
        time_axes.set_ylabel("CPU time")
        rate_axes.set_title("Floating-point throughput")
        rate_axes.set_ylabel("GFLOP/s")
        rate_axes.set_ylim(bottom=0)

        for axes in (time_axes, rate_axes):
            axes.set_xticks(positions, labels)
            axes.set_xlabel("Square dimension N")
            axes.grid(axis="x", visible=False)

        handles, legend_labels = rate_axes.get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.085),
            ncols=6,
            frameon=False,
            title="Loop order",
        )
        figure.suptitle("Scalar row-major matmul: loop-order scaling", fontsize=16)
        figure.text(
            0.5,
            0.015,
            "Median of 15 repetitions · GCC 13.3.0 -O3 · Intel i7-10750H · WSL2",
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.21, 1, 0.93))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(figure)
    return output


def plot_rectangular_summary(
    results: BenchmarkResults,
    output_path: str | Path,
) -> Path:
    """Plot throughput for three M/N/K permutations with equal operation counts."""

    plt, _ = _matplotlib()
    values = _lookup(results)
    maximum = max(
        values[(order, shape)]["FLOP/s"] / 1e9
        for order in LOOP_ORDERS
        for shape in RECTANGULAR_SHAPES
    )

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes_group = plt.subplots(1, 3, figsize=(14.5, 5.1), sharex=True, sharey=True)
        for axes, shape in zip(axes_group, RECTANGULAR_SHAPES, strict=True):
            rates = [values[(order, shape)]["FLOP/s"] / 1e9 for order in BAR_ORDERS]
            bars = axes.barh(
                BAR_ORDERS,
                rates,
                color=[COLORS[order] for order in BAR_ORDERS],
            )
            axes.invert_yaxis()
            axes.set_xlim(0, maximum * 1.2)
            axes.set_title(f"M={shape[0]}, N={shape[1]}, K={shape[2]}", fontsize=11)
            axes.set_xlabel("GFLOP/s")
            axes.grid(axis="y", visible=False)
            axes.bar_label(bars, labels=[f"{rate:.2f}" for rate in rates], padding=3, fontsize=8)

        axes_group[0].set_ylabel("Loop order")
        figure.suptitle("Shape sensitivity at an equal 134.2 million FLOPs", fontsize=16)
        figure.text(
            0.5,
            0.015,
            "Median of 15 repetitions · each shape has M×N×K = 67,108,864",
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.07, 1, 0.92))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(figure)
    return output


def main() -> None:
    results = load_results(DEFAULT_DATA, aggregate="median", benchmark_filter=r"^BM_MatMul")
    square = plot_square_summary(results, DEFAULT_OUTPUT / "matmul-square-loop-orders.png")
    rectangular = plot_rectangular_summary(
        results, DEFAULT_OUTPUT / "matmul-rectangular-loop-orders.png"
    )
    print(square)
    print(rectangular)


if __name__ == "__main__":
    main()
