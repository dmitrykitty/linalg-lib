#!/usr/bin/env python3
"""Reusable Matplotlib plots for Google Benchmark JSON results."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


TIME_TO_SECONDS = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
NON_COUNTER_FIELDS = {
    "aggregate_name",
    "aggregate_unit",
    "cpu_time",
    "error_message",
    "error_occurred",
    "family_index",
    "iterations",
    "name",
    "per_family_instance_index",
    "real_time",
    "repetition_index",
    "repetitions",
    "run_name",
    "run_type",
    "threads",
    "time_unit",
}


@dataclass(frozen=True)
class BenchmarkRow:
    """One benchmark implementation and argument combination."""

    name: str
    family: str
    arguments: tuple[str, ...]
    values: dict[str, float]

    @property
    def shape_label(self) -> str:
        return " x ".join(self.arguments) if self.arguments else "no arguments"


@dataclass(frozen=True)
class BenchmarkResults:
    """Parsed benchmark rows plus measurement context."""

    rows: tuple[BenchmarkRow, ...]
    context: dict[str, Any]
    aggregate: str

    @property
    def shapes(self) -> tuple[tuple[str, ...], ...]:
        return tuple(dict.fromkeys(row.arguments for row in self.rows))

    @property
    def metrics(self) -> tuple[str, ...]:
        names = {metric for row in self.rows for metric in row.values}
        return tuple(metric for metric in ("cpu_time", "real_time") if metric in names) + tuple(
            sorted(names - {"cpu_time", "real_time"})
        )


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _run_name(row: dict[str, Any]) -> str:
    return str(row.get("run_name") or row.get("name") or "unnamed benchmark")


def _statistic(name: str) -> Callable[[list[float]], float]:
    return {
        "median": statistics.median,
        "mean": statistics.fmean,
        "min": min,
        "max": max,
    }[name]


def _select_rows(
    rows: list[dict[str, Any]], aggregate: str, name_filter: re.Pattern[str]
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if isinstance(row, dict) and name_filter.search(_run_name(row)):
            groups[_run_name(row)].append(row)

    selected: list[dict[str, Any]] = []
    reduce_values = _statistic(aggregate)
    for run_name, group in groups.items():
        existing = next((row for row in group if row.get("aggregate_name") == aggregate), None)
        if existing is not None:
            selected.append(existing)
            continue

        raw_rows = [row for row in group if not row.get("aggregate_name")]
        if not raw_rows:
            available = sorted(
                str(row["aggregate_name"]) for row in group if row.get("aggregate_name")
            )
            raise ValueError(
                f"{run_name!r} has no {aggregate!r} or raw repetitions; "
                f"available aggregates: {', '.join(available)}"
            )

        combined = dict(raw_rows[0])
        for key in set().union(*(row.keys() for row in raw_rows)):
            values = [float(row[key]) for row in raw_rows if _finite_number(row.get(key))]
            if values:
                combined[key] = reduce_values(values)
        combined["run_name"] = run_name
        selected.append(combined)
    return selected


def _normalize_values(row: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    time_unit = str(row.get("time_unit", "ns"))
    if time_unit not in TIME_TO_SECONDS:
        raise ValueError(f"unsupported Google Benchmark time unit: {time_unit!r}")

    for metric in ("cpu_time", "real_time"):
        if _finite_number(row.get(metric)):
            values[metric] = float(row[metric]) * TIME_TO_SECONDS[time_unit]

    for key, value in row.items():
        if key not in NON_COUNTER_FIELDS and _finite_number(value):
            values[key] = float(value)
    return values


def load_results(
    json_path: str | Path,
    *,
    aggregate: str = "median",
    benchmark_filter: str = ".*",
) -> BenchmarkResults:
    """Load Google Benchmark JSON and select one statistic per benchmark run.

    Existing Google Benchmark aggregate rows are reused. If the file contains raw
    repetitions instead, the requested statistic is calculated here.
    """

    if aggregate not in {"median", "mean", "min", "max"}:
        raise ValueError("aggregate must be median, mean, min, or max")

    path = Path(json_path)
    try:
        with path.open("r", encoding="utf-8") as source:
            document = json.load(source)
    except FileNotFoundError as error:
        raise ValueError(f"input file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON at line {error.lineno}, column {error.colno}") from error

    if not isinstance(document, dict) or not isinstance(document.get("benchmarks"), list):
        raise ValueError("expected a Google Benchmark JSON object with a 'benchmarks' array")

    try:
        name_filter = re.compile(benchmark_filter)
    except re.error as error:
        raise ValueError(f"invalid benchmark filter: {error}") from error

    selected = _select_rows(document["benchmarks"], aggregate, name_filter)
    parsed: list[BenchmarkRow] = []
    for row in selected:
        name = _run_name(row)
        parts = name.split("/")
        arguments = tuple(part for part in parts[1:] if ":" not in part)
        parsed.append(
            BenchmarkRow(
                name=name,
                family=parts[0].removeprefix("BM_"),
                arguments=arguments,
                values=_normalize_values(row),
            )
        )

    if not parsed:
        raise ValueError("no benchmark rows matched the filter")
    return BenchmarkResults(tuple(parsed), dict(document.get("context", {})), aggregate)


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


def _metric_info(metric: str) -> tuple[str, str, bool]:
    if metric == "cpu_time":
        return "CPU time", "time", True
    if metric == "real_time":
        return "Wall time", "time", True
    if metric == "bytes_per_second":
        return "Logical bandwidth", "bytes_rate", False
    if metric == "items_per_second":
        return "Items per second", "rate", False
    if metric.lower() in {"flop/s", "flops", "flops/s"}:
        return "Floating-point throughput", "flops", False
    return metric.replace("_", " "), "number", False


def _scaled(value: float, suffix: str) -> str:
    prefixes = ("", "k", "M", "G", "T")
    index = 0
    while abs(value) >= 1000 and index < len(prefixes) - 1:
        value /= 1000
        index += 1
    return f"{value:.3g} {prefixes[index]}{suffix}"


def _format_value(value: float, kind: str) -> str:
    if kind == "time":
        if value < 1e-6:
            return f"{value * 1e9:.3g} ns"
        if value < 1e-3:
            return f"{value * 1e6:.3g} µs"
        if value < 1:
            return f"{value * 1e3:.3g} ms"
        return f"{value:.3g} s"
    if kind == "bytes_rate":
        return _scaled(value, "B/s")
    if kind == "flops":
        return _scaled(value, "FLOP/s")
    if kind == "rate":
        return _scaled(value, "/s")
    return f"{value:.4g}"


def _save_figure(figure: Any, output_path: str | Path) -> Path:
    output = Path(output_path)
    if not output.suffix:
        output = output.with_suffix(".svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", dpi=180)
    return output


def plot_comparison(
    results: BenchmarkResults,
    output_path: str | Path,
    *,
    metric: str = "cpu_time",
    arguments: Sequence[str | int] | None = None,
    title: str | None = None,
) -> Path:
    """Create a horizontal bar chart comparing kernels for one argument shape."""

    plt, FuncFormatter = _matplotlib()
    wanted = tuple(str(value) for value in arguments) if arguments is not None else None
    if wanted is None:
        if len(results.shapes) != 1:
            choices = ", ".join("/".join(shape) for shape in results.shapes)
            raise ValueError(f"comparison needs one shape; choose one of: {choices}")
        wanted = results.shapes[0]

    rows = [row for row in results.rows if row.arguments == wanted and metric in row.values]
    if not rows:
        raise ValueError(f"no {metric!r} values found for shape {'/'.join(wanted)}")

    label, kind, lower_is_better = _metric_info(metric)
    rows.sort(key=lambda row: row.values[metric], reverse=not lower_is_better)
    values = [row.values[metric] for row in rows]
    colors = ["#1976d2", *("#78909c" for _ in rows[1:])]

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(figsize=(10, max(3.5, 0.55 * len(rows) + 1.5)))
        bars = axes.barh([row.family for row in rows], values, color=colors)
        axes.invert_yaxis()
        axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_value(value, kind)))
        axes.set_xlabel(label)
        axes.set_title(title or f"{label}: {' × '.join(wanted)}")
        axes.bar_label(bars, labels=[_format_value(value, kind) for value in values], padding=4)
        axes.grid(axis="y", visible=False)
        axes.margins(x=0.18)
        figure.text(0.99, 0.01, f"{results.aggregate} · best result highlighted", ha="right", fontsize=8)
        saved = _save_figure(figure, output_path)
        plt.close(figure)
    return saved


def plot_scaling(
    results: BenchmarkResults,
    output_path: str | Path,
    *,
    metric: str = "cpu_time",
    title: str | None = None,
    log_y: bool = False,
) -> Path:
    """Create one line per benchmark family across all argument shapes."""

    plt, FuncFormatter = _matplotlib()
    label, kind, _ = _metric_info(metric)
    shapes = results.shapes
    families = tuple(dict.fromkeys(row.family for row in results.rows))
    lookup = {(row.family, row.arguments): row.values.get(metric) for row in results.rows}
    if not any(value is not None for value in lookup.values()):
        raise ValueError(f"metric {metric!r} is not present in the results")

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(figsize=(max(9, len(shapes) * 1.15), 5.5))
        positions = list(range(len(shapes)))
        for family in families:
            values = [lookup.get((family, shape), math.nan) for shape in shapes]
            axes.plot(positions, values, marker="o", linewidth=2, label=family)
        axes.set_xticks(positions, [" × ".join(shape) or "no arguments" for shape in shapes])
        axes.tick_params(axis="x", rotation=30)
        axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_value(value, kind)))
        axes.set_xlabel("Benchmark arguments / shape")
        axes.set_ylabel(label)
        axes.set_title(title or f"{label} across benchmark shapes")
        if log_y:
            axes.set_yscale("log")
        axes.legend(frameon=False, ncols=2)
        figure.text(0.99, 0.01, results.aggregate, ha="right", fontsize=8)
        saved = _save_figure(figure, output_path)
        plt.close(figure)
    return saved


def plot_benchmarks(
    json_path: str | Path,
    output_path: str | Path,
    *,
    metric: str = "cpu_time",
    aggregate: str = "median",
    benchmark_filter: str = ".*",
    arguments: Sequence[str | int] | None = None,
    kind: str = "auto",
    title: str | None = None,
    log_y: bool = False,
) -> Path:
    """Load a JSON file and create either a comparison or scaling plot.

    ``kind='auto'`` uses a comparison when one shape is selected or present and
    a scaling plot when the JSON contains several shapes.
    """

    results = load_results(
        json_path, aggregate=aggregate, benchmark_filter=benchmark_filter
    )
    if metric not in results.metrics:
        raise ValueError(f"unknown metric {metric!r}; available: {', '.join(results.metrics)}")

    selected_kind = kind
    if selected_kind == "auto":
        selected_kind = "comparison" if arguments is not None or len(results.shapes) == 1 else "scaling"
    if selected_kind == "comparison":
        return plot_comparison(
            results, output_path, metric=metric, arguments=arguments, title=title
        )
    if selected_kind == "scaling":
        return plot_scaling(
            results, output_path, metric=metric, title=title, log_y=log_y
        )
    raise ValueError("kind must be auto, comparison, or scaling")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create documentation-ready Matplotlib plots from Google Benchmark JSON."
    )
    parser.add_argument("input", type=Path, help="Google Benchmark JSON file")
    parser.add_argument("-o", "--output", type=Path, help="output SVG or PNG path")
    parser.add_argument("--metric", default="cpu_time", help="metric to plot")
    parser.add_argument(
        "--aggregate", choices=("median", "mean", "min", "max"), default="median"
    )
    parser.add_argument("--filter", default=".*", metavar="REGEX", help="benchmark-name filter")
    parser.add_argument("--shape", help="argument shape such as 512/512 or 256/256/256")
    parser.add_argument("--kind", choices=("auto", "comparison", "scaling"), default="auto")
    parser.add_argument("--title")
    parser.add_argument("--log-y", action="store_true", help="use a logarithmic y-axis")
    parser.add_argument("--list", action="store_true", help="list metrics and shapes, then exit")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    try:
        results = load_results(
            arguments.input,
            aggregate=arguments.aggregate,
            benchmark_filter=arguments.filter,
        )
        if arguments.list:
            print("Metrics:", ", ".join(results.metrics))
            print("Shapes:", ", ".join("/".join(shape) or "no arguments" for shape in results.shapes))
            return 0

        shape = tuple(arguments.shape.split("/")) if arguments.shape else None
        metric_name = re.sub(r"[^a-z0-9]+", "-", arguments.metric.lower()).strip("-")
        output = arguments.output or arguments.input.with_name(
            f"{arguments.input.stem}-{metric_name}.svg"
        )
        saved = plot_benchmarks(
            arguments.input,
            output,
            metric=arguments.metric,
            aggregate=arguments.aggregate,
            benchmark_filter=arguments.filter,
            arguments=shape,
            kind=arguments.kind,
            title=arguments.title,
            log_y=arguments.log_y,
        )
        print(saved.resolve())
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
