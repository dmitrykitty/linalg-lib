#!/usr/bin/env python3
"""Summarize a controlled GEMM benchmark run and render the published GEMM charts as static SVG.

Two steps, so the charts can be regenerated from small committed files:

1. ``--from-results DIR`` reads a ``tools/run_benchmarks.sh`` output directory (Google Benchmark
   JSON + ``perf stat`` CSV) and writes ``docs/data/gemm-<date>.csv`` and
   ``docs/data/gemm-perf-<date>.csv`` with medians, CVs, and derived counters.
2. The charts are always drawn from those CSV files, each as ``<name>-light.svg`` and
   ``<name>-dark.svg`` in ``docs/images/`` (embed them with ``<picture>`` and
   ``prefers-color-scheme``).

Standard library only. Usage, from the repository root:

    python3 tools/plot_gemm_results.py \\
        --from-results benchmark-results/20261018-000908-b0dc003-dirty --date 2026-10-18
    python3 tools/plot_gemm_results.py --date 2026-10-18
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from html import escape
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "docs/data"
IMAGE_DIR = REPOSITORY_ROOT / "docs/images"

# i7-10750H with turbo disabled: 2.6 GHz x 2 FMA ports x (4 double | 8 float) lanes x 2 FLOP.
CORE_FREQUENCY_GHZ = 2.6
CORES = 6
PEAK_FLOP_PER_CYCLE = {"f64": 16, "f32": 32}
MACHINE = "i7-10750H, 6 cores at fixed 2.6 GHz, SMT off"

# Variants in optimization order -> (label, colour key). Colour follows the entity in every chart:
# our 1-thread kernel is always blue, the 6-thread run always orange. OpenBLAS reuses the colour of
# our variant with the same thread count and is told apart by a dashed line or a light bar.
VARIANTS = {
    "NaiveIjk": ("naive ijk", "naive"),
    "ScalarIkj": ("ikj loop order", "ikj"),
    "BlockedPortable": ("blocked + packed, portable kernel", "portable"),
    "BlockedAvx2": ("blocked + AVX2/FMA, 1 thread", "avx2"),
    "BlockedAvx2Mt": ("blocked + AVX2/FMA, 6 threads", "avx2mt"),
    "OpenBlas": ("OpenBLAS 0.3.32, 1 thread", "avx2"),
    "OpenBlasMt": ("OpenBLAS 0.3.32, 6 threads", "avx2mt"),
}
SQUARE_SIZES = (64, 128, 256, 512, 1024, 2048, 4096)
THREAD_SIZES = (256, 512, 1024, 2048)
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Validated categorical palette (dataviz reference instance): the first five slots in their fixed
# order, light and dark steps; plus an ordinal blue ramp for matrix sizes and the chart chrome.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "base": "#c3c2b7",
        "series": {"avx2": "#2a78d6", "avx2mt": "#eb6834", "portable": "#1baf7a",
                   "ikj": "#eda100", "naive": "#e87ba4"},
        "ramp": ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "base": "#383835",
        "series": {"avx2": "#3987e5", "avx2mt": "#d95926", "portable": "#199e70",
                   "ikj": "#c98500", "naive": "#d55181"},
        # On the dark surface the larger sizes get the lighter, higher-contrast steps.
        "ramp": ["#184f95", "#256abf", "#5598e7", "#9ec5f4"],
    },
}


def peak_gflops(precision: str, cores: int) -> float:
    return PEAK_FLOP_PER_CYCLE[precision] * CORE_FREQUENCY_GHZ * cores


# --------------------------------------------------------------------------- summarize raw results


def _benchmark_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())["benchmarks"]


def _split_name(run_name: str) -> tuple[str, list[int]]:
    parts = [part for part in run_name.split("/") if part not in {"real_time", "process_time"}]
    return parts[0].removeprefix("BM_"), [int(part) for part in parts[1:]]


def summarize(results_dir: Path, date: str) -> tuple[Path, Path]:
    """Write the median/CV summary and the perf-counter summary of one run."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = DATA_DIR / f"gemm-{date}.csv"
    rows = []
    for json_path in sorted(results_dir.glob("*.json")):
        aggregates: dict[str, dict[str, dict]] = defaultdict(dict)
        for row in _benchmark_rows(json_path):
            if row.get("run_type") == "aggregate" and not row.get("error_occurred"):
                aggregates[row["run_name"]][row["aggregate_name"]] = row
        for run_name, by_statistic in aggregates.items():
            median, cv = by_statistic.get("median"), by_statistic.get("cv")
            if median is None or cv is None:
                continue
            family, arguments = _split_name(run_name)
            threads = median.get("gemm_threads", median.get("threads", 1))
            if family.startswith("MatMulBlockedAvx2FmaThreads"):
                threads = arguments[3]
            rows.append({
                "group": json_path.stem,
                "family": family,
                "m": arguments[0],
                "n": arguments[1],
                "k": arguments[2],
                "threads": int(threads),
                "median_gflops": round(median["FLOP/s"] / 1e9, 4),
                "median_seconds": median["real_time"] * {"ns": 1e-9, "us": 1e-6, "ms": 1e-3,
                                                        "s": 1.0}[median["time_unit"]],
                "cv_real_time": round(cv["real_time"], 4),
            })
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["group"], r["family"], r["m"], r["n"],
                                                       r["k"], r["threads"])))

    perf_path = DATA_DIR / f"gemm-perf-{date}.csv"
    perf_rows = []
    for csv_path in sorted((results_dir / "perf").glob("*.csv")):
        counters = {}
        for record in csv.reader(csv_path.open()):
            if len(record) > 2 and record[0] and not record[0].startswith("#"):
                try:
                    counters[record[2]] = float(record[0])
                except ValueError:
                    pass
        benchmark = _benchmark_rows(csv_path.with_suffix(".json"))[0]
        threads = int(benchmark.get("gemm_threads", benchmark.get("threads", 1)))
        gflops = benchmark["FLOP/s"] / 1e9
        ghz = counters["cycles"] / (counters["task-clock"] * 1e6)
        perf_rows.append({
            "variant": csv_path.stem,
            "threads": threads,
            "gflops_1024": round(gflops, 3),
            "ghz": round(ghz, 3),
            "ipc": round(counters["instructions"] / counters["cycles"], 3),
            "flop_per_cycle_per_core": round(gflops / (ghz * threads), 3),
            "llc_loads_per_kilo_instruction": round(
                1000 * counters["LLC-loads"] / counters["instructions"], 3),
            "l2_miss_stall_share": round(
                counters["cycle_activity.stalls_l2_miss"] / counters["cycles"], 4),
            "context_switches": int(counters["context-switches"]),
        })
    order = list(VARIANTS)
    perf_rows.sort(key=lambda r: order.index(r["variant"]))
    with perf_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(perf_rows[0]))
        writer.writeheader()
        writer.writerows(perf_rows)
    return summary_path, perf_path



# --------------------------------------------------------------------------- SVG primitives


class Svg:
    """A small SVG writer with theme tokens (same conventions as the event engine's charts)."""

    def __init__(self, width: int, height: int, theme: str, title: str):
        self.t = THEMES[theme]
        self.width, self.height, self.title = width, height, title
        self.parts: list[str] = []  # the frame is added in save(), once the height is final

    def colour(self, token: str) -> str:
        return self.t.get(token) or self.t["series"].get(token) or token

    def text(self, x, y, content, size=12, color="ink2", anchor="start", weight=400, halo=False):
        common = (f'x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
                  f'font-weight="{weight}"')
        if halo:
            # A surface-coloured outline drawn underneath keeps reference lines from crossing the
            # glyphs. A separate element instead of paint-order, which not every renderer supports.
            surface = self.t["surface"]
            self.parts.append(f'<text {common} fill="{surface}" stroke="{surface}" '
                              f'stroke-width="4" stroke-linejoin="round" aria-hidden="true">'
                              f'{escape(content)}</text>')
        self.parts.append(f'<text {common} fill="{self.colour(color)}">{escape(content)}</text>')

    def line(self, x1, y1, x2, y2, color, width=1.0, dash=None):
        dasharray = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{self.colour(color)}" stroke-width="{width}" stroke-linecap="round"'
            f'{dasharray}/>')

    def polyline(self, points, color, width=2.0, dash=None):
        dasharray = f' stroke-dasharray="{dash}"' if dash else ""
        coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{self.colour(color)}" '
            f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dasharray}/>')

    def dot(self, x, y, color, tooltip, hollow=False, radius=4.0):
        # 2px surface ring keeps overlapping markers apart; hollow = OpenBLAS.
        fill = self.t["surface"] if hollow else self.colour(color)
        stroke = self.colour(color) if hollow else self.t["surface"]
        self.parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="2"><title>{escape(tooltip)}</title></circle>')

    def hbar(self, x0, x1, y, height, color, tooltip, light=False):
        # Square at the baseline, 4px rounded data end.
        r = min(4.0, (x1 - x0) / 2, height / 2)
        top, bottom = y - height / 2, y + height / 2
        path = (f"M{x0:.1f},{top:.1f} H{x1 - r:.1f} Q{x1:.1f},{top:.1f} {x1:.1f},{top + r:.1f} "
                f"V{bottom - r:.1f} Q{x1:.1f},{bottom:.1f} {x1 - r:.1f},{bottom:.1f} "
                f"H{x0:.1f} Z")
        colour = self.colour(color)
        style = (f'fill="{colour}" fill-opacity="0.28" stroke="{colour}" stroke-width="1.5"'
                 if light else f'fill="{colour}"')
        self.parts.append(f'<path d="{path}" {style}><title>{escape(tooltip)}</title></path>')

    def legend(self, x, y, entries, max_x=None):
        """entries: (label, colour, kind) with kind 'dot', 'dash', 'light' or 'line'."""

        start = x
        for label, color, kind in entries:
            width = 30 + 6.6 * len(label)
            if max_x and x + width > max_x:
                x, y = start, y + 20
            colour = self.colour(color)
            if kind == "dot":
                self.parts.append(f'<circle cx="{x + 5:.1f}" cy="{y - 4:.1f}" r="5" '
                                  f'fill="{colour}"/>')
            elif kind == "light":
                self.parts.append(f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" '
                                  f'rx="2" fill="{colour}" fill-opacity="0.28" stroke="{colour}" '
                                  f'stroke-width="1.5"/>')
            else:
                dash = ' stroke-dasharray="5 4"' if kind == "dash" else ""
                self.parts.append(f'<line x1="{x - 2:.1f}" y1="{y - 4:.1f}" x2="{x + 14:.1f}" '
                                  f'y2="{y - 4:.1f}" stroke="{colour}" stroke-width="2"{dash}/>')
            self.text(x + 18, y, label, size=12)
            x += width
        return y

    def header(self, title, subtitle):
        self.text(24, 32, title, size=15, color="ink", weight=600)
        self.text(24, 52, subtitle, size=12)

    def save(self, path: Path) -> Path:
        w, h = self.width, self.height
        frame = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family=\'{FONT}\' role="img">',
            f"<title>{escape(self.title)}</title>",
            f'<rect width="{w}" height="{h}" rx="8" fill="{self.t["surface"]}"/>',
        ]
        path.write_text("\n".join(frame + self.parts + ["</svg>"]) + "\n")
        return path


def log_scale(lo, hi, a, b):
    span = math.log10(hi) - math.log10(lo)
    return lambda v: a + (math.log10(v) - math.log10(lo)) / span * (b - a)


def linear_scale(lo, hi, a, b):
    return lambda v: a + (v - lo) / (hi - lo) * (b - a)


def direct_labels(svg: Svg, x: float, slots: list[tuple[float, str]]) -> None:
    """Writes series labels at the right end of a line chart, nudged apart so they never overlap."""

    previous = -1e9
    for label_y, text in sorted(slots):
        label_y = max(label_y, previous + 15)
        svg.text(x, label_y + 4, text, size=12)
        previous = label_y


def fmt_gflops(value: float) -> str:
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}" if value >= 1 else f"{value:.2f}"


def fmt_tick(value: float) -> str:
    return f"{value:,.0f}" if value >= 1 else f"{value:g}"


# --------------------------------------------------------------------------- data access


def _load_summary(path: Path) -> dict[tuple[str, int, int, int, int], float]:
    values = {}
    with path.open() as file:
        for row in csv.DictReader(file):
            key = (row["family"], int(row["m"]), int(row["n"]), int(row["k"]), int(row["threads"]))
            values[key] = float(row["median_gflops"])
    return values


def _square_series(values, family: str) -> list[tuple[int, float]]:
    return sorted((m, v) for (f, m, n, k, _), v in values.items() if f == family and m == n == k)


# --------------------------------------------------------------------------- charts


def ladder_chart(values, theme: str, date: str) -> Path:
    """Every optimization step at 2048^3, double, as horizontal bars on a log axis."""

    size = 2048
    order = ["NaiveIjk", "ScalarIkj", "BlockedPortable", "BlockedAvx2", "OpenBlas",
             "BlockedAvx2Mt", "OpenBlasMt"]
    gflops = {f: values[(f"Gemm{f}_f64", size, size, size, CORES if f.endswith("Mt") else 1)]
              for f in order}
    naive = gflops["NaiveIjk"]

    width, row_h, top = 920, 38, 132
    svg = Svg(width, top + row_h * len(order) + 56, theme, "GEMM optimization ladder, 2048^3")
    svg.header(f"GEMM 2048 × 2048 × 2048, double (GFLOP/s, log scale, higher is better)",
               f"Median of 5 runs · {MACHINE} · {date}")
    svg.legend(24, 80, [("our implementation", "ink2", "dot"),
                        ("OpenBLAS 0.3.32 (same thread count)", "ink2", "light")])
    plot_l, plot_r = 262, width - 190
    scale = log_scale(0.1, 1000, plot_l, plot_r)
    axis_y = top + row_h * len(order) + 2
    for tick in (0.1, 1, 10, 100, 1000):
        svg.line(scale(tick), top - 18, scale(tick), axis_y, "grid")
        svg.text(scale(tick), axis_y + 16, fmt_tick(tick), size=11, color="muted", anchor="middle")
    for cores, label in ((1, "1-core peak"), (CORES, "6-core peak")):
        peak = peak_gflops("f64", cores)
        svg.line(scale(peak), top - 18, scale(peak), axis_y, "muted", dash="2 4")
        svg.text(scale(peak) + 4, top - 22, f"{label} {peak:.1f}", size=11, color="muted")
    svg.line(plot_l, top - 18, plot_l, axis_y, "base")
    for row, family in enumerate(order):
        cy = top + row * row_h + row_h / 2
        label, colour = VARIANTS[family]
        value = gflops[family]
        light = family.startswith("OpenBlas")
        svg.text(plot_l - 12, cy + 4, label, size=12, color="ink2", anchor="end")
        svg.hbar(plot_l, scale(value), cy, 22, colour,
                 f"{label}: {value:.2f} GFLOP/s, {value / naive:,.0f}× naive", light=light)
        svg.text(scale(value) + 8, cy + 4, f"{fmt_gflops(value)}", size=12,
                 color="ink", weight=600 if not light else 400, halo=True)
        svg.text(scale(value) + 8 + 7.2 * len(fmt_gflops(value)) + 6, cy + 4,
                 f"{value / naive:,.0f}× naive", size=12, color="muted", halo=True)
    return svg.save(IMAGE_DIR / f"gemm-ladder-2048-{theme}.svg")


def _line_panel(svg, x0, y0, panel_w, panel_h, values, precision, title):
    plot_l, plot_r = x0 + 46, x0 + panel_w - 8
    plot_t, plot_b = y0 + 26, y0 + panel_h - 30
    x = log_scale(64, 4096, plot_l, plot_r)
    y = log_scale(0.1, 1000, plot_b, plot_t)
    svg.text(x0, y0 + 10, title, size=13, color="ink", weight=600)
    for tick in (0.1, 1, 10, 100, 1000):
        svg.line(plot_l, y(tick), plot_r, y(tick), "grid")
        svg.text(plot_l - 8, y(tick) + 4, fmt_tick(tick), size=11, color="muted", anchor="end")
    for size in SQUARE_SIZES:
        svg.text(x(size), plot_b + 18, str(size), size=11, color="muted", anchor="middle")
    svg.line(plot_l, plot_b, plot_r, plot_b, "base")
    for cores in (1, CORES):
        peak = peak_gflops(precision, cores)
        svg.line(plot_l, y(peak), plot_r, y(peak), "muted", dash="2 4")
        svg.text(plot_r, y(peak) - 5, f"{cores}-core peak {peak:.0f}", size=10, color="muted",
                 anchor="end")
    for family, (label, colour) in VARIANTS.items():
        points = _square_series(values, f"Gemm{family}_{precision}")
        if not points:
            continue
        openblas = family.startswith("OpenBlas")
        svg.polyline([(x(n), y(v)) for n, v in points], colour, dash="5 4" if openblas else None)
        for n, v in points:
            svg.dot(x(n), y(v), colour, f"{label}, {precision}, n={n}: {v:.2f} GFLOP/s",
                    hollow=openblas, radius=3.5)


def size_chart(values, theme: str, date: str) -> Path:
    """GFLOP/s against matrix size for every variant; double and float side by side."""

    width, panel_w, panel_h = 920, 427, 330
    svg = Svg(width, 128 + panel_h + 10, theme, "GEMM throughput by matrix size")
    svg.header("GEMM throughput by matrix size (GFLOP/s, log-log, higher is better)",
               f"Square n × n × n, median of 5 runs · {MACHINE} · {date}")
    entries = [(VARIANTS[f][0], VARIANTS[f][1], "dash" if f.startswith("OpenBlas") else "dot")
               for f in VARIANTS]
    bottom = svg.legend(24, 78, entries, max_x=width - 24)
    top = bottom + 18
    _line_panel(svg, 24, top, panel_w, panel_h, values, "f64", "double")
    _line_panel(svg, 24 + panel_w + 18, top, panel_w, panel_h, values, "f32", "float")
    svg.height = top + panel_h + 10  # the legend may have wrapped onto a second row
    return svg.save(IMAGE_DIR / f"gemm-size-scaling-{theme}.svg")


def openblas_chart(values, theme: str, date: str) -> Path:
    """Our throughput as a percentage of OpenBLAS with the same thread count."""

    width, height = 920, 420
    svg = Svg(width, height, theme, "GEMM relative to OpenBLAS")
    svg.header("Throughput relative to OpenBLAS 0.3.32 with the same thread count (%)",
               f"100% = parity · square n × n × n, medians of 5 runs · {MACHINE}"
               f" · {date}")
    series = (("f64", 1, "double, 1 thread", "avx2", None),
              ("f64", CORES, "double, 6 threads", "avx2mt", None),
              ("f32", 1, "float, 1 thread", "avx2", "5 4"),
              ("f32", CORES, "float, 6 threads", "avx2mt", "5 4"))
    svg.legend(24, 78, [(label, colour, "dash" if dash else "dot")
                        for _, _, label, colour, dash in series])
    plot_l, plot_r, plot_t, plot_b = 64, width - 170, 104, height - 48
    x = log_scale(64, 4096, plot_l, plot_r)
    y = linear_scale(0, 110, plot_b, plot_t)
    for tick in (0, 25, 50, 75, 100):
        svg.line(plot_l, y(tick), plot_r, y(tick), "base" if tick in (0, 100) else "grid",
                 width=1.5 if tick == 100 else 1)
        svg.text(plot_l - 10, y(tick) + 4, f"{tick}%", size=11, color="muted", anchor="end")
    for size in SQUARE_SIZES:
        svg.text(x(size), plot_b + 18, str(size), size=11, color="muted", anchor="middle")
    label_slots = []
    for precision, threads, label, colour, dash in series:
        ours = "BlockedAvx2Mt" if threads > 1 else "BlockedAvx2"
        blas = "OpenBlasMt" if threads > 1 else "OpenBlas"
        mine = dict(_square_series(values, f"Gemm{ours}_{precision}"))
        reference = dict(_square_series(values, f"Gemm{blas}_{precision}"))
        points = [(n, 100 * mine[n] / reference[n]) for n in sorted(mine)]
        svg.polyline([(x(n), y(p)) for n, p in points], colour, dash=dash)
        for n, p in points:
            svg.dot(x(n), y(p), colour,
                    f"{label}, n={n}: {p:.0f}% of OpenBLAS ({mine[n]:.1f} vs {reference[n]:.1f} "
                    f"GFLOP/s)", hollow=dash is not None, radius=3.5)
        label_slots.append((y(points[-1][1]), f"{label}  {points[-1][1]:.0f}%"))
    direct_labels(svg, plot_r + 10, label_slots)
    return svg.save(IMAGE_DIR / f"gemm-vs-openblas-{theme}.svg")


def thread_chart(values, theme: str, date: str) -> Path:
    """Speedup over 1..6 threads for four sizes, against the ideal line."""

    width, height = 920, 420
    svg = Svg(width, height, theme, "GEMM thread scaling")
    svg.header("Thread scaling, blocked AVX2/FMA GEMM, double (speedup over 1 thread)",
               f"One thread per physical core · medians of 5 runs · {MACHINE}"
               f" · {date}")
    ramp = svg.t["ramp"]
    entries = [(f"{size}³", ramp[i], "dot") for i, size in enumerate(THREAD_SIZES)]
    svg.legend(24, 78, entries + [("ideal (linear)", "muted", "dash")])
    plot_l, plot_r, plot_t, plot_b = 64, width - 190, 104, height - 48
    x = linear_scale(1, CORES, plot_l, plot_r)
    y = linear_scale(0, CORES, plot_b, plot_t)
    for tick in range(0, CORES + 1):
        svg.line(plot_l, y(tick), plot_r, y(tick), "base" if tick == 0 else "grid")
        svg.text(plot_l - 10, y(tick) + 4, f"{tick}×", size=11, color="muted", anchor="end")
    for threads in range(1, CORES + 1):
        svg.text(x(threads), plot_b + 18, str(threads), size=11, color="muted", anchor="middle")
    svg.text((plot_l + plot_r) / 2, plot_b + 36, "threads", size=11, color="muted",
             anchor="middle")
    svg.line(x(1), y(1), x(CORES), y(CORES), "muted", width=1.5, dash="5 4")
    label_slots = []
    for i, size in enumerate(THREAD_SIZES):
        gflops = [values[("MatMulBlockedAvx2FmaThreads", size, size, size, t)]
                  for t in range(1, CORES + 1)]
        speedup = [g / gflops[0] for g in gflops]
        svg.polyline([(x(t), y(s)) for t, s in zip(range(1, CORES + 1), speedup)], ramp[i])
        for t, (s, g) in enumerate(zip(speedup, gflops), start=1):
            svg.dot(x(t), y(s), ramp[i], f"{size}³, {t} threads: {s:.2f}× "
                    f"({g:.1f} GFLOP/s, efficiency {100 * s / t:.0f}%)")
        label_slots.append((y(speedup[-1]),
                            f"{size}³  {speedup[-1]:.2f}× ({100 * speedup[-1] / CORES:.0f}%)"))
    direct_labels(svg, plot_r + 10, label_slots)
    return svg.save(IMAGE_DIR / f"gemm-thread-scaling-{theme}.svg")


def perf_chart(perf_path: Path, theme: str, date: str) -> Path:
    """FLOP per cycle per core and the share of cycles stalled on L2 misses, per variant."""

    with perf_path.open() as file:
        rows = list(csv.DictReader(file))
    width, row_h, top = 920, 34, 118
    svg = Svg(width, top + row_h * len(rows) + 50, theme, "GEMM hardware counters")
    svg.header("Why each step is faster: perf stat at 1024 × 1024 × 1024, double",
               f"Left: FLOP per cycle per core (peak 16). Right: cycles stalled on L2 misses. "
               f"{MACHINE} · {date}")
    label_w = 238
    panels = ((label_w + 24, 590, "flop_per_cycle_per_core", 18, (0, 4, 8, 12, 16),
               lambda v: f"{v:.2f}"),
              (630, width - 40, "l2_miss_stall_share", 0.5, (0, 0.1, 0.2, 0.3, 0.4, 0.5),
               lambda v: f"{100 * v:.1f}%"))
    axis_y = top + row_h * len(rows) + 2
    svg.text(panels[0][0], top - 20, "FLOP per cycle per core", size=12, color="ink",
             weight=600)
    svg.text(panels[1][0], top - 20, "cycles waiting for memory (L2 miss)", size=12,
             color="ink", weight=600)
    for left, right, field, maximum, ticks, fmt in panels:
        scale = linear_scale(0, maximum, left, right)
        for tick in ticks:
            svg.line(scale(tick), top - 8, scale(tick), axis_y, "grid" if tick else "base")
            svg.text(scale(tick), axis_y + 16,
                     f"{tick:g}" if field.startswith("flop") else f"{100 * tick:.0f}%",
                     size=11, color="muted", anchor="middle")
        if field.startswith("flop"):
            svg.line(scale(16), top - 8, scale(16), axis_y, "muted", dash="2 4")
            svg.text(scale(16) - 4, top - 2, "peak 16", size=10, color="muted", anchor="end")
        for row_index, row in enumerate(rows):
            cy = top + row_index * row_h + row_h / 2
            label, colour = VARIANTS[row["variant"]]
            value = float(row[field])
            light = row["variant"].startswith("OpenBlas")
            svg.hbar(left, max(scale(value), left + 1), cy, 20, colour,
                     f"{label}: {fmt(value)}", light=light)
            svg.text(max(scale(value), left + 1) + 6, cy + 4, fmt(value), size=11, color="ink2")
    for row_index, row in enumerate(rows):
        cy = top + row_index * row_h + row_h / 2
        svg.text(label_w + 12, cy + 4, VARIANTS[row["variant"]][0], size=12, anchor="end")
    return svg.save(IMAGE_DIR / f"gemm-perf-counters-{theme}.svg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-results", type=Path,
                        help="tools/run_benchmarks.sh output directory to summarize first")
    parser.add_argument("--date", required=True, help="date tag of the summary, e.g. 2026-10-18")
    arguments = parser.parse_args()

    if arguments.from_results:
        for path in summarize(arguments.from_results, arguments.date):
            print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")
    values = _load_summary(DATA_DIR / f"gemm-{arguments.date}.csv")
    perf = DATA_DIR / f"gemm-perf-{arguments.date}.csv"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        for path in (ladder_chart(values, theme, arguments.date),
                     size_chart(values, theme, arguments.date),
                     openblas_chart(values, theme, arguments.date),
                     thread_chart(values, theme, arguments.date),
                     perf_chart(perf, theme, arguments.date)):
            print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
