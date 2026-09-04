# Benchmark plot visualizer

This optional Python tool turns Google Benchmark JSON output into documentation-ready SVG or PNG
plots. It is separate from the C++ library and adds no production dependency.

The tool supports two plot styles:

- **comparison** -- a horizontal bar chart comparing implementations for one argument shape;
- **scaling** -- one line per implementation across several argument shapes.

`auto` mode selects comparison when one shape is present or explicitly selected and scaling when
the JSON contains several shapes.

## Files

```text
tools/plot_visualizer/
    __init__.py         Public Python imports
    __main__.py         `python -m` entry point
    generate_matmul_figures.py  Rebuilds the committed matmul report figures
    plot_benchmarks.py  JSON parsing, plotting API, and CLI
    requirements.txt    Optional Matplotlib dependency
    README.md           This guide
```

## Installation

Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r tools/plot_visualizer/requirements.txt
```

The virtual environment is ignored by Git. Activate it again in a new terminal before generating
plots:

```bash
source .venv/bin/activate
```

## 1. Save benchmark results as JSON

Google Benchmark only writes JSON when both output options are provided:

```bash
mkdir -p benchmark-results
./build/release/benchmarks/linalg_benchmarks \
    --benchmark_filter='^BM_MatMul.*Kernel' \
    --benchmark_min_time=0.1s \
    --benchmark_repetitions=7 \
    --benchmark_enable_random_interleaving=true \
    --benchmark_report_aggregates_only=true \
    --benchmark_out=benchmark-results/matmul.json \
    --benchmark_out_format=json
```

Raw files under `benchmark-results/` are intentionally ignored by Git because results are specific
to one machine and measurement environment.

## 2. Inspect the available data

Before selecting options, list the metrics and argument shapes stored in the file:

```bash
python3 -m tools.plot_visualizer benchmark-results/matmul.json --list
```

Example output:

```text
Metrics: cpu_time, real_time, FLOP/s
Shapes: 64/64/64, 128/128/128, 256/256/256
```

## 3. Create plots

The shortest command uses `cpu_time`, the median statistic, automatic plot selection, and an SVG
output beside the JSON file:

```bash
python3 -m tools.plot_visualizer benchmark-results/matmul.json
```

Create a focused kernel comparison for one matrix shape:

```bash
python3 -m tools.plot_visualizer benchmark-results/matmul.json \
    --shape=512/512/512 \
    --metric='FLOP/s' \
    --output=docs/images/matmul-512.svg
```

Create a scaling plot across every shape:

```bash
python3 -m tools.plot_visualizer benchmark-results/matmul.json \
    --kind=scaling \
    --metric=cpu_time \
    --log-y \
    --output=docs/images/matmul-scaling.svg
```

Use a regular expression to include only selected benchmark names:

```bash
python3 -m tools.plot_visualizer benchmark-results/all.json \
    --filter='^BM_Transpose' \
    --output=docs/images/transpose.svg
```

The output format is selected by the filename extension. SVG is recommended for the repository
because it stays sharp at different display sizes. Use an output ending in `.png` when a raster
image is required.

## Rebuild the published matmul figures

The repository's current matmul report uses a reproducible two-panel square-scaling figure and a
three-panel rectangular-shape figure. Rebuild both from the selected committed aggregate data:

```bash
python3 -m tools.plot_visualizer.generate_matmul_figures
```

Input:

```text
docs/data/matmul-loop-orders-2026-08-30.json
```

Outputs:

```text
docs/images/matmul-square-loop-orders.png
docs/images/matmul-rectangular-loop-orders.png
```

## CLI options

```text
input                 Google Benchmark JSON path
--output, -o          SVG or PNG output path
--metric              cpu_time, real_time, or a benchmark counter such as FLOP/s
--aggregate           median, mean, min, or max; median is the default
--filter              regular expression applied to benchmark run names
--shape               slash-separated benchmark arguments, for example 512/512/512
--kind                auto, comparison, or scaling
--title               custom chart title
--log-y               logarithmic y-axis for scaling plots
--list                print available metrics and shapes without plotting
```

Run `python3 -m tools.plot_visualizer --help` for the generated command help.

## Python API

Use the public functions when a documentation script needs more control:

```python
from tools.plot_visualizer import load_results, plot_comparison

results = load_results(
    "benchmark-results/matmul.json",
    aggregate="median",
    benchmark_filter=r"^BM_MatMul.*Kernel",
)

plot_comparison(
    results,
    "docs/images/matmul-512.svg",
    metric="FLOP/s",
    arguments=(512, 512, 512),
)
```

Available public functions:

- `load_results(...)` parses JSON and returns `BenchmarkResults`;
- `plot_comparison(...)` plots implementations for one argument shape;
- `plot_scaling(...)` plots implementations across all shapes;
- `plot_benchmarks(...)` loads and plots in one call.

The parser uses Google Benchmark's existing aggregate rows when available. If the JSON contains raw
repetitions, it calculates the requested median, mean, minimum, or maximum itself. Benchmark times
are normalized internally to seconds and displayed using readable nanosecond, microsecond,
millisecond, or second units.

## Add a plot to Markdown

Keep selected publication plots under `docs/images/`, then reference them normally.

From the repository-level `README.md`:

```markdown
![Matrix multiplication loop-order benchmark](docs/images/matmul-512.svg)
```

From `docs/performance.md`:

```markdown
![Matrix multiplication loop-order benchmark](images/matmul-512.svg)
```

Commit a plot only together with enough context to reproduce and interpret it: CPU, compiler,
build flags, benchmark command, statistic, and measurement limitations.

## Troubleshooting

### `No module named matplotlib`

Activate the virtual environment and install the requirements:

```bash
source .venv/bin/activate
python3 -m pip install -r tools/plot_visualizer/requirements.txt
```

### Unknown metric or shape

Inspect the file before plotting:

```bash
python3 -m tools.plot_visualizer benchmark-results/matmul.json --list
```

Metric names are case-sensitive because they come directly from Google Benchmark counters.
