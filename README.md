# Linear Algebra Library

A numerical linear algebra library in modern C++. The project is designed to
demonstrate ownership-aware C++ design, numerical reasoning, benchmarking, profiling, cache-aware
optimization, SIMD, and concurrency through incremental, measured development.

The project currently has its owning `Matrix` and `Vector` foundations plus the first Phase 2
operations: arithmetic, broadcasting, transpose, multiplication, trace, and vector/matrix norms.
Matrix multiplication runs on a cache-blocked, packed GEMM with an AVX2/FMA micro-kernel and a
dynamic-scheduling thread pool. Every optimization is measured before it becomes the default; see
[Performance](#performance).

**Headline (double, 2048 x 2048 x 2048, i7-10750H at a fixed 2.6 GHz):** 33.7 GFLOP/s on one core
(81% of peak, 94% of OpenBLAS), 172 GFLOP/s on six cores (5.1x, 88% of OpenBLAS), 116x faster than
a naive `ijk` loop, with results bitwise identical for any thread count.

## Implemented functionality

### Matrix

- contiguous row-major `float` or `double` storage through `DynamicMatrix<T>`;
- `Matrix` remains an alias for `DynamicMatrix<double>`;
- default, dimension, fill, flat-sequence, and nested initializer-list construction;
- copy and move ownership semantics;
- `rows()`, `cols()`, `size()`, `empty()`, and contiguous `data()` access;
- unchecked `operator()(row, col)` and checked `at(row, col)` access;
- in-place matrix addition and subtraction;
- in-place scalar addition, subtraction, and multiplication;
- value-returning addition, subtraction, and scalar multiplication;
- matrix-vector multiplication;
- matrix-matrix multiplication through a cache-blocked, packed GEMM: AVX2/FMA micro-kernel chosen
  at run time (portable fallback), multithreaded above 2 x 128^3 FLOP;
- adding a vector to every row or every column;
- constructing a matrix by repeating a vector as rows or columns;
- transpose and trace;
- one, Frobenius, and infinity norms selected with `MatrixNorm`.

For a float matrix, use `linalg::DynamicMatrix<float>`:

```cpp
linalg::DynamicMatrix<float> small{{1.0f, 2.0f}, {3.0f, 4.0f}};
small *= 2.0f;
```

Construction, access, arithmetic, transpose, matrix-vector multiplication, and matrix-matrix
multiplication work for both scalar types. Trace, norms, broadcasting, and repeating vectors
currently accept the double-precision aliases; their float versions are a later migration step.

### Static matrix

`StaticMatrix<T, Rows, Cols>` stores a fixed number of `float` or `double` elements directly
inside the object using `std::array`. Its dimensions are part of the type:

```cpp
linalg::StaticMatrix<double, 2, 3> matrix{{1.0, 2.0, 3.0}, {4.0}};
matrix.at(1, 2) = 6.0;

constexpr linalg::StaticMatrix<float, 2, 2> identity{{1.0f, 0.0f}, {0.0f, 1.0f}};
static_assert(identity(1, 1) == 1.0f);
```

Static matrices provide construction, ownership, and the same core access API as dynamic
matrices: `rows()`, `cols()`, `size()`, `empty()`, `data()`, `operator()`, and `at()`. Transpose,
matrix-matrix multiplication, and matrix-vector multiplication support both scalar types and
return fixed-shape results. Elementwise arithmetic, norms, trace, and dynamic/static conversions
are not provided for static matrices yet.

- Default construction fills every element with zero.
- Parentheses fill every element: `StaticMatrix<double, 2, 3>(4.0)`.
- `StaticMatrix<double, 2, 3>::from_values({4.0})` initializes only the first element to 4.0.
  The named flat-list factory avoids ambiguity with nested lists such as `{{1.0}, {2.0}}`.
- `from_values()` and explicit `std::span<const T>` construction copy row-major values and
  zero-pad the rest.
- Nested lists zero-pad short rows and missing rows. Too many rows, too many columns in a row,
  or too many flat values throw `std::invalid_argument`; the fixed shape never expands.
- Zero dimensions are supported. Shapes whose storage-size calculation would overflow are
  rejected at compile time.
- Copies own independent storage. Compiler-generated moves operate on the inline array;
  both objects keep their fixed shape, and moving `float`/`double` elements copies their values.

Construction, access, transpose, and multiplication support `constexpr`, so valid small examples
can be evaluated at compile time. Ordinary calls still execute at runtime. Inline storage avoids
a separate heap allocation,
but the object's location determines whether that storage is on the stack, heap, or elsewhere.
Prefer dynamic matrices for large or runtime-sized shapes.

Both owning types share initialization and indexing helpers in `include/linalg/internal/`;
storage and dimension validation remain specific to each type. This keeps the implementation
small without introducing inheritance or a storage-policy framework.

### Vector

- contiguous owning `float` or `double` storage through `DynamicVector<T>`;
- `Vector` remains an alias for `DynamicVector<double>`;
- default, size, fill, initializer-list, and `std::span` construction;
- copy and move ownership semantics;
- `size()`, `empty()`, and contiguous `data()` access;
- unchecked `operator[]` and checked `at()` access;
- in-place vector addition and subtraction;
- in-place scalar addition, subtraction, and multiplication;
- value-returning addition, subtraction, and scalar multiplication;
- L2 normalization with validation for zero and non-finite vectors;
- L1, L2, and infinity norms selected with `VectorNorm`.

### Static vector

`StaticVector<T, Size>` owns a fixed-length `std::array` of `float` or `double` elements.
It provides construction, copy/move semantics, `size()`, `empty()`, `data()`, `operator[]`,
and checked `at()` access:

```cpp
linalg::StaticVector<double, 3> values{1.0, 2.0}; // {1.0, 2.0, 0.0}
linalg::StaticVector<double, 3> filled(4.0); // {4.0, 4.0, 4.0}
values.at(2) = 3.0;

constexpr linalg::StaticVector<float, 3> axis{1.0f, 0.0f, 0.0f};
static_assert(axis.size() == 3 && axis[0] == 1.0f);
```

The template specifies the length; parentheses fill every element rather than supplying a
runtime size. Braces contain values and zero-pad any remaining elements. Explicit
`std::span<const T>` construction likewise copies values and pads missing elements. Excess
input throws `std::invalid_argument`; `at()` throws `std::out_of_range` for invalid indices.
Unchecked access requires a valid index and asserts that precondition in Debug builds.

Zero-length vectors are supported, and lengths whose byte-size calculation would overflow
are rejected at compile time. Inline storage has no separate heap allocation. Copies and
moves use the compiler-generated `std::array` behavior: each object retains its fixed length
and independent storage, and moving `float`/`double` values copies them.

Construction and access support `constexpr`. Multiplying a matching static matrix by a static
vector returns a static vector and also supports constant evaluation. Static vector arithmetic,
norms, normalization, and dynamic/static conversions remain later steps.

Matrices and static vectors reuse input validation and copying helpers in
`include/linalg/internal/storage.hpp`.
Dynamic and static vectors also share checked and unchecked access helpers; their owning storage
remains separate.

## Basic usage

```cpp
#include <linalg/linalg.hpp>

int main() {
    linalg::Matrix matrix{{1.0, 2.0, 3.0},
                          {4.0, 5.0, 6.0}};
    const linalg::Vector row_values{10.0, 20.0, 30.0};

    const linalg::Matrix adjusted =
        linalg::add_to_each_row(matrix, row_values);
    const linalg::Matrix transposed = linalg::transpose(matrix);

    const double matrix_size =
        linalg::norm(matrix, linalg::MatrixNorm::frobenius);

    linalg::Vector vector{1.0, 2.0, 3.0};
    const linalg::Vector other{4.0, 5.0, 6.0};

    vector += other;
    vector -= 1.0;
    vector *= 2.0;

    const double vector_length =
        linalg::norm(vector, linalg::VectorNorm::l2);

    return adjusted.at(0, 0) == 11.0 &&
                   transposed.at(0, 1) == 4.0 &&
                   matrix_size > 0.0 && vector_length > 0.0
               ? 0
               : 1;
}
```

For `Vector`, braces contain values while parentheses specify a size:

```cpp
linalg::Vector one_value{3.0}; // one element containing 3.0
linalg::Vector three_zeroes(3); // three zero-initialized elements
```

`DynamicVector<T>` likewise supports `float` and `double`. Its constructors, element access,
and arithmetic work for both types:

```cpp
linalg::DynamicVector<float> samples{1.0f, 2.0f, 3.0f};
samples *= 2.0f;
```

Existing `linalg::Vector` code still uses `double`. Vector norms, normalization, and broadcast
functions currently take the double-precision aliases; their float versions are a later migration
step. Matrix-vector multiplication supports both scalar types. Operations between different
scalar types are not provided yet; convert elements explicitly when needed.

Vector addition and subtraction require equal sizes. Matrix addition and subtraction require equal
shapes. Broadcast operations require a vector length matching the relevant matrix dimension.
Trace requires a square, non-empty matrix. Invalid dimensions throw `std::invalid_argument`;
checked element access throws `std::out_of_range`.

### Shared transpose and multiplication

The public operations select the result container; each calculation lives in one shared kernel
under `include/linalg/internal/kernels.hpp`. These internal helpers overwrite a preallocated,
non-aliasing result and are not public API. No inheritance or duplicated dynamic/static loops are
needed.

```cpp
linalg::DynamicMatrix<float> dynamic{{1.0f, 2.0f}, {3.0f, 4.0f}};
auto dynamic_product = linalg::multiply(dynamic, dynamic); // DynamicMatrix<float>

constexpr linalg::StaticMatrix<double, 2, 3> fixed{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
constexpr linalg::StaticVector<double, 3> vector{7.0, 8.0, 9.0};
constexpr auto transposed = linalg::transpose(fixed); // StaticMatrix<double, 3, 2>
constexpr auto applied = linalg::multiply(fixed, vector); // StaticVector<double, 2>
constexpr auto product = linalg::multiply(fixed, transposed); // StaticMatrix<double, 2, 2>
static_assert(applied[0] == 50.0 && product(0, 0) == 14.0);
```

Dynamic multiplication checks dimensions at runtime and throws `std::invalid_argument` on a
mismatch. Incompatible static multiplication shapes have no matching overload and are rejected
at compile time. Public multiplication currently requires the same scalar type and either two
dynamic operands or two static operands; mixed ownership is not exposed yet.

The library uses indexed column-row transpose (contiguous destination writes). Static matrices
multiply with the `constexpr` `ikj` kernel (contiguous reads of the right matrix and updates of the
result), which suits small fixed shapes and constant evaluation. Dynamic matrices multiply with the
blocked GEMM below.

### Blocked GEMM

`DynamicMatrix` multiplication calls `internal::gemm` ([src/gemm.cpp](src/gemm.cpp)), a
GotoBLAS/BLIS-style implementation:

- **cache blocking and packing:** five loops sized to the cache levels (packed B panel in L3,
  packed A block in each core's L2, B micro-panel in L1); packing makes every kernel read sequential
  and aligned, and zero-pads edges;
- **AVX2/FMA micro-kernel:** a 6 x 8 (double) or 6 x 16 (float) tile held in 12 `ymm`
  accumulators, software prefetch of the packed panels and the C tile; enabled per function with
  `[[gnu::target("avx2,fma")]]` and selected with a runtime CPU check, so the library still runs on
  any x86-64 (and elsewhere with the portable kernel);
- **threads:** a persistent `std::jthread` pool; each `pc` step is split into chunks taken from an
  atomic counter, so a slow or preempted thread does fewer chunks instead of stalling the others.
  The k dimension is never split, so results are bitwise identical for any thread count. One
  thread is used below 2 x 128^3 FLOP, where waking the pool costs more than it saves; the default
  otherwise is the number of physical cores the process may use.

[docs/performance.md](docs/performance.md) documents each step as a measured experiment.

## Requirements

- CMake 3.20 or newer
- A C++20 compiler such as GCC or Clang
- Git and network access when GoogleTest or Google Benchmark are not installed locally
- Optional: OpenBLAS (`libopenblas-dev` on Ubuntu) to add BLAS baselines to the GEMM benchmarks
- Optional: Python 3 (standard library only) to redraw the performance charts

GoogleTest 1.17.0 and Google Benchmark 1.9.5 are development-only dependencies. CMake first looks
for installed packages and otherwise fetches the pinned releases. OpenBLAS is found through
`pkg-config` and used only by the benchmarks (`-DLINALG_BENCHMARK_OPENBLAS=OFF` disables it).

## Build and test

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

## Sanitizers

```bash
cmake --preset sanitized
cmake --build --preset sanitized
ctest --preset sanitized --output-on-failure
```

ThreadSanitizer cannot be combined with AddressSanitizer, so it has its own preset:

```bash
cmake --preset tsan
cmake --build --preset tsan
ctest --preset tsan --output-on-failure
```

## Release build and benchmarks

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
./build/release/benchmarks/linalg_benchmarks
```

List registered benchmarks before choosing a filter:

```bash
./build/release/benchmarks/linalg_benchmarks --benchmark_list_tests
```

Run a quick comparison of one operation and size:

```bash
./build/release/benchmarks/linalg_benchmarks \
    --benchmark_filter='^BM_Transpose.*Kernel/512/512$' \
    --benchmark_min_time=0.1s \
    --benchmark_repetitions=7 \
    --benchmark_enable_random_interleaving=true \
    --benchmark_report_aggregates_only=true
```

Run all six matmul loop orders for one shape. Arguments are M/N/K, representing
an M-by-K matrix multiplied by a K-by-N matrix:

```bash
./build/release/benchmarks/linalg_benchmarks \
    --benchmark_filter='^BM_MatMul.*Kernel/256/256/256$' \
    --benchmark_min_time=0.1s \
    --benchmark_repetitions=7 \
    --benchmark_enable_random_interleaving=true \
    --benchmark_report_aggregates_only=true
```

Compare every GEMM optimization step and OpenBLAS at one size (`_f64` or `_f32`; OpenBLAS
families appear when it is installed):

```bash
./build/release/benchmarks/linalg_benchmarks \
    --benchmark_filter='^BM_Gemm(ScalarIkj|BlockedAvx2|BlockedAvx2Mt)_f64/1024/' \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true
```

Benchmarks are separated by operation family in `benchmarks/`, while one executable and Google
Benchmark's regular-expression filter provide a common runner. `gemm_comparison_benchmark.cpp`
holds the GEMM ladder (naive `ijk`, `ikj`, blocked portable, blocked AVX2 on one and all cores,
OpenBLAS); `multiply_benchmark.cpp` holds the six scalar loop orders and thread scaling. Kernel-only
timings use a preallocated result, while end-to-end timings also include result allocation and
destruction. Multithreaded benchmarks report wall-clock time.

### Full controlled run and charts

Publishable numbers come from [tools/run_benchmarks.sh](tools/run_benchmarks.sh). It refuses to
run unless the CPU governor is `performance`, turbo and SMT are off, no CPUs are isolated, and the
machine is idle (or marks the results uncontrolled with `--force`), records the environment, and
writes JSON plus `perf stat` counters to `benchmark-results/`:

```bash
tools/run_benchmarks.sh
python3 tools/plot_gemm_results.py --from-results benchmark-results/<run> --date <YYYY-MM-DD>
```

The second command writes the CSV summaries in `docs/data/` and the light/dark SVG charts in
`docs/images/`.

## Performance

Controlled run on 2026-10-18: Intel i7-10750H, 6 cores at a fixed 2.6 GHz (turbo off), SMT off,
GCC 15.2, OpenBLAS 0.3.32 as the baseline, median of 5 repetitions. Peak is 41.6 GFLOP/s per core in
double (16 FLOP/cycle x 2.6 GHz).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/gemm-ladder-2048-dark.svg">
  <img alt="GFLOP/s at 2048 cubed for naive ijk, ikj, blocked portable, blocked AVX2 on 1 and 6 threads, and OpenBLAS on 1 and 6 threads, on a log scale" src="docs/images/gemm-ladder-2048-light.svg">
</picture>

| double, 2048^3 | GFLOP/s | % of peak | vs naive `ijk` | vs OpenBLAS |
|---|---:|---:|---:|---:|
| naive `ijk` | 0.29 | 0.7% | 1x | |
| `ikj` loop order | 3.00 | 7.2% | 10x | |
| blocked + packed, portable kernel | 5.23 | 12.6% | 18x | |
| **blocked + AVX2/FMA, 1 thread** | **33.7** | **80.9%** | **116x** | **94%** |
| **blocked + AVX2/FMA, 6 threads** | **172.1** | **68.9%** | **595x** | **88%** |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/gemm-vs-openblas-dark.svg">
  <img alt="Our throughput as a percentage of OpenBLAS for double and float on 1 and 6 threads across sizes" src="docs/images/gemm-vs-openblas-light.svg">
</picture>

What each step contributes, measured with `perf stat` at 1024^3:

- **loop order** (`ijk` -> `ikj`): no more column walks over B; cycles stalled on L2 misses fall
  from 42% to 8%;
- **cache blocking and packing:** memory stalls fall to 1.9%, but a scalar SSE2 kernel stays at
  2 FLOP/cycle;
- **AVX2/FMA micro-kernel:** the same data flow at 12.9 FLOP/cycle, 81% of peak;
- **six threads with dynamic chunks:** 5.1x (85% parallel efficiency at 1024-2048^3); dynamic
  scheduling recovered 12-20% over a static split, and prefetching the packed panels across page
  boundaries another 5-17%.

The [performance documentation](docs/performance.md) has the size sweep from 64^3 to 4096^3, float
results, thread scaling, hardware counters, each experiment's method and A/B data, and the
limitations. The earlier scalar loop-order experiment is kept there as
[Experiment 4](docs/performance.md#experiment-4----scalar-matmul-loop-ordering).

## Repository layout

```text
include/linalg/   Public headers
src/              Library implementation
tests/            GoogleTest correctness tests
benchmarks/       Google Benchmark experiments
examples/         Small API usage programs
cmake/            Project CMake modules
docs/             Measured performance notes, CSV summaries (docs/data), charts (docs/images)
tools/            Benchmark runner (run_benchmarks.sh) and chart generator (plot_gemm_results.py)
```
