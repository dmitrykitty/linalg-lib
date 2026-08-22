# Linear Algebra Library

A numerical linear algebra library in modern C++. The project is designed to
demonstrate ownership-aware C++ design, numerical reasoning, benchmarking, profiling, cache-aware
optimization, SIMD, and concurrency through incremental, measured development.

The project currently has its owning `Matrix` and `Vector` foundations plus the first Phase 2
operations: arithmetic, broadcasting, transpose, trace, and vector/matrix norms. Performance
experiments compare alternative traversal and access strategies before changing public
implementations. More advanced numerical algorithms and optimized matrix multiplication kernels
are intentionally being added incrementally.

## Implemented functionality

### Matrix

- contiguous row-major `double` storage;
- default, dimension, fill, flat-sequence, and nested initializer-list construction;
- copy and move ownership semantics;
- `rows()`, `cols()`, `size()`, `empty()`, and contiguous `data()` access;
- unchecked `operator()(row, col)` and checked `at(row, col)` access;
- in-place matrix addition and subtraction;
- in-place scalar addition, subtraction, and multiplication;
- value-returning addition, subtraction, and scalar multiplication;
- adding a vector to every row or every column;
- constructing a matrix by repeating a vector as rows or columns;
- transpose and trace;
- one, Frobenius, and infinity norms selected with `MatrixNorm`.

### Vector

- contiguous owning `double` storage;
- default, size, fill, initializer-list, and `std::span` construction;
- copy and move ownership semantics;
- `size()`, `empty()`, and contiguous `data()` access;
- unchecked `operator[]` and checked `at()` access;
- in-place vector addition and subtraction;
- in-place scalar addition, subtraction, and multiplication;
- value-returning addition, subtraction, and scalar multiplication;
- L1, L2, and infinity norms selected with `VectorNorm`.

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

Vector addition and subtraction require equal sizes. Matrix addition and subtraction require equal
shapes. Broadcast operations require a vector length matching the relevant matrix dimension.
Trace requires a square, non-empty matrix. Invalid dimensions throw `std::invalid_argument`;
checked element access throws `std::out_of_range`.

## Requirements

- CMake 3.20 or newer
- A C++20 compiler such as GCC or Clang
- Git and network access when GoogleTest or Google Benchmark are not installed locally

GoogleTest 1.17.0 and Google Benchmark 1.9.5 are development-only dependencies. CMake first looks
for installed packages and otherwise fetches the pinned releases.

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

Benchmarks are separated by operation family in `benchmarks/`, while one executable and Google
Benchmark's regular-expression filter provide a common runner. Current experiments include
broadcasting, transpose, and trace. Kernel-only timings isolate element-processing loops;
end-to-end transpose timings also include result allocation.

The main findings so far are:

- transpose performance is dominated by traversal order and contiguous destination writes;
- after inlining, indexed and pointer implementations with the same traversal compile and perform
  similarly;
- manually replacing indexed access with pointers did not produce a consistent improvement for
  `add_to_each_row`;
- direct-data and `operator()` trace implementations are currently indistinguishable within the
  observed measurement noise.

See [docs/performance.md](docs/performance.md) for commands, environment details, measured tables,
limitations, and interpretation. Published performance claims should always be based on repeatable
Release measurements on documented hardware.

## Repository layout

```text
include/linalg/   Public headers
src/              Library implementation
tests/            GoogleTest correctness tests
benchmarks/       Google Benchmark experiments
examples/         Small API usage programs
cmake/            Project CMake modules
docs/             Architecture and measured performance notes
```

