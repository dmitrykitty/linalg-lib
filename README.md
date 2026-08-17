# Linear Algebra Library

A learning-focused numerical linear algebra library in modern C++. The project is designed to
demonstrate ownership-aware C++ design, numerical reasoning, benchmarking, profiling, cache-aware
optimization, SIMD, and concurrency through incremental, measured development.

The project currently has its owning `Matrix` and `Vector` foundations plus the first Phase 2
arithmetic and matrix-broadcast operations. More advanced numerical algorithms and optimized
matrix multiplication kernels are intentionally being added incrementally. See [PLAN.md](PLAN.md)
for the complete roadmap.

## Implemented functionality

### Matrix

- contiguous row-major `double` storage;
- default, dimension, fill, flat-sequence, and nested initializer-list construction;
- copy and move ownership semantics;
- `rows()`, `cols()`, `size()`, `empty()`, and contiguous `data()` access;
- unchecked `operator()(row, col)` and checked `at(row, col)` access;
- in-place matrix addition and subtraction;
- in-place scalar addition, subtraction, and multiplication;
- adding a vector to every row or every column;
- constructing a matrix by repeating a vector as rows or columns.

### Vector

- contiguous owning `double` storage;
- default, size, fill, initializer-list, and `std::span` construction;
- copy and move ownership semantics;
- `size()`, `empty()`, and contiguous `data()` access;
- unchecked `operator[]` and checked `at()` access;
- in-place vector addition and subtraction;
- in-place scalar addition, subtraction, and multiplication.

## Basic usage

```cpp
#include <linalg/linalg.hpp>

int main() {
    linalg::Matrix matrix{{1.0, 2.0, 3.0},
                          {4.0, 5.0, 6.0}};
    const linalg::Vector row_values{10.0, 20.0, 30.0};

    const linalg::Matrix adjusted =
        linalg::add_to_each_row(matrix, row_values);

    linalg::Vector vector{1.0, 2.0, 3.0};
    const linalg::Vector other{4.0, 5.0, 6.0};

    vector += other;
    vector -= 1.0;
    vector *= 2.0;

    return adjusted.at(0, 0) == 11.0 && vector.at(0) == 8.0 ? 0 : 1;
}
```

For `Vector`, braces contain values while parentheses specify a size:

```cpp
linalg::Vector one_value{3.0}; // one element containing 3.0
linalg::Vector three_zeroes(3); // three zero-initialized elements
```

Vector addition and subtraction require equal sizes. Matrix addition and subtraction require equal
shapes. Broadcast operations require a vector length matching the relevant matrix dimension.
Invalid dimensions throw `std::invalid_argument`; checked element access throws
`std::out_of_range`.

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

To run only the current `add_to_each_row` traversal experiment:

```bash
./build/release/benchmarks/linalg_benchmarks \
    --benchmark_filter=AddToEachRow \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true
```

This benchmark compares the current indexed implementation with an experimental contiguous-memory
implementation. It reports both end-to-end timings, which include result allocation and copying,
and kernel-only timings, which isolate the element-processing loop. Published performance claims
should always be based on repeatable Release measurements on documented hardware.

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

## License

No project license has been selected yet. Choose one before publishing or accepting external
contributions.

