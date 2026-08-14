# Linear Algebra Library

A learning-focused numerical linear algebra library in modern C++. The project is designed to
demonstrate ownership-aware C++ design, numerical reasoning, benchmarking, profiling, cache-aware
optimization, SIMD, and concurrency through incremental, measured development.

The repository is currently at **Phase 0 complete / Phase 1 skeleton**. The public namespace and
empty `Matrix` and `Vector` types exist, but no mathematical functionality has been implemented.
See `PLAN.md` for the roadmap and `AGENTS_URGENT.md` for the current scope boundary.

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
ctest --preset debug
```

## Sanitizers

```bash
cmake --preset sanitized
cmake --build --preset sanitized
ctest --preset sanitized
```

## Release benchmark smoke check

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
./build/release/benchmarks/linalg_benchmarks --benchmark_dry_run
```

The current benchmark only validates the Google Benchmark integration. It must not be interpreted
as a library performance result.

## Repository layout

```text
include/linalg/   Public headers
src/              Library implementation
tests/            GoogleTest correctness tests
benchmarks/       Google Benchmark executables
examples/         Small API usage programs
cmake/            Project CMake modules
docs/             Architecture and measured performance notes
```

## License

No project license has been selected yet. Choose one before publishing or accepting external
contributions.

