#include <linalg/matrix.hpp>

#include <benchmark/benchmark.h>

static void BM_FrameworkSmoke(benchmark::State& state) {
    for (auto _ : state) {
        (void)_;
        linalg::Matrix matrix;
        benchmark::DoNotOptimize(matrix);
    }
}

BENCHMARK(BM_FrameworkSmoke);
BENCHMARK_MAIN();

