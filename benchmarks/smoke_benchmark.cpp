#include <linalg/matrix.hpp>

#include <benchmark/benchmark.h>

namespace {

void BM_FrameworkSmoke(benchmark::State& state) {
    for (auto _ : state) {
        (void)_;
        linalg::Matrix matrix;
        benchmark::DoNotOptimize(matrix);
    }
}

} // namespace

BENCHMARK(BM_FrameworkSmoke);

