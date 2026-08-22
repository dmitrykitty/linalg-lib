#include <linalg/matrix.hpp>
#include <linalg/operations.hpp>

#include <benchmark/benchmark.h>

#include <cstdint>
#include <stdexcept>

namespace {

using size_type = linalg::Matrix::size_type;

double trace_data_access(const linalg::Matrix& matrix) {
    if (matrix.empty() || matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("matrix must be square and non-empty");
    }

    const double* data = matrix.data();
    const size_type diagonal_stride = matrix.cols() + 1;
    long double result = 0.0L;
    for (size_type index = 0; index < matrix.rows(); ++index) {
        result += data[index * diagonal_stride];
    }
    return static_cast<double>(result);
}

double trace_indexed(const linalg::Matrix& matrix) {
    if (matrix.empty() || matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("matrix must be square and non-empty");
    }

    long double result = 0.0L;
    for (size_type index = 0; index < matrix.rows(); ++index) {
        result += matrix(index, index);
    }
    return static_cast<double>(result);
}

template <typename Operation>
void benchmark_trace(benchmark::State& state, Operation operation) {
    const auto dimension = static_cast<size_type>(state.range(0));
    const linalg::Matrix matrix(dimension, dimension, 1.0);

    if (operation(matrix) != static_cast<double>(dimension)) {
        state.SkipWithError("trace candidate produced an incorrect result");
        return;
    }

    for (auto _ : state) {
        (void)_;
        benchmark::DoNotOptimize(matrix.data());
        benchmark::DoNotOptimize(operation(matrix));
    }

    const auto diagonal_elements = static_cast<std::int64_t>(dimension);
    const auto bytes_per_element = static_cast<std::int64_t>(sizeof(double));
    state.SetItemsProcessed(state.iterations() * diagonal_elements);
    state.SetBytesProcessed(
        state.iterations() * diagonal_elements * bytes_per_element);
}

void BM_TraceDataAccess(benchmark::State& state) {
    benchmark_trace(state, trace_data_access);
}

void BM_TraceIndexedAccess(benchmark::State& state) {
    benchmark_trace(state, trace_indexed);
}

void add_trace_sizes(benchmark::Benchmark* registration) {
    registration->Arg(64);
    registration->Arg(256);
    registration->Arg(512);
    registration->Arg(1024);
    registration->Arg(2048);
}

BENCHMARK(BM_TraceDataAccess)->Apply(add_trace_sizes);
BENCHMARK(BM_TraceIndexedAccess)->Apply(add_trace_sizes);

} // namespace
