#include <linalg/matrix.hpp>
#include <linalg/operations.hpp>
#include <linalg/vector.hpp>

#include <benchmark/benchmark.h>

#include <cstdint>
#include <stdexcept>

namespace {

using size_type = linalg::Matrix::size_type;

void add_to_each_row_indexed_kernel(
    linalg::Matrix& matrix,
    const linalg::Vector& values) {
    for (size_type row = 0; row < matrix.rows(); ++row) {
        for (size_type col = 0; col < matrix.cols(); ++col) {
            matrix(row, col) += values[col];
        }
    }
}

void add_to_each_row_contiguous_kernel(
    linalg::Matrix& matrix,
    const linalg::Vector& values) {
    if (matrix.empty()) {
        return;
    }

    double* output = matrix.data();
    const double* increments = values.data();
    const size_type column_count = matrix.cols();

    for (size_type row = 0; row < matrix.rows(); ++row) {
        for (size_type col = 0; col < column_count; ++col) {
            output[col] += increments[col];
        }
        output += column_count;
    }
}

linalg::Matrix add_to_each_row_contiguous(
    const linalg::Matrix& matrix,
    const linalg::Vector& values) {
    if (values.size() != matrix.cols()) {
        throw std::invalid_argument("row vector size must equal matrix column count");
    }

    linalg::Matrix result(matrix);
    add_to_each_row_contiguous_kernel(result, values);
    return result;
}

template <typename Operation>
void benchmark_end_to_end(benchmark::State& state, Operation operation) {
    const auto rows = static_cast<size_type>(state.range(0));
    const auto cols = static_cast<size_type>(state.range(1));
    const linalg::Matrix matrix(rows, cols, 1.0);
    const linalg::Vector values(cols, 2.0);

    for (auto _ : state) {
        (void)_;
        linalg::Matrix result = operation(matrix, values);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    const auto element_count = static_cast<std::int64_t>(matrix.size());
    state.SetItemsProcessed(state.iterations() * element_count);
}

template <typename Operation>
void benchmark_kernel(benchmark::State& state, Operation operation) {
    const auto rows = static_cast<size_type>(state.range(0));
    const auto cols = static_cast<size_type>(state.range(1));
    linalg::Matrix matrix(rows, cols, 1.0);
    const linalg::Vector values(cols, 2.0);

    for (auto _ : state) {
        (void)_;
        operation(matrix, values);
        benchmark::DoNotOptimize(matrix.data());
        benchmark::ClobberMemory();
    }

    const auto element_count = static_cast<std::int64_t>(matrix.size());
    state.SetItemsProcessed(state.iterations() * element_count);
}

void BM_AddToEachRowCurrentEndToEnd(benchmark::State& state) {
    benchmark_end_to_end(state, linalg::add_to_each_row);
}

void BM_AddToEachRowContiguousEndToEnd(benchmark::State& state) {
    benchmark_end_to_end(state, add_to_each_row_contiguous);
}

void BM_AddToEachRowIndexedKernel(benchmark::State& state) {
    benchmark_kernel(state, add_to_each_row_indexed_kernel);
}

void BM_AddToEachRowContiguousKernel(benchmark::State& state) {
    benchmark_kernel(state, add_to_each_row_contiguous_kernel);
}

void add_matrix_sizes(benchmark::Benchmark* registration) {
    registration->Args({64, 64});
    registration->Args({256, 256});
    registration->Args({512, 512});
    registration->Args({1024, 1024});
    registration->Args({64, 1024});
    registration->Args({1024, 64});
}

BENCHMARK(BM_AddToEachRowCurrentEndToEnd)->Apply(add_matrix_sizes);
BENCHMARK(BM_AddToEachRowContiguousEndToEnd)->Apply(add_matrix_sizes);
BENCHMARK(BM_AddToEachRowIndexedKernel)->Apply(add_matrix_sizes);
BENCHMARK(BM_AddToEachRowContiguousKernel)->Apply(add_matrix_sizes);

} // namespace
