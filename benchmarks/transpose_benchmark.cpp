#include <linalg/matrix.hpp>
#include <linalg/operations.hpp>

#include <benchmark/benchmark.h>

#include <cstdint>

namespace {

using size_type = linalg::Matrix::size_type;

linalg::Matrix make_transpose_input(size_type rows, size_type cols) {
    linalg::Matrix matrix(rows, cols);
    for (size_type index = 0; index < matrix.size(); ++index) {
        matrix.data()[index] = static_cast<double>(index);
    }
    return matrix;
}

bool matrices_equal(const linalg::Matrix& left, const linalg::Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        return false;
    }

    for (size_type index = 0; index < left.size(); ++index) {
        if (left.data()[index] != right.data()[index]) {
            return false;
        }
    }
    return true;
}

void transpose_indexed_col_row_traversal_kernel(
    const linalg::Matrix& input,
    linalg::Matrix& output) {
    for (size_type col = 0; col < input.cols(); ++col) {
        for (size_type row = 0; row < input.rows(); ++row) {
            output(col, row) = input(row, col);
        }
    }
}

void transpose_indexed_row_col_traversal_kernel(
    const linalg::Matrix& input,
    linalg::Matrix& output) {
    for (size_type row = 0; row < input.rows(); ++row) {
        for (size_type col = 0; col < input.cols(); ++col) {
            output(col, row) = input(row, col);
        }
    }
}

void transpose_source_row_pointer_kernel(
    const linalg::Matrix& input,
    linalg::Matrix& output) {
    if (input.empty()) {
        return;
    }

    const size_type source_columns = input.cols();
    const size_type destination_columns = output.cols();
    const double* source = input.data();
    double* destination = output.data();

    for (size_type row = 0; row < input.rows(); ++row) {
        const double* source_row = source + row * source_columns;
        for (size_type col = 0; col < source_columns; ++col) {
            destination[col * destination_columns + row] = source_row[col];
        }
    }
}

void transpose_destination_row_pointer_kernel(
    const linalg::Matrix& input,
    linalg::Matrix& output) {
    if (input.empty()) {
        return;
    }

    const size_type source_columns = input.cols();
    const size_type destination_columns = output.cols();
    const double* source = input.data();
    double* destination = output.data();

    for (size_type col = 0; col < source_columns; ++col) {
        double* destination_row = destination + col * destination_columns;
        for (size_type row = 0; row < input.rows(); ++row) {
            destination_row[row] = source[row * source_columns + col];
        }
    }
}

linalg::Matrix transpose_source_row_pointer(const linalg::Matrix& input) {
    linalg::Matrix result(input.cols(), input.rows());
    transpose_source_row_pointer_kernel(input, result);
    return result;
}

linalg::Matrix transpose_destination_row_pointer(const linalg::Matrix& input) {
    linalg::Matrix result(input.cols(), input.rows());
    transpose_destination_row_pointer_kernel(input, result);
    return result;
}

void set_transpose_counters(
    benchmark::State& state,
    const linalg::Matrix& input) {
    const auto elements = static_cast<std::int64_t>(input.size());
    const auto bytes_per_element = static_cast<std::int64_t>(2U * sizeof(double));
    state.SetItemsProcessed(state.iterations() * elements);
    state.SetBytesProcessed(state.iterations() * elements * bytes_per_element);
}

template <typename Operation>
void benchmark_transpose_end_to_end(
    benchmark::State& state,
    Operation operation) {
    const auto rows = static_cast<size_type>(state.range(0));
    const auto cols = static_cast<size_type>(state.range(1));
    const linalg::Matrix input = make_transpose_input(rows, cols);
    const linalg::Matrix expected = linalg::transpose(input);

    if (!matrices_equal(operation(input), expected)) {
        state.SkipWithError("transpose candidate produced an incorrect result");
        return;
    }

    for (auto _ : state) {
        (void)_;
        linalg::Matrix result = operation(input);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    set_transpose_counters(state, input);
}

template <typename Kernel>
void benchmark_transpose_kernel(benchmark::State& state, Kernel kernel) {
    const auto rows = static_cast<size_type>(state.range(0));
    const auto cols = static_cast<size_type>(state.range(1));
    const linalg::Matrix input = make_transpose_input(rows, cols);
    const linalg::Matrix expected = linalg::transpose(input);
    linalg::Matrix result(cols, rows);

    kernel(input, result);
    if (!matrices_equal(result, expected)) {
        state.SkipWithError("transpose kernel produced an incorrect result");
        return;
    }

    for (auto _ : state) {
        (void)_;
        kernel(input, result);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    set_transpose_counters(state, input);
}

void BM_TransposeIndexedEndToEnd(benchmark::State& state) {
    benchmark_transpose_end_to_end(state, linalg::transpose);
}

void BM_TransposeSourceRowPointerEndToEnd(benchmark::State& state) {
    benchmark_transpose_end_to_end(state, transpose_source_row_pointer);
}

void BM_TransposeDestinationRowPointerEndToEnd(benchmark::State& state) {
    benchmark_transpose_end_to_end(state, transpose_destination_row_pointer);
}

void BM_TransposeIndexedColRowKernel(benchmark::State& state) {
    benchmark_transpose_kernel(state, transpose_indexed_col_row_traversal_kernel);
}

void BM_TransposeIndexedRowColKernel(benchmark::State& state) {
    benchmark_transpose_kernel(state, transpose_indexed_row_col_traversal_kernel);
}

void BM_TransposeSourceRowPointerKernel(benchmark::State& state) {
    benchmark_transpose_kernel(state, transpose_source_row_pointer_kernel);
}

void BM_TransposeDestinationRowPointerKernel(benchmark::State& state) {
    benchmark_transpose_kernel(state, transpose_destination_row_pointer_kernel);
}

void add_transpose_sizes(benchmark::Benchmark* registration) {
    registration->Args({64, 64});
    registration->Args({256, 256});
    registration->Args({512, 512});
    registration->Args({1024, 1024});
    registration->Args({64, 1024});
    registration->Args({1024, 64});
    registration->Args({511, 513});
}

BENCHMARK(BM_TransposeIndexedEndToEnd)->Apply(add_transpose_sizes);
BENCHMARK(BM_TransposeSourceRowPointerEndToEnd)->Apply(add_transpose_sizes);
BENCHMARK(BM_TransposeDestinationRowPointerEndToEnd)->Apply(add_transpose_sizes);
BENCHMARK(BM_TransposeIndexedColRowKernel)->Apply(add_transpose_sizes);
BENCHMARK(BM_TransposeIndexedRowColKernel)->Apply(add_transpose_sizes);
BENCHMARK(BM_TransposeSourceRowPointerKernel)->Apply(add_transpose_sizes);
BENCHMARK(BM_TransposeDestinationRowPointerKernel)->Apply(add_transpose_sizes);

} // namespace
