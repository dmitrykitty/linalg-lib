#include <linalg/matrix.hpp>
#include <linalg/operations.hpp>

#include <benchmark/benchmark.h>

#include <cstdint>


namespace {

using size_type = linalg::Matrix::size_type;
using Matrix = linalg::Matrix;

void clear_result(Matrix& result) noexcept {
    for (size_type index = 0; index < result.size(); ++index) {
        result.data()[index] = 0.0;
    }
}

//each result(row, col) calculated one by one (rows by rows)
void multiply_ijk_kernel(
    Matrix& result,
    const Matrix& left,
    const Matrix& right) {
    clear_result(result);
    for (size_type row = 0; row < left.rows(); ++row) {
        for (size_type col = 0; col < right.cols(); ++col) {
            for (size_type inner = 0; inner < left.cols(); ++inner) {
                result(row, col) += left(row, inner) * right(inner, col);
            }
        }
    }
}

//each row from result calculated partially 
//(num1, num2, num3)   -> (num1 + num1.1, num2 + num2.2, num3 + num3.3)
//(0,    0,    0)          -------//--------
void multiply_ikj_kernel(
    Matrix& result,
    const Matrix& left,
    const Matrix& right) {
    clear_result(result);
    for (size_type row = 0; row < left.rows(); ++row) {
        for (size_type inner = 0; inner < left.cols(); ++inner) {
            for (size_type col = 0; col < right.cols(); ++col) {
                result(row, col) += left(row, inner) * right(inner, col);
            }
        }
    }
}

//each result(row, col) calculated one by one (cols by cols)
void multiply_jik_kernel(
    Matrix& result,
    const Matrix& left,
    const Matrix& right) {
    clear_result(result);
    for (size_type col = 0; col < right.cols(); ++col) {
        for (size_type row = 0; row < left.rows(); ++row) {
            for (size_type inner = 0; inner < left.cols(); ++inner) {
                result(row, col) += left(row, inner) * right(inner, col);
            }
        }
    }
}

//each col from result calculated partially 
//(num1, 0, 0)   -> (num1 + num1.1, 0, 0)
//(num2, 0, 0)   -> (num2 + num2.2, 0, 0)
void multiply_jki_kernel(
    Matrix& result,
    const Matrix& left,
    const Matrix& right) {
    clear_result(result);
    for (size_type col = 0; col < right.cols(); ++col) {
        for (size_type inner = 0; inner < left.cols(); ++inner) {
            for (size_type row = 0; row < left.rows(); ++row) {
                result(row, col) += left(row, inner) * right(inner, col);
            }
        }
    }
}

//each result(row, col) calculated partially row by row
void multiply_kij_kernel(
    Matrix& result,
    const Matrix& left,
    const Matrix& right) {
    clear_result(result);
    for (size_type inner = 0; inner < left.cols(); ++inner) {
        for (size_type row = 0; row < left.rows(); ++row) {
            for (size_type col = 0; col < right.cols(); ++col) {
                result(row, col) += left(row, inner) * right(inner, col);
            }
        }
    }
}

//each result(row, col) calculated partially col by col
void multiply_kji_kernel(
    Matrix& result,
    const Matrix& left,
    const Matrix& right) {
    clear_result(result);
    for (size_type inner = 0; inner < left.cols(); ++inner) {
        for (size_type col = 0; col < right.cols(); ++col) {
            for (size_type row = 0; row < left.rows(); ++row) {
                result(row, col) += left(row, inner) * right(inner, col);
            }
        }
    }
}

Matrix make_input(size_type rows, size_type cols, std::uint64_t offset) {
    Matrix matrix(rows, cols);
    for (size_type index = 0; index < matrix.size(); ++index) {
        const auto pattern = static_cast<std::int64_t>(
            (static_cast<std::uint64_t>(index) * 17U + offset) % 23U
        );
        //-11 to add negativer numbers as well, so final range [-2,75, 2.75]
        matrix.data()[index] = static_cast<double>(pattern - 11) / 4.0;
    }
    return matrix;
}

bool matrices_equal(const Matrix& left, const Matrix& right) {
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

template <typename Operation>
bool kernel_is_correct(Operation operation) {
    const Matrix left = make_input(5, 7, 3);
    const Matrix right = make_input(7, 3, 11);
    const Matrix expected = linalg::multiply(left, right);
    Matrix result(expected.rows(), expected.cols(), 42.0);

    operation(result, left, right);
    if (!matrices_equal(result, expected)) {
        return false;
    }

    operation(result, left, right);
    return matrices_equal(result, expected);
}

/*
Floating Point Operations
multiplications + additions
= M × N × K + M × N × K
= 2 × M × N × K
1 GFLOP/s = 1,000,000,000 FLOP/s
*/
void set_matmul_counters(
    benchmark::State& state,
    size_type rows,
    size_type cols,
    size_type inner) {
    const auto row_count = static_cast<std::int64_t>(rows);
    const auto column_count = static_cast<std::int64_t>(cols);
    const auto inner_count = static_cast<std::int64_t>(inner);
    const auto flop_count = 2 * row_count * column_count * inner_count;

    state.counters["FLOP/s"] = benchmark::Counter(
        static_cast<double>(flop_count),
        benchmark::Counter::kIsIterationInvariantRate);
}

template <typename Operation>
void benchmark_kernel(benchmark::State& state, Operation operation) {
    if (!kernel_is_correct(operation)) {
        state.SkipWithError("matmul kernel produced an incorrect result");
        return;
    }

    // A is rows x inner, B is inner x cols, and C is rows x cols.
    const auto rows = static_cast<size_type>(state.range(0));
    const auto cols = static_cast<size_type>(state.range(1));
    const auto inner = static_cast<size_type>(state.range(2));
    const Matrix left = make_input(rows, inner, 3);
    const Matrix right = make_input(inner, cols, 11);
    Matrix result(rows, cols);

    for (auto _ : state) {
        (void)_;
        operation(result, left, right);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    set_matmul_counters(state, rows, cols, inner);
}

void benchmark_reference_end_to_end(benchmark::State& state) {
    const auto rows = static_cast<size_type>(state.range(0));
    const auto cols = static_cast<size_type>(state.range(1));
    const auto inner = static_cast<size_type>(state.range(2));
    const Matrix left = make_input(rows, inner, 3);
    const Matrix right = make_input(inner, cols, 11);

    for (auto _ : state) {
        (void)_;
        Matrix result = linalg::multiply(left, right);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    set_matmul_counters(state, rows, cols, inner);
}

void BM_MatMulReferenceEndToEnd(benchmark::State& state) {
    benchmark_reference_end_to_end(state);
}

void BM_MatMulIjkKernel(benchmark::State& state) {
    benchmark_kernel(state, multiply_ijk_kernel);
}

void BM_MatMulIkjKernel(benchmark::State& state) {
    benchmark_kernel(state, multiply_ikj_kernel);
}

void BM_MatMulJikKernel(benchmark::State& state) {
    benchmark_kernel(state, multiply_jik_kernel);
}

void BM_MatMulJkiKernel(benchmark::State& state) {
    benchmark_kernel(state, multiply_jki_kernel);
}

void BM_MatMulKijKernel(benchmark::State& state) {
    benchmark_kernel(state, multiply_kij_kernel);
}

void BM_MatMulKjiKernel(benchmark::State& state) {
    benchmark_kernel(state, multiply_kji_kernel);
}

void add_matmul_sizes(benchmark::Benchmark* registration) {
    registration->Args({63, 65, 67});
    registration->Args({64, 64, 64});
    registration->Args({128, 128, 128});
    registration->Args({256, 256, 256});
    registration->Args({512, 512, 512});
    registration->Args({64, 1024, 1024});
    registration->Args({1024, 64, 1024});
    registration->Args({1024, 1024, 64});
}

BENCHMARK(BM_MatMulReferenceEndToEnd)->Apply(add_matmul_sizes);
BENCHMARK(BM_MatMulIjkKernel)->Apply(add_matmul_sizes);
BENCHMARK(BM_MatMulIkjKernel)->Apply(add_matmul_sizes);
BENCHMARK(BM_MatMulJikKernel)->Apply(add_matmul_sizes);
BENCHMARK(BM_MatMulJkiKernel)->Apply(add_matmul_sizes);
BENCHMARK(BM_MatMulKijKernel)->Apply(add_matmul_sizes);
BENCHMARK(BM_MatMulKjiKernel)->Apply(add_matmul_sizes);

} // namespace
