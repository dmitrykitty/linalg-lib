#include <linalg/internal/gemm.hpp>
#include <linalg/matrix.hpp>
#include <linalg/operations.hpp>

#include <benchmark/benchmark.h>

#include <cstdint>
#include <thread>


namespace {

using size_type = linalg::Matrix::size_type;
using Matrix = linalg::Matrix;

void clear_result(Matrix& result) noexcept {
    for (size_type index = 0; index < result.size(); ++index) {
        result.data()[index] = 0.0;
    }
}

//each result(row, col) calculated one by one (rows by rows)
void accumulate_ijk(
    Matrix& result,
    const Matrix& left,
    const Matrix& right) {
    for (size_type row = 0; row < left.rows(); ++row) {
        for (size_type col = 0; col < right.cols(); ++col) {
            for (size_type inner = 0; inner < left.cols(); ++inner) {
                result(row, col) += left(row, inner) * right(inner, col);
            }
        }
    }
}

void multiply_ijk_kernel(Matrix& result, const Matrix& left, const Matrix& right) {
    clear_result(result);
    accumulate_ijk(result, left, right);
}

// Keep an independent ijk oracle and the historical reference end-to-end case.
Matrix multiply_reference(const Matrix& left, const Matrix& right) {
    Matrix result(left.rows(), right.cols());
    accumulate_ijk(result, left, right);
    return result;
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
    const Matrix expected = multiply_reference(left, right);
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

template <typename Operation>
void benchmark_end_to_end(benchmark::State& state, Operation operation) {
    const auto rows = static_cast<size_type>(state.range(0));
    const auto cols = static_cast<size_type>(state.range(1));
    const auto inner = static_cast<size_type>(state.range(2));
    const Matrix left = make_input(rows, inner, 3);
    const Matrix right = make_input(inner, cols, 11);

    for (auto _ : state) {
        (void)_;
        Matrix result = operation(left, right);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    set_matmul_counters(state, rows, cols, inner);
}

void BM_MatMulReferenceEndToEnd(benchmark::State& state) {
    benchmark_end_to_end(state, multiply_reference);
}

void BM_MatMulLibraryEndToEnd(benchmark::State& state) {
    benchmark_end_to_end(state, [](const Matrix& left, const Matrix& right) {
        return linalg::multiply(left, right);
    });
}

void gemm_into(Matrix& result, const Matrix& left, const Matrix& right,
               linalg::internal::GemmKernel kernel, size_type threads) {
    linalg::internal::gemm(left.rows(), right.cols(), left.cols(), left.data(), left.cols(),
                           right.data(), right.cols(), result.data(), result.cols(), kernel,
                           threads);
}

// The path used by linalg::multiply for dynamic matrices: blocked GEMM, best micro-kernel, and
// the automatic thread count. Registered with UseRealTime() because work runs on pool threads.
// The per-kernel blocked variants (portable, AVX2, 1 or all cores) live in
// gemm_comparison_benchmark.cpp.
void BM_MatMulLibraryKernel(benchmark::State& state) {
    benchmark_kernel(state, [](Matrix& result, const Matrix& left, const Matrix& right) {
        gemm_into(result, left, right, linalg::internal::GemmKernel::automatic, 0);
    });
}

// AVX2/FMA GEMM with an explicit thread count in the fourth argument (M/N/K/threads).
void BM_MatMulBlockedAvx2FmaThreads(benchmark::State& state) {
    if (!linalg::internal::gemm_kernel_available(linalg::internal::GemmKernel::avx2_fma)) {
        state.SkipWithError("AVX2/FMA is not supported by this CPU");
        return;
    }
    const auto threads = static_cast<size_type>(state.range(3));
    // gemm() caps the thread count at the logical CPU count, so larger requests would silently
    // repeat a smaller measurement (for example 12 threads with SMT disabled).
    if (threads > std::thread::hardware_concurrency()) {
        state.SkipWithMessage("more threads requested than logical CPUs are online");
        return;
    }
    benchmark_kernel(state, [threads](Matrix& result, const Matrix& left, const Matrix& right) {
        gemm_into(result, left, right, linalg::internal::GemmKernel::avx2_fma, threads);
    });
    // Not "threads": Google Benchmark already writes a "threads" field (its own benchmark threads)
    // into the JSON, and a counter of the same name would duplicate the key.
    state.counters["gemm_threads"] = static_cast<double>(threads);
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

// Square baseline for "naive ijk vs ikj vs blocked" comparisons. 2048^3 naive ijk takes tens of
// seconds per iteration; filter these out for quick runs with --benchmark_filter='-/2048/'.
void add_large_square_sizes(benchmark::Benchmark* registration) {
    registration->Args({1024, 1024, 1024});
    registration->Args({2048, 2048, 2048});
}

// Thread scaling: 1..6 physical cores, plus 12 to show the effect of SMT siblings. Run with
// SMT disabled (or a taskset over one thread per core) when measuring physical-core scaling.
void add_thread_scaling_sizes(benchmark::Benchmark* registration) {
    for (const std::int64_t threads : {1, 2, 3, 4, 5, 6, 12}) {
        registration->Args({256, 256, 256, threads});
        registration->Args({512, 512, 512, threads});
        registration->Args({1024, 1024, 1024, threads});
        registration->Args({2048, 2048, 2048, threads});
    }
    for (const std::int64_t threads : {1, 6}) {
        // Small squares locate the serial/parallel crossover used by the automatic thread count.
        for (const std::int64_t size : {64, 96, 128, 160, 192}) {
            registration->Args({size, size, size, threads});
        }
        registration->Args({64, 1024, 1024, threads});
        registration->Args({1024, 64, 1024, threads});
        registration->Args({1024, 1024, 64, threads});
    }
}

BENCHMARK(BM_MatMulReferenceEndToEnd)->Apply(add_matmul_sizes);
BENCHMARK(BM_MatMulLibraryEndToEnd)->Apply(add_matmul_sizes)->UseRealTime();
BENCHMARK(BM_MatMulLibraryKernel)
    ->Apply(add_matmul_sizes)
    ->Apply(add_large_square_sizes)
    ->UseRealTime();
BENCHMARK(BM_MatMulBlockedAvx2FmaThreads)->Apply(add_thread_scaling_sizes)->UseRealTime();
BENCHMARK(BM_MatMulIjkKernel)->Apply(add_matmul_sizes)->Apply(add_large_square_sizes);
BENCHMARK(BM_MatMulIkjKernel)->Apply(add_matmul_sizes)->Apply(add_large_square_sizes);
BENCHMARK(BM_MatMulJikKernel)->Apply(add_matmul_sizes);
BENCHMARK(BM_MatMulJkiKernel)->Apply(add_matmul_sizes);
BENCHMARK(BM_MatMulKijKernel)->Apply(add_matmul_sizes);
BENCHMARK(BM_MatMulKjiKernel)->Apply(add_matmul_sizes);

} // namespace
