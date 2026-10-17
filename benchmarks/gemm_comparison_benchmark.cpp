// Head-to-head GEMM comparison: every optimization step of the library against the naive loop
// and, when available, OpenBLAS with the same thread counts.
//
// All variants compute C = A * B for row-major matrices into a preallocated C, use the same random
// inputs, and report wall-clock time (UseRealTime) so single- and multi-threaded rows are directly
// comparable. FLOP/s counts 2 * M * N * K operations per iteration. Every family exists for double
// (suffix _f64) and float (suffix _f32); float doubles the SIMD width and is the precision a GPU
// comparison needs.
//
// Families, in optimization order:
//   BM_GemmNaiveIjk          textbook ijk triple loop, 1 thread (sizes up to 2048)
//   BM_GemmScalarIkj         cache-friendly ikj loop order (previous library default), 1 thread
//                            (sizes up to 2048)
//   BM_GemmBlockedPortable   cache blocking + packing, plain C++ micro-kernel, 1 thread
//   BM_GemmBlockedAvx2       cache blocking + packing, AVX2/FMA micro-kernel, 1 thread
//   BM_GemmBlockedAvx2Mt     the above on gemm_default_thread_count() threads
//   BM_GemmOpenBlas          cblas_[ds]gemm, 1 thread                (needs LINALG_HAS_OPENBLAS)
//   BM_GemmOpenBlasMt        cblas_[ds]gemm, same thread count as Mt (needs LINALG_HAS_OPENBLAS)
//
// OpenBLAS worker threads keep spinning for a while after a call. Run the OpenBLAS families in a
// separate process from the library's multithreaded rows (use --benchmark_filter) rather than
// interleaving them, or the spinning threads steal cores from the measured variant.

#include <linalg/internal/gemm.hpp>
#include <linalg/internal/kernels.hpp>
#include <linalg/matrix.hpp>

#include <benchmark/benchmark.h>

#if LINALG_HAS_OPENBLAS
#include <cblas.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using linalg::internal::GemmKernel;
using size_type = std::size_t;

template <typename T>
using Matrix = linalg::DynamicMatrix<T>;

// C = A * B for row-major operands; C has already been allocated with the right shape.
template <typename T>
using GemmVariant = std::function<void(const Matrix<T>& a, const Matrix<T>& b, Matrix<T>& c)>;

template <typename T>
void naive_ijk(const Matrix<T>& a, const Matrix<T>& b, Matrix<T>& c) {
    for (size_type row = 0; row < a.rows(); ++row) {
        for (size_type col = 0; col < b.cols(); ++col) {
            T sum{};
            for (size_type inner = 0; inner < a.cols(); ++inner) {
                sum += a(row, inner) * b(inner, col);
            }
            c(row, col) = sum;
        }
    }
}

template <typename T>
void scalar_ikj(const Matrix<T>& a, const Matrix<T>& b, Matrix<T>& c) {
    linalg::internal::matmul_into(a, b, c);
}

template <typename T>
GemmVariant<T> blocked(GemmKernel kernel, size_type threads) {
    return [kernel, threads](const Matrix<T>& a, const Matrix<T>& b, Matrix<T>& c) {
        linalg::internal::gemm(a.rows(), b.cols(), a.cols(), a.data(), a.cols(), b.data(),
                               b.cols(), c.data(), c.cols(), kernel, threads);
    };
}

#if LINALG_HAS_OPENBLAS
template <typename T>
GemmVariant<T> openblas(size_type threads) {
    return [threads](const Matrix<T>& a, const Matrix<T>& b, Matrix<T>& c) {
        openblas_set_num_threads(static_cast<int>(threads));
        const auto m = static_cast<blasint>(a.rows());
        const auto n = static_cast<blasint>(b.cols());
        const auto k = static_cast<blasint>(a.cols());
        if constexpr (std::is_same_v<T, double>) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a.data(), k,
                        b.data(), n, 0.0, c.data(), n);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a.data(), k,
                        b.data(), n, 0.0f, c.data(), n);
        }
    };
}
#endif

template <typename T>
Matrix<T> random_matrix(size_type rows, size_type cols, std::uint32_t seed) {
    std::mt19937 engine(seed);
    std::uniform_real_distribution<T> distribution(T{-1}, T{1});
    Matrix<T> matrix(rows, cols);
    for (size_type index = 0; index < matrix.size(); ++index) {
        matrix.data()[index] = distribution(engine);
    }
    return matrix;
}

// Checks a variant against a double-accumulated ijk result on an odd-sized product before timing
// it, using the inner-product error bound |error| <= (k + 2) * eps * sum |a_ip * b_pj|.
template <typename T>
bool variant_is_correct(const GemmVariant<T>& variant) {
    const Matrix<T> a = random_matrix<T>(67, 131, 1);
    const Matrix<T> b = random_matrix<T>(131, 45, 2);
    Matrix<T> actual(a.rows(), b.cols(), std::numeric_limits<T>::quiet_NaN());
    variant(a, b, actual);

    const double epsilon = std::numeric_limits<T>::epsilon();
    for (size_type row = 0; row < a.rows(); ++row) {
        for (size_type col = 0; col < b.cols(); ++col) {
            double expected = 0.0;
            double magnitude = 0.0;
            for (size_type inner = 0; inner < a.cols(); ++inner) {
                const double product = static_cast<double>(a(row, inner)) * b(inner, col);
                expected += product;
                magnitude += std::abs(product);
            }
            const double tolerance = static_cast<double>(a.cols() + 2) * epsilon * magnitude;
            if (!(std::abs(actual(row, col) - expected) <= tolerance)) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
void run_variant(benchmark::State& state, const GemmVariant<T>& variant, size_type threads) {
    if (!variant_is_correct(variant)) {
        state.SkipWithError("GEMM variant produced an incorrect result");
        return;
    }

    const auto m = static_cast<size_type>(state.range(0));
    const auto n = static_cast<size_type>(state.range(1));
    const auto k = static_cast<size_type>(state.range(2));
    const Matrix<T> a = random_matrix<T>(m, k, 3);
    const Matrix<T> b = random_matrix<T>(k, n, 4);
    Matrix<T> c(m, n);

    for (auto _ : state) {
        (void)_;
        variant(a, b, c);
        benchmark::DoNotOptimize(c.data());
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                         static_cast<double>(k);
    state.counters["FLOP/s"] =
        benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
    // Not "threads": Google Benchmark already writes a "threads" field (its own benchmark threads)
    // into the JSON, and a counter of the same name would duplicate the key.
    state.counters["gemm_threads"] = static_cast<double>(threads);
}

template <typename T>
struct Family {
    std::string name;
    GemmVariant<T> variant;
    size_type threads;
    std::int64_t largest_size;
};

template <typename T>
std::vector<Family<T>> families() {
    const size_type cores = linalg::internal::gemm_default_thread_count();
    std::vector<Family<T>> result{
        // Naive and ikj are the slow baselines; 2048^3 naive ijk takes about a minute per run.
        {"BM_GemmNaiveIjk", naive_ijk<T>, 1, 2048},
        {"BM_GemmScalarIkj", scalar_ikj<T>, 1, 2048},
        {"BM_GemmBlockedPortable", blocked<T>(GemmKernel::portable, 1), 1, 4096},
    };
    if (linalg::internal::gemm_kernel_available(GemmKernel::avx2_fma)) {
        result.push_back({"BM_GemmBlockedAvx2", blocked<T>(GemmKernel::avx2_fma, 1), 1, 4096});
        result.push_back(
            {"BM_GemmBlockedAvx2Mt", blocked<T>(GemmKernel::avx2_fma, cores), cores, 4096});
    }
#if LINALG_HAS_OPENBLAS
    result.push_back({"BM_GemmOpenBlas", openblas<T>(1), 1, 4096});
    result.push_back({"BM_GemmOpenBlasMt", openblas<T>(cores), cores, 4096});
#endif
    return result;
}

// Square sizes from L1/L2-resident (64^3) to far beyond L3 (4096^3 double: 3 x 128 MiB operands).
// Non-powers of two sit between the powers of two: they show the trend without the cache-set
// conflicts that power-of-two row strides can cause, and they exercise the edge tiles.
constexpr std::int64_t square_sizes[] = {64,  96,   128,  192,  256,  384,  512,
                                         768, 1024, 1536, 2048, 3072, 4096};

template <typename T>
void register_families(const std::string& suffix) {
    for (Family<T>& family : families<T>()) {
        benchmark::Benchmark* registration = benchmark::RegisterBenchmark(
            family.name + suffix,
            [variant = family.variant, threads = family.threads](benchmark::State& state) {
                run_variant(state, variant, threads);
            });
        for (const std::int64_t size : square_sizes) {
            if (size <= family.largest_size) {
                registration->Args({size, size, size});
            }
        }
        registration->UseRealTime()->Unit(benchmark::kMillisecond);
    }
}

[[maybe_unused]] const bool registered = [] {
    register_families<double>("_f64");
    register_families<float>("_f32");
    return true;
}();

} // namespace
