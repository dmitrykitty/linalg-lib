#include <linalg/internal/gemm.hpp>
#include <linalg/operations.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using linalg::internal::GemmKernel;
using size_type = std::size_t;

std::vector<GemmKernel> available_kernels() {
    std::vector<GemmKernel> kernels{GemmKernel::automatic, GemmKernel::portable};
    if (linalg::internal::gemm_kernel_available(GemmKernel::avx2_fma)) {
        kernels.push_back(GemmKernel::avx2_fma);
    }
    return kernels;
}

std::string kernel_name(GemmKernel kernel) {
    switch (kernel) {
    case GemmKernel::automatic:
        return "automatic";
    case GemmKernel::portable:
        return "portable";
    case GemmKernel::avx2_fma:
        return "avx2_fma";
    }
    return "unknown";
}

// Row-major storage with an optional row stride larger than the logical column count.
template <typename T>
struct Strided {
    size_type rows;
    size_type cols;
    size_type stride;
    std::vector<T> values;

    Strided(size_type row_count, size_type col_count, size_type padding, T fill)
        : rows(row_count), cols(col_count), stride(col_count + padding),
          values(row_count * (col_count + padding), fill) {}

    T& operator()(size_type row, size_type col) {
        return values[row * stride + col];
    }

    const T& operator()(size_type row, size_type col) const {
        return values[row * stride + col];
    }
};

template <typename T>
Strided<T> random_matrix(size_type rows, size_type cols, size_type padding, std::mt19937& engine) {
    // Padding holds a sentinel that would poison the result if the kernel ever read it.
    Strided<T> matrix(rows, cols, padding, std::numeric_limits<T>::quiet_NaN());
    std::uniform_real_distribution<T> distribution(T{-1}, T{1});
    for (size_type row = 0; row < rows; ++row) {
        for (size_type col = 0; col < cols; ++col) {
            matrix(row, col) = distribution(engine);
        }
    }
    return matrix;
}

/*
Compares against an extended-precision ijk reference with the standard forward error bound for
inner products: |fl(c_ij) - c_ij| <= gamma_k * sum_p |a_ip| |b_pj|, gamma_k ~= k * epsilon / 2.
The test uses (k + 2) * epsilon, which leaves room for any summation order the blocked kernel
chooses while still catching any wrong, missing, or duplicated term.
*/
template <typename T>
void expect_matches_reference(const Strided<T>& a, const Strided<T>& b, const Strided<T>& c,
                              const std::string& context) {
    const size_type k = a.cols;
    const long double epsilon = std::numeric_limits<T>::epsilon();
    for (size_type row = 0; row < a.rows; ++row) {
        for (size_type col = 0; col < b.cols; ++col) {
            long double exact = 0.0L;
            long double magnitude = 0.0L;
            for (size_type inner = 0; inner < k; ++inner) {
                const long double product = static_cast<long double>(a(row, inner)) *
                                            static_cast<long double>(b(inner, col));
                exact += product;
                magnitude += std::fabs(product);
            }
            const long double actual = c(row, col);
            const long double tolerance =
                static_cast<long double>(k + 2) * epsilon * magnitude +
                static_cast<long double>(std::numeric_limits<T>::min());
            ASSERT_LE(std::fabs(actual - exact), tolerance)
                << context << " at (" << row << ", " << col << ")";
        }
    }
}

template <typename T>
void check_shape(size_type m, size_type n, size_type k, GemmKernel kernel, size_type padding = 0,
                 size_type threads = 0) {
    std::mt19937 engine(static_cast<std::mt19937::result_type>(m * 1'000'003U + n * 1'009U + k));
    const Strided<T> a = random_matrix<T>(m, k, padding, engine);
    const Strided<T> b = random_matrix<T>(k, n, padding, engine);
    // Stale contents must be overwritten; padding columns must be left untouched.
    const T sentinel = T{12345};
    Strided<T> c(m, n, padding, sentinel);
    for (size_type row = 0; row < m; ++row) {
        for (size_type col = 0; col < n; ++col) {
            c(row, col) = std::numeric_limits<T>::quiet_NaN();
        }
    }

    linalg::internal::gemm(m, n, k, a.values.data(), a.stride, b.values.data(), b.stride,
                           c.values.data(), c.stride, kernel, threads);

    const std::string context = kernel_name(kernel) + " " + std::to_string(m) + "x" +
                                std::to_string(n) + "x" + std::to_string(k) +
                                " padding=" + std::to_string(padding) +
                                " threads=" + std::to_string(threads);
    expect_matches_reference(a, b, c, context);
    for (size_type row = 0; row < m; ++row) {
        for (size_type col = n; col < c.stride; ++col) {
            ASSERT_EQ(c(row, col), sentinel) << context << " wrote padding at (" << row << ", "
                                             << col << ")";
        }
    }
}

// Identical bit patterns, not merely close values (also distinguishes 0.0 from -0.0).
template <typename T>
bool same_bits(T left, T right) {
    using Bits = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;
    return std::bit_cast<Bits>(left) == std::bit_cast<Bits>(right);
}

template <typename T>
class GemmTest : public ::testing::Test {};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(GemmTest, ScalarTypes);

TYPED_TEST(GemmTest, MatchesReferenceForSizesThatAreNotMultiplesOfTheRegisterTile) {
    const size_type edges[] = {1, 7, 13, 63, 65, 127, 129};
    const size_type depths[] = {1, 7, 65, 129};
    for (const GemmKernel kernel : available_kernels()) {
        for (const size_type m : edges) {
            for (const size_type n : edges) {
                for (const size_type k : depths) {
                    check_shape<TypeParam>(m, n, k, kernel);
                }
            }
        }
    }
}

TYPED_TEST(GemmTest, MatchesReferenceAcrossCacheBlockBoundaries) {
    const size_type depths_around_kc[] = {255, 256, 257, 513};
    const size_type rows_around_mc[] = {71, 72, 73, 143, 144, 145, 150}; // mc: 72 double, 144 float
    const size_type columns_around_nc[] = {4079, 4080, 4081, 8170};
    for (const GemmKernel kernel : available_kernels()) {
        for (const size_type k : depths_around_kc) {
            check_shape<TypeParam>(13, 17, k, kernel);
        }
        for (const size_type m : rows_around_mc) {
            check_shape<TypeParam>(m, 9, 5, kernel);
        }
        for (const size_type n : columns_around_nc) {
            check_shape<TypeParam>(7, n, 3, kernel);
        }
    }
}

TYPED_TEST(GemmTest, HonoursRowStridesLargerThanTheLogicalWidth) {
    for (const GemmKernel kernel : available_kernels()) {
        check_shape<TypeParam>(1, 1, 1, kernel, 3);
        check_shape<TypeParam>(13, 21, 9, kernel, 5);
        check_shape<TypeParam>(77, 35, 300, kernel, 1);
    }
}

TYPED_TEST(GemmTest, ZeroDepthProducesAZeroMatrix) {
    using T = TypeParam;
    for (const GemmKernel kernel : available_kernels()) {
        std::vector<T> c(6, T{7});
        linalg::internal::gemm(2, 3, 0, static_cast<const T*>(nullptr), 0,
                               static_cast<const T*>(nullptr), 3, c.data(), 3, kernel);
        for (const T value : c) {
            EXPECT_EQ(value, T{0}) << kernel_name(kernel);
        }
    }
}

TYPED_TEST(GemmTest, EmptyOutputIsANoOp) {
    using T = TypeParam;
    for (const GemmKernel kernel : available_kernels()) {
        const std::vector<T> a(4, T{1});
        linalg::internal::gemm(0, 0, 4, a.data(), 4, a.data(), 0, static_cast<T*>(nullptr), 0,
                               kernel);
        linalg::internal::gemm(0, 4, 1, static_cast<const T*>(nullptr), 1, a.data(), 4,
                               static_cast<T*>(nullptr), 4, kernel);
    }
}

TYPED_TEST(GemmTest, IsExactForSmallIntegerValues) {
    using T = TypeParam;
    // Every product and partial sum is an exactly representable integer.
    const linalg::DynamicMatrix<T> left{{T{1}, T{2}, T{3}}, {T{4}, T{5}, T{6}}};
    const linalg::DynamicMatrix<T> right{{T{7}, T{8}}, {T{9}, T{10}}, {T{11}, T{12}}};
    for (const GemmKernel kernel : available_kernels()) {
        linalg::DynamicMatrix<T> product(2, 2);
        linalg::internal::gemm(2, 2, 3, left.data(), 3, right.data(), 2, product.data(), 2,
                               kernel);
        EXPECT_EQ(product(0, 0), T{58});
        EXPECT_EQ(product(0, 1), T{64});
        EXPECT_EQ(product(1, 0), T{139});
        EXPECT_EQ(product(1, 1), T{154});
    }
}

TYPED_TEST(GemmTest, PublicMultiplyUsesACorrectKernelForLargeMatrices) {
    using T = TypeParam;
    std::mt19937 engine(42);
    std::uniform_real_distribution<T> distribution(T{-1}, T{1});
    linalg::DynamicMatrix<T> left(150, 300);
    linalg::DynamicMatrix<T> right(300, 70);
    for (size_type index = 0; index < left.size(); ++index) {
        left.data()[index] = distribution(engine);
    }
    for (size_type index = 0; index < right.size(); ++index) {
        right.data()[index] = distribution(engine);
    }

    const linalg::DynamicMatrix<T> product = linalg::multiply(left, right);

    Strided<T> a(left.rows(), left.cols(), 0, T{});
    Strided<T> b(right.rows(), right.cols(), 0, T{});
    Strided<T> c(product.rows(), product.cols(), 0, T{});
    a.values.assign(left.data(), left.data() + left.size());
    b.values.assign(right.data(), right.data() + right.size());
    c.values.assign(product.data(), product.data() + product.size());
    expect_matches_reference(a, b, c, "multiply");
}

TYPED_TEST(GemmTest, MatchesReferenceForEveryThreadCount) {
    struct Shape {
        size_type m;
        size_type n;
        size_type k;
    };
    const Shape shapes[] = {
        {7, 7, 7},       // fewer micro-tiles than threads
        {129, 127, 65},  // ragged row and column panels
        {300, 200, 513}, // several kc panels, so B is re-packed between barriers
        {6, 4100, 20},   // one row panel: the grid must split columns; crosses nc
        {1000, 8, 10},   // one column panel: rows only
        {77, 35, 300},   // with padded strides below
    };
    const size_type thread_counts[] = {1, 2, 3, 4, 5, 6, 8, 12};
    for (const GemmKernel kernel : available_kernels()) {
        for (const size_type threads : thread_counts) {
            for (const Shape& shape : shapes) {
                check_shape<TypeParam>(shape.m, shape.n, shape.k, kernel, 0, threads);
            }
            check_shape<TypeParam>(77, 35, 300, kernel, 3, threads);
        }
    }
}

TYPED_TEST(GemmTest, ConcurrentCallersShareThePoolSafely) {
    // Only one caller can own the pool; the others must fall back to serial execution.
    std::vector<std::thread> callers;
    for (size_type caller = 0; caller < 4; ++caller) {
        callers.emplace_back([caller] {
            for (size_type repeat = 0; repeat < 3; ++repeat) {
                check_shape<TypeParam>(150 + caller, 140, 260, GemmKernel::automatic, 0, 6);
            }
        });
    }
    for (std::thread& caller : callers) {
        caller.join();
    }
}

TYPED_TEST(GemmTest, ResultIsBitwiseIdenticalForEveryThreadCount) {
    // Threads split only rows and columns of C, never the k dimension, so every element is
    // accumulated by the same micro-kernel in the same pc order whatever the thread count.
    using T = TypeParam;
    const size_type row_counts[] = {6, 150, 333};
    const size_type parallel_thread_counts[] = {2, 3, 4, 5, 6, 12};
    std::mt19937 engine(7);
    for (const GemmKernel kernel : available_kernels()) {
        for (const size_type m : row_counts) {
            const size_type n = 4100; // crosses nc, so column groups are exercised as well
            const size_type k = 300;  // two pc panels
            const Strided<T> a = random_matrix<T>(m, k, 0, engine);
            const Strided<T> b = random_matrix<T>(k, n, 0, engine);
            std::vector<T> serial(m * n);
            linalg::internal::gemm(m, n, k, a.values.data(), k, b.values.data(), n, serial.data(),
                                   n, kernel, 1);
            for (const size_type threads : parallel_thread_counts) {
                std::vector<T> parallel(m * n);
                linalg::internal::gemm(m, n, k, a.values.data(), k, b.values.data(), n,
                                       parallel.data(), n, kernel, threads);
                ASSERT_TRUE(std::equal(serial.begin(), serial.end(), parallel.begin(),
                                       same_bits<T>))
                    << kernel_name(kernel) << " m=" << m << " threads=" << threads;
            }
        }
    }
}

TEST(GemmKernelTest, DefaultThreadCountIsPositive) {
    EXPECT_GE(linalg::internal::gemm_default_thread_count(), 1U);
}

TEST(GemmKernelTest, PortableAndAutomaticKernelsAreAlwaysAvailable) {
    EXPECT_TRUE(linalg::internal::gemm_kernel_available(GemmKernel::automatic));
    EXPECT_TRUE(linalg::internal::gemm_kernel_available(GemmKernel::portable));
}

TEST(GemmKernelTest, UnavailableKernelIsRejected) {
    if (linalg::internal::gemm_kernel_available(GemmKernel::avx2_fma)) {
        GTEST_SKIP() << "AVX2/FMA is available on this CPU";
    }
    const double a = 1.0;
    const double b = 2.0;
    double c = 0.0;
    EXPECT_THROW(linalg::internal::gemm(1, 1, 1, &a, 1, &b, 1, &c, 1, GemmKernel::avx2_fma),
                 std::invalid_argument);
}

} // namespace
