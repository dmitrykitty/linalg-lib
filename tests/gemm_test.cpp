#include <linalg/internal/gemm.hpp>
#include <linalg/operations.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
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
void check_shape(size_type m, size_type n, size_type k, GemmKernel kernel, size_type padding = 0) {
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
                           c.values.data(), c.stride, kernel);

    const std::string context = kernel_name(kernel) + " " + std::to_string(m) + "x" +
                                std::to_string(n) + "x" + std::to_string(k) +
                                " padding=" + std::to_string(padding);
    expect_matches_reference(a, b, c, context);
    for (size_type row = 0; row < m; ++row) {
        for (size_type col = n; col < c.stride; ++col) {
            ASSERT_EQ(c(row, col), sentinel) << context << " wrote padding at (" << row << ", "
                                             << col << ")";
        }
    }
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
    for (const GemmKernel kernel : available_kernels()) {
        // Depth around the kc block.
        for (const size_type k : std::initializer_list<size_type>{255, 256, 257, 513}) {
            check_shape<TypeParam>(13, 17, k, kernel);
        }
        // Rows around the mc block (72 for double, 144 for float).
        for (const size_type m : std::initializer_list<size_type>{71, 72, 73, 143, 144, 145, 150}) {
            check_shape<TypeParam>(m, 9, 5, kernel);
        }
        // Columns around the nc block.
        for (const size_type n : std::initializer_list<size_type>{4079, 4080, 4081, 8170}) {
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
