#include <linalg/operations.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

template <typename Left, typename Right>
concept MultiplicationAvailable = requires(const Left& left, const Right& right) {
    linalg::multiply(left, right);
};

using FixedLeft = linalg::StaticMatrix<float, 2, 3>;
using FixedRight = linalg::StaticMatrix<float, 3, 4>;
static_assert(std::is_same_v<decltype(linalg::transpose(std::declval<FixedLeft>())),
                             linalg::StaticMatrix<float, 3, 2>>);
static_assert(std::is_same_v<decltype(linalg::multiply(
                                 std::declval<FixedLeft>(), std::declval<FixedRight>())),
                             linalg::StaticMatrix<float, 2, 4>>);
static_assert(std::is_same_v<decltype(linalg::multiply(
                                 std::declval<FixedLeft>(),
                                 std::declval<linalg::StaticVector<float, 3>>())),
                             linalg::StaticVector<float, 2>>);
static_assert(!MultiplicationAvailable<FixedLeft, linalg::StaticMatrix<float, 4, 2>>);
static_assert(!MultiplicationAvailable<FixedLeft, linalg::StaticVector<float, 4>>);
static_assert(!MultiplicationAvailable<FixedLeft, linalg::StaticMatrix<double, 3, 4>>);
static_assert(!MultiplicationAvailable<linalg::DynamicMatrix<float>, linalg::DynamicMatrix<double>>);
static_assert(!MultiplicationAvailable<linalg::DynamicMatrix<float>, linalg::DynamicVector<double>>);
static_assert(!MultiplicationAvailable<FixedLeft, linalg::DynamicMatrix<float>>);

template <typename T>
constexpr bool static_operations_work_in_constant_expressions() {
    const linalg::StaticMatrix<T, 2, 3> left{{T{1}, T{2}, T{3}}, {T{4}, T{5}, T{6}}};
    const linalg::StaticMatrix<T, 3, 2> right{{T{7}, T{8}}, {T{9}, T{10}}, {T{11}, T{12}}};
    const linalg::StaticVector<T, 3> vector{T{7}, T{8}, T{9}};
    const auto transposed = linalg::transpose(left);
    const auto product = linalg::multiply(left, right);
    const auto applied = linalg::multiply(left, vector);
    return transposed.rows() == 3 && transposed.cols() == 2 && transposed(2, 1) == T{6} &&
           product(0, 0) == T{58} && product(1, 1) == T{154} &&
           applied[0] == T{50} && applied[1] == T{122};
}

static_assert(static_operations_work_in_constant_expressions<float>());
static_assert(static_operations_work_in_constant_expressions<double>());

template <typename T>
class SharedOperationsTest : public ::testing::Test {};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SharedOperationsTest, ScalarTypes);

TYPED_TEST(SharedOperationsTest, TransposePreservesValuesAndSelectsTheCorrectResultContainer) {
    using T = TypeParam;
    const linalg::DynamicMatrix<T> dynamic{{T{1}, T{2}, T{3}}, {T{4}, T{5}, T{6}}};
    const linalg::StaticMatrix<T, 2, 3> fixed{{T{1}, T{2}, T{3}}, {T{4}, T{5}, T{6}}};
    const auto dynamic_result = linalg::transpose(dynamic);
    const auto fixed_result = linalg::transpose(fixed);
    static_assert(std::is_same_v<typename decltype(dynamic_result)::value_type, T>);
    static_assert(std::is_same_v<std::remove_const_t<decltype(dynamic_result)>,
                                 linalg::DynamicMatrix<T>>);
    static_assert(std::is_same_v<std::remove_const_t<decltype(fixed_result)>,
                                 linalg::StaticMatrix<T, 3, 2>>);

    EXPECT_EQ(dynamic_result.rows(), 3U);
    EXPECT_EQ(dynamic_result.cols(), 2U);
    for (std::size_t row = 0; row < dynamic.rows(); ++row) {
        for (std::size_t col = 0; col < dynamic.cols(); ++col) {
            EXPECT_EQ(dynamic_result(col, row), dynamic(row, col));
            EXPECT_EQ(fixed_result(col, row), fixed(row, col));
        }
    }
    const auto round_trip = linalg::transpose(fixed_result);
    EXPECT_EQ(round_trip(1, 2), T{6});
    EXPECT_EQ(dynamic(0, 1), T{2});
}

TYPED_TEST(SharedOperationsTest, TransposeSupportsEmptyShapesWithoutLosingDimensions) {
    using T = TypeParam;
    const auto no_rows = linalg::transpose(linalg::DynamicMatrix<T>(0, 3));
    const auto no_cols = linalg::transpose(linalg::DynamicMatrix<T>(3, 0));
    const auto fixed_no_rows = linalg::transpose(linalg::StaticMatrix<T, 0, 3>{});
    const auto fixed_no_cols = linalg::transpose(linalg::StaticMatrix<T, 3, 0>{});
    EXPECT_EQ(no_rows.rows(), 3U);
    EXPECT_EQ(no_rows.cols(), 0U);
    EXPECT_EQ(no_cols.rows(), 0U);
    EXPECT_EQ(no_cols.cols(), 3U);
    EXPECT_EQ(fixed_no_rows.rows(), 3U);
    EXPECT_EQ(fixed_no_rows.cols(), 0U);
    EXPECT_EQ(fixed_no_cols.rows(), 0U);
    EXPECT_EQ(fixed_no_cols.cols(), 3U);
    EXPECT_TRUE(no_rows.empty());
    EXPECT_TRUE(no_cols.empty());
    EXPECT_TRUE(fixed_no_rows.empty());
    EXPECT_TRUE(fixed_no_cols.empty());
}

TYPED_TEST(SharedOperationsTest, MatrixMultiplySupportsDynamicAndStaticRectangularInputs) {
    using T = TypeParam;
    const linalg::DynamicMatrix<T> left{{T{1}, T{2}, T{3}}, {T{4}, T{5}, T{6}}};
    const linalg::DynamicMatrix<T> right{{T{7}, T{8}}, {T{9}, T{10}}, {T{11}, T{12}}};
    const linalg::StaticMatrix<T, 2, 3> fixed_left{{T{1}, T{2}, T{3}}, {T{4}, T{5}, T{6}}};
    const linalg::StaticMatrix<T, 3, 2> fixed_right{{T{7}, T{8}}, {T{9}, T{10}}, {T{11}, T{12}}};
    const auto dynamic = linalg::multiply(left, right);
    const auto fixed = linalg::multiply(fixed_left, fixed_right);
    const T expected[] = {T{58}, T{64}, T{139}, T{154}};
    static_assert(std::is_same_v<std::remove_const_t<decltype(dynamic)>, linalg::DynamicMatrix<T>>);
    static_assert(std::is_same_v<std::remove_const_t<decltype(fixed)>, linalg::StaticMatrix<T, 2, 2>>);
    ASSERT_EQ(dynamic.size(), 4U);
    for (std::size_t index = 0; index < dynamic.size(); ++index) {
        EXPECT_EQ(dynamic.data()[index], expected[index]);
        EXPECT_EQ(fixed.data()[index], expected[index]);
    }
    EXPECT_EQ(left(1, 2), T{6});
    EXPECT_EQ(right(2, 1), T{12});
    EXPECT_EQ(fixed_left(1, 2), T{6});
    EXPECT_EQ(fixed_right(2, 1), T{12});
}

TYPED_TEST(SharedOperationsTest, MatrixVectorMultiplySelectsDynamicOrFixedLengthResult) {
    using T = TypeParam;
    const linalg::DynamicMatrix<T> matrix{{T{1}, T{2}, T{3}}, {T{4}, T{5}, T{6}}};
    const linalg::DynamicVector<T> vector{T{7}, T{8}, T{9}};
    const linalg::StaticMatrix<T, 2, 3> fixed_matrix{{T{1}, T{2}, T{3}}, {T{4}, T{5}, T{6}}};
    const linalg::StaticVector<T, 3> fixed_vector{T{7}, T{8}, T{9}};
    const auto dynamic = linalg::multiply(matrix, vector);
    const auto fixed = linalg::multiply(fixed_matrix, fixed_vector);
    static_assert(std::is_same_v<std::remove_const_t<decltype(dynamic)>, linalg::DynamicVector<T>>);
    static_assert(std::is_same_v<std::remove_const_t<decltype(fixed)>, linalg::StaticVector<T, 2>>);
    EXPECT_EQ(dynamic[0], T{50});
    EXPECT_EQ(dynamic[1], T{122});
    EXPECT_EQ(fixed[0], T{50});
    EXPECT_EQ(fixed[1], T{122});
    EXPECT_EQ(vector[0], T{7});
    EXPECT_EQ(fixed_vector[0], T{7});
}

TYPED_TEST(SharedOperationsTest, MatrixMultiplySupportsZeroInnerAndOutputDimensions) {
    using T = TypeParam;
    const auto dynamic = linalg::multiply(linalg::DynamicMatrix<T>(2, 0),
                                         linalg::DynamicMatrix<T>(0, 3));
    const auto fixed = linalg::multiply(linalg::StaticMatrix<T, 2, 0>{},
                                       linalg::StaticMatrix<T, 0, 3>{});
    ASSERT_EQ(dynamic.size(), 6U);
    ASSERT_EQ(fixed.size(), 6U);
    for (std::size_t index = 0; index < dynamic.size(); ++index) {
        EXPECT_EQ(dynamic.data()[index], T{});
        EXPECT_EQ(fixed.data()[index], T{});
    }
    const auto no_rows = linalg::multiply(linalg::StaticMatrix<T, 0, 3>{},
                                         linalg::StaticMatrix<T, 3, 4>{});
    const auto no_cols = linalg::multiply(linalg::StaticMatrix<T, 3, 2>{},
                                         linalg::StaticMatrix<T, 2, 0>{});
    EXPECT_EQ(no_rows.rows(), 0U);
    EXPECT_EQ(no_rows.cols(), 4U);
    EXPECT_EQ(no_cols.rows(), 3U);
    EXPECT_EQ(no_cols.cols(), 0U);
    EXPECT_TRUE(no_rows.empty());
    EXPECT_TRUE(no_cols.empty());
}

TYPED_TEST(SharedOperationsTest, MatrixVectorMultiplySupportsEmptyRowsAndColumns) {
    using T = TypeParam;
    const auto no_rows = linalg::multiply(linalg::StaticMatrix<T, 0, 3>{},
                                         linalg::StaticVector<T, 3>{});
    const auto no_cols = linalg::multiply(linalg::StaticMatrix<T, 3, 0>{},
                                         linalg::StaticVector<T, 0>{});
    const auto dynamic = linalg::multiply(linalg::DynamicMatrix<T>(3, 0),
                                         linalg::DynamicVector<T>{});
    EXPECT_TRUE(no_rows.empty());
    ASSERT_EQ(no_cols.size(), 3U);
    for (std::size_t index = 0; index < no_cols.size(); ++index) {
        EXPECT_EQ(no_cols[index], T{});
        EXPECT_EQ(dynamic[index], T{});
    }
}

TYPED_TEST(SharedOperationsTest, DynamicMultiplyRejectsIncompatibleDimensions) {
    using T = TypeParam;
    const linalg::DynamicMatrix<T> left(2, 3, T{1});
    const linalg::DynamicMatrix<T> wrong_right(2, 4, T{2});
    const linalg::DynamicVector<T> wrong_vector(2, T{3});
    EXPECT_THROW((void)linalg::multiply(left, wrong_right), std::invalid_argument);
    EXPECT_THROW((void)linalg::multiply(left, wrong_vector), std::invalid_argument);
    EXPECT_EQ(left(0, 0), T{1});
}

TYPED_TEST(SharedOperationsTest, KernelsOverwritePreviouslyPopulatedOutputOnEveryCall) {
    using T = TypeParam;
    const linalg::StaticMatrix<T, 2, 2> matrix{{T{1}, T{2}}, {T{3}, T{4}}};
    const linalg::StaticVector<T, 2> vector{T{5}, T{6}};
    linalg::StaticMatrix<T, 2, 2> product(T{42});
    linalg::StaticMatrix<T, 2, 2> transposed(T{42});
    linalg::StaticVector<T, 2> applied(T{42});
    for (int iteration = 0; iteration < 2; ++iteration) {
        linalg::internal::matmul_into(matrix, matrix, product);
        linalg::internal::transpose_into(matrix, transposed);
        linalg::internal::matvec_into(matrix, vector, applied);
        EXPECT_EQ(product(0, 0), T{7});
        EXPECT_EQ(product(1, 1), T{22});
        EXPECT_EQ(transposed(0, 1), T{3});
        EXPECT_EQ(applied[0], T{17});
        EXPECT_EQ(applied[1], T{39});
    }
    const linalg::StaticMatrix<T, 2, 0> empty_left;
    const linalg::StaticMatrix<T, 0, 2> empty_right;
    linalg::internal::matmul_into(empty_left, empty_right, product);
    linalg::internal::matvec_into(empty_left, linalg::StaticVector<T, 0>{}, applied);
    EXPECT_EQ(product(0, 0), T{});
    EXPECT_EQ(product(1, 1), T{});
    EXPECT_EQ(applied[0], T{});
    EXPECT_EQ(applied[1], T{});
}

TYPED_TEST(SharedOperationsTest, SharedKernelsAreIndependentOfInputAndOutputStorage) {
    using T = TypeParam;
    const linalg::DynamicMatrix<T> dynamic{{T{1}, T{2}}, {T{3}, T{4}}};
    const linalg::StaticMatrix<T, 2, 2> fixed{{T{1}, T{2}}, {T{3}, T{4}}};
    const linalg::DynamicVector<T> vector{T{5}, T{6}};
    linalg::StaticMatrix<T, 2, 2> fixed_output;
    linalg::DynamicMatrix<T> dynamic_output(2, 2);
    linalg::StaticVector<T, 2> vector_output;
    linalg::internal::matmul_into(dynamic, fixed, fixed_output);
    linalg::internal::transpose_into(fixed, dynamic_output);
    linalg::internal::matvec_into(dynamic, vector, vector_output);
    EXPECT_EQ(fixed_output(0, 0), T{7});
    EXPECT_EQ(dynamic_output(0, 1), T{3});
    EXPECT_EQ(vector_output[1], T{39});
}

TYPED_TEST(SharedOperationsTest, MatrixMultiplyMatchesAnIndependentHigherPrecisionReference) {
    using T = TypeParam;
    linalg::DynamicMatrix<T> left(7, 9);
    linalg::DynamicMatrix<T> right(9, 5);
    for (std::size_t index = 0; index < left.size(); ++index) {
        left.data()[index] = static_cast<T>(static_cast<int>(index % 23) - 11) / T{10};
    }
    for (std::size_t index = 0; index < right.size(); ++index) {
        right.data()[index] = static_cast<T>(static_cast<int>(index % 17) - 8) / T{10};
    }
    const linalg::StaticMatrix<T, 7, 9> fixed_left{std::span<const T>{left.data(), left.size()}};
    const linalg::StaticMatrix<T, 9, 5> fixed_right{std::span<const T>{right.data(), right.size()}};
    const auto dynamic = linalg::multiply(left, right);
    const auto fixed = linalg::multiply(fixed_left, fixed_right);
    for (std::size_t row = 0; row < left.rows(); ++row) {
        for (std::size_t col = 0; col < right.cols(); ++col) {
            long double reference = 0.0L;
            for (std::size_t inner = 0; inner < left.cols(); ++inner) {
                reference += static_cast<long double>(left(row, inner)) * right(inner, col);
            }
            const double expected = static_cast<double>(reference);
            const double tolerance = 64.0 * std::numeric_limits<T>::epsilon() *
                                     std::max(1.0, std::abs(expected));
            EXPECT_NEAR(dynamic(row, col), expected, tolerance);
            EXPECT_NEAR(fixed(row, col), expected, tolerance);
        }
    }
}

TYPED_TEST(SharedOperationsTest, ZeroFactorsDoNotSuppressNonFiniteArithmetic) {
    using T = TypeParam;
    const T infinity = std::numeric_limits<T>::infinity();
    const linalg::StaticMatrix<T, 1, 1> zero{{T{0}}};
    const linalg::StaticMatrix<T, 1, 1> non_finite{{infinity}};
    const linalg::StaticVector<T, 1> vector{infinity};
    EXPECT_TRUE(std::isnan(linalg::multiply(zero, non_finite)(0, 0)));
    EXPECT_TRUE(std::isnan(linalg::multiply(zero, vector)[0]));
    EXPECT_TRUE(std::isinf(linalg::transpose(non_finite)(0, 0)));
}

TYPED_TEST(SharedOperationsTest, TransposeHandlesOddWideAndTallShapes) {
    using T = TypeParam;
    for (const auto& [rows, cols] : {std::pair{3U, 5U}, std::pair{1U, 7U}, std::pair{7U, 1U}}) {
        linalg::DynamicMatrix<T> input(rows, cols);
        for (std::size_t index = 0; index < input.size(); ++index) {
            input.data()[index] = static_cast<T>(index);
        }
        const auto output = linalg::transpose(input);
        for (std::size_t row = 0; row < input.rows(); ++row) {
            for (std::size_t col = 0; col < input.cols(); ++col) {
                EXPECT_EQ(output(col, row), input(row, col));
            }
        }
    }
}

} // namespace
