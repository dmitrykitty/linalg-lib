#include <linalg/linalg.hpp>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

template <typename T>
constexpr bool construction_and_access_work_in_constant_expressions() {
    using Matrix = linalg::StaticMatrix<T, 2, 3>;
    const Matrix zero;
    const Matrix filled(T{7});
    const Matrix nested{{T{1}, T{2}}, {T{3}}};
    const std::array<T, 2> values{T{4}, T{5}};
    const Matrix from_span{std::span<const T>{values}};
    Matrix flat = Matrix::from_values({T{1}, T{2}, T{3}, T{4}});
    flat.at(1, 2) = T{6};
    flat(0, 0) = T{9};
    flat.data()[1] = T{8};
    const Matrix copy = flat;

    return zero.rows() == 2 && zero.cols() == 3 && zero.size() == 6 && !zero.empty() &&
           zero(1, 2) == T{} && filled.at(1, 2) == T{7} && nested(1, 0) == T{3} &&
           nested(1, 2) == T{} && from_span(0, 1) == T{5} && from_span(1, 2) == T{} &&
           copy.at(1, 2) == T{6} && copy(0, 0) == T{9} && copy.data()[1] == T{8};
}

static_assert(construction_and_access_work_in_constant_expressions<float>());
static_assert(construction_and_access_work_in_constant_expressions<double>());

using FixedMatrix = linalg::StaticMatrix<double, 2, 3>;
static_assert(std::is_trivially_copyable_v<FixedMatrix>);
static_assert(std::is_nothrow_move_constructible_v<FixedMatrix>);
static_assert(std::is_nothrow_move_assignable_v<FixedMatrix>);
static_assert(!std::is_constructible_v<FixedMatrix, std::size_t, std::size_t>);
static_assert(!std::is_convertible_v<std::span<const double>, FixedMatrix>);
static_assert(!std::is_assignable_v<FixedMatrix&, linalg::StaticMatrix<double, 3, 2>>);
static_assert(!std::is_assignable_v<FixedMatrix&, linalg::StaticMatrix<float, 2, 3>>);

// A zero extent permits an empty matrix even when the other extent is very large.
constexpr auto maximum_size = std::numeric_limits<std::size_t>::max();
static_assert(linalg::StaticMatrix<double, 0, maximum_size>{}.size() == 0);
static_assert(linalg::StaticMatrix<double, maximum_size, 0>{}.size() == 0);

template <typename T>
class StaticMatrixTest : public ::testing::Test {};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(StaticMatrixTest, ScalarTypes);

TYPED_TEST(StaticMatrixTest, DefaultAndFillConstructionPreserveFixedDimensions) {
    using T = TypeParam;
    using Matrix = linalg::StaticMatrix<T, 2, 3>;
    const Matrix zero;
    const Matrix filled(T{4});
    const Matrix one_value = Matrix::from_values({T{4}});

    EXPECT_EQ(zero.rows(), 2U);
    EXPECT_EQ(zero.cols(), 3U);
    EXPECT_EQ(zero.size(), 6U);
    EXPECT_FALSE(zero.empty());
    for (std::size_t index = 0; index < zero.size(); ++index) {
        EXPECT_EQ(zero.data()[index], T{});
        EXPECT_EQ(filled.data()[index], T{4});
    }
    EXPECT_EQ(one_value(0, 0), T{4});
    EXPECT_EQ(one_value(0, 1), T{});
}

TYPED_TEST(StaticMatrixTest, FlatListsUseRowMajorLayoutAndZeroPadding) {
    using T = TypeParam;
    using Matrix = linalg::StaticMatrix<T, 2, 3>;
    const Matrix full = Matrix::from_values({T{1}, T{2}, T{3}, T{4}, T{5}, T{6}});
    const Matrix partial = Matrix::from_values({T{1}, T{2}, T{3}, T{4}});
    const Matrix empty_input = Matrix::from_values({});

    const std::array<T, 6> expected{T{1}, T{2}, T{3}, T{4}, T{5}, T{6}};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        EXPECT_EQ(full.data()[index], expected[index]);
        EXPECT_EQ(empty_input.data()[index], T{});
    }
    EXPECT_EQ(&full(1, 0), full.data() + 3);
    EXPECT_EQ(&full(1, 2), full.data() + 5);
    EXPECT_EQ(partial(1, 0), T{4});
    EXPECT_EQ(partial(1, 1), T{});
    EXPECT_EQ(partial(1, 2), T{});
}

TYPED_TEST(StaticMatrixTest, SpanConstructionCopiesValuesAndPadsMissingElements) {
    using T = TypeParam;
    using Matrix = linalg::StaticMatrix<T, 2, 3>;
    std::array<T, 4> values{T{1}, T{2}, T{3}, T{4}};
    const Matrix matrix{std::span<const T>{values}};
    const Matrix empty_input{std::span<const T>{}};
    values[0] = T{9};

    EXPECT_EQ(matrix(0, 0), T{1});
    EXPECT_EQ(matrix(1, 0), T{4});
    EXPECT_EQ(matrix(1, 2), T{});
    EXPECT_NE(matrix.data(), values.data());
    for (std::size_t index = 0; index < empty_input.size(); ++index) {
        EXPECT_EQ(empty_input.data()[index], T{});
    }
}

TYPED_TEST(StaticMatrixTest, NestedListsPadShortRowsEmptyRowsAndMissingRows) {
    using T = TypeParam;
    using Matrix = linalg::StaticMatrix<T, 3, 3>;
    const Matrix ragged{{T{1}, T{2}, T{3}}, {T{4}}};
    const Matrix empty_rows{{}, {T{7}, T{8}}, {}};
    const Matrix singletons{{T{1}}, {T{2}}};

    EXPECT_EQ(ragged.rows(), 3U);
    EXPECT_EQ(ragged.cols(), 3U);
    EXPECT_EQ(ragged(1, 0), T{4});
    EXPECT_EQ(ragged(1, 1), T{});
    EXPECT_EQ(ragged(2, 0), T{});
    EXPECT_EQ(ragged(2, 2), T{});
    EXPECT_EQ(empty_rows(0, 0), T{});
    EXPECT_EQ(empty_rows(1, 0), T{7});
    EXPECT_EQ(empty_rows(1, 1), T{8});
    EXPECT_EQ(empty_rows(1, 2), T{});
    EXPECT_EQ(empty_rows(2, 2), T{});
    EXPECT_EQ(singletons(0, 0), T{1});
    EXPECT_EQ(singletons(1, 0), T{2});
    EXPECT_EQ(singletons(0, 1), T{});
}

TYPED_TEST(StaticMatrixTest, ConstructorsRejectInputsThatExceedTheFixedShape) {
    using T = TypeParam;
    using Matrix = linalg::StaticMatrix<T, 2, 2>;
    const std::array<T, 5> values{};

    EXPECT_THROW((void)Matrix::from_values({T{1}, T{2}, T{3}, T{4}, T{5}}),
                 std::invalid_argument);
    EXPECT_THROW((void)(Matrix{std::span<const T>{values}}), std::invalid_argument);
    EXPECT_THROW((void)(Matrix{{T{1}, T{2}, T{3}}}), std::invalid_argument);
    EXPECT_THROW((void)(Matrix{{T{1}}, {T{2}}, {T{3}}}), std::invalid_argument);
    EXPECT_THROW((void)(Matrix{{}, {}, {}}), std::invalid_argument);
}

TYPED_TEST(StaticMatrixTest, MutableAndConstAccessHaveTheSameCoreApiAsDynamicMatrix) {
    using T = TypeParam;
    using Matrix = linalg::StaticMatrix<T, 2, 3>;
    static_assert(std::is_same_v<typename Matrix::value_type, T>);
    static_assert(std::is_same_v<typename Matrix::size_type, std::size_t>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix&>().data()), T*>);
    static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().data()), const T*>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix&>()(0, 0)), T&>);
    static_assert(std::is_same_v<decltype(std::declval<const Matrix&>()(0, 0)), const T&>);
    static_assert(std::is_same_v<decltype(std::declval<Matrix&>().at(0, 0)), T&>);
    static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().at(0, 0)), const T&>);

    Matrix matrix;
    matrix(0, 1) = T{2};
    matrix.at(1, 2) = T{6};
    matrix.data()[3] = T{4};
    const Matrix& read_only = matrix;
    EXPECT_EQ(read_only(0, 1), T{2});
    EXPECT_EQ(read_only.at(1, 2), T{6});
    EXPECT_EQ(read_only.data()[3], T{4});
}

TYPED_TEST(StaticMatrixTest, CheckedAccessRejectsInvalidIndicesBeforeCalculatingAnOffset) {
    using T = TypeParam;
    linalg::StaticMatrix<T, 2, 3> matrix;
    const auto& read_only = matrix;

    EXPECT_THROW(matrix.at(2, 0), std::out_of_range);
    EXPECT_THROW(matrix.at(0, 3), std::out_of_range);
    EXPECT_THROW(matrix.at(maximum_size, maximum_size), std::out_of_range);
    EXPECT_THROW(read_only.at(2, 0), std::out_of_range);
    EXPECT_THROW(read_only.at(0, 3), std::out_of_range);
    EXPECT_THROW(read_only.at(maximum_size, maximum_size), std::out_of_range);
}

TYPED_TEST(StaticMatrixTest, ZeroDimensionsAreEmptyAndHaveNoValidElementIndices) {
    using T = TypeParam;
    using NoRows = linalg::StaticMatrix<T, 0, 3>;
    using NoCols = linalg::StaticMatrix<T, 3, 0>;
    using ZeroByZero = linalg::StaticMatrix<T, 0, 0>;
    const NoRows no_rows{std::span<const T>{}};
    const NoCols no_cols{{}, {}, {}};
    const ZeroByZero zero_by_zero;

    EXPECT_EQ(no_rows.rows(), 0U);
    EXPECT_EQ(no_rows.cols(), 3U);
    EXPECT_EQ(no_cols.rows(), 3U);
    EXPECT_EQ(no_cols.cols(), 0U);
    EXPECT_EQ(no_rows.size(), 0U);
    EXPECT_EQ(no_cols.size(), 0U);
    EXPECT_TRUE(no_rows.empty());
    EXPECT_TRUE(no_cols.empty());
    EXPECT_TRUE(zero_by_zero.empty());
    EXPECT_THROW(no_rows.at(0, 0), std::out_of_range);
    EXPECT_THROW(no_cols.at(0, 0), std::out_of_range);
    EXPECT_THROW(zero_by_zero.at(0, 0), std::out_of_range);
    EXPECT_THROW((void)NoRows::from_values({T{1}}), std::invalid_argument);
    EXPECT_THROW((void)(NoRows{{}}), std::invalid_argument);
    EXPECT_THROW((void)(NoCols{{T{1}}}), std::invalid_argument);
    EXPECT_THROW((void)(NoCols{{}, {}, {}, {}}), std::invalid_argument);
}

TYPED_TEST(StaticMatrixTest, CopiesAndMovesOwnIndependentStorageAndKeepTheirShape) {
    using T = TypeParam;
    using Matrix = linalg::StaticMatrix<T, 2, 2>;
    const Matrix original{{T{1}, T{2}}, {T{3}, T{4}}};
    Matrix copy = original;
    copy(0, 0) = T{9};
    EXPECT_EQ(original(0, 0), T{1});
    EXPECT_NE(copy.data(), original.data());

    Matrix assigned;
    assigned = original;
    assigned(1, 1) = T{8};
    EXPECT_EQ(original(1, 1), T{4});

    Matrix moved = std::move(copy);
    EXPECT_EQ(moved(0, 0), T{9});
    EXPECT_NE(moved.data(), copy.data());
    EXPECT_EQ(copy.rows(), 2U);
    EXPECT_EQ(copy.cols(), 2U);
    EXPECT_FALSE(copy.empty());
    EXPECT_EQ(copy(0, 0), T{9});
    moved(0, 0) = T{7};
    EXPECT_EQ(copy(0, 0), T{9});

    Matrix move_assigned;
    move_assigned = std::move(moved);
    EXPECT_EQ(move_assigned(0, 0), T{7});
    EXPECT_NE(move_assigned.data(), moved.data());
    EXPECT_EQ(moved.rows(), 2U);
    EXPECT_EQ(moved.cols(), 2U);
    EXPECT_EQ(moved(0, 0), T{7});
}

TYPED_TEST(StaticMatrixTest, NestedConstructionMatchesDynamicMatrixWhenShapesMatch) {
    using T = TypeParam;
    const linalg::StaticMatrix<T, 2, 3> fixed{{T{1}, T{2}, T{3}}, {T{4}}};
    const linalg::DynamicMatrix<T> dynamic{{T{1}, T{2}, T{3}}, {T{4}}};

    ASSERT_EQ(fixed.rows(), dynamic.rows());
    ASSERT_EQ(fixed.cols(), dynamic.cols());
    ASSERT_EQ(fixed.size(), dynamic.size());
    for (std::size_t index = 0; index < fixed.size(); ++index) {
        EXPECT_EQ(fixed.data()[index], dynamic.data()[index]);
    }
}

} // namespace
