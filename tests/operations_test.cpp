#include <linalg/operations.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

TEST(MatrixArithmeticTest, CompoundMatrixOperationsUpdateInPlace) {
    linalg::Matrix matrix{{1.0, 2.0}, {3.0, 4.0}};
    const linalg::Matrix other{{10.0, 20.0}, {30.0, 40.0}};

    EXPECT_EQ(&(matrix += other), &matrix);
    EXPECT_DOUBLE_EQ(matrix(0, 0), 11.0);
    EXPECT_DOUBLE_EQ(matrix(1, 1), 44.0);

    EXPECT_EQ(&(matrix -= other), &matrix);
    EXPECT_DOUBLE_EQ(matrix(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix(1, 1), 4.0);

    EXPECT_EQ(&(matrix *= 2.0), &matrix);
    EXPECT_DOUBLE_EQ(matrix(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(matrix(1, 0), 6.0);
}

TEST(MatrixArithmeticTest, CompoundMatrixOperationsRejectDifferentShapesBeforeMutation) {
    linalg::Matrix matrix{{1.0, 2.0}};
    const linalg::Matrix wrong_shape{{1.0}, {2.0}};

    EXPECT_THROW(matrix += wrong_shape, std::invalid_argument);
    EXPECT_DOUBLE_EQ(matrix(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix(0, 1), 2.0);

    EXPECT_THROW(matrix -= wrong_shape, std::invalid_argument);
    EXPECT_DOUBLE_EQ(matrix(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix(0, 1), 2.0);
}

TEST(MatrixArithmeticTest, ScalarCompoundOperationsUpdateEveryElement) {
    static_assert(noexcept(std::declval<linalg::Matrix&>() += 1.0));
    static_assert(noexcept(std::declval<linalg::Matrix&>() -= 1.0));

    linalg::Matrix matrix{{1.0, 2.0}, {3.0, 4.0}};

    EXPECT_EQ(&(matrix += 5.0), &matrix);
    EXPECT_EQ(&(matrix -= 2.0), &matrix);

    EXPECT_DOUBLE_EQ(matrix(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(matrix(0, 1), 5.0);
    EXPECT_DOUBLE_EQ(matrix(1, 0), 6.0);
    EXPECT_DOUBLE_EQ(matrix(1, 1), 7.0);
}

TEST(MatrixBroadcastTest, AddsVectorToEveryRowWithoutChangingInputs) {
    const linalg::Matrix matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const linalg::Vector values{10.0, 20.0, 30.0};

    const linalg::Matrix result = linalg::add_to_each_row(matrix, values);

    EXPECT_DOUBLE_EQ(result(0, 0), 11.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 22.0);
    EXPECT_DOUBLE_EQ(result(0, 2), 33.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 14.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 25.0);
    EXPECT_DOUBLE_EQ(result(1, 2), 36.0);
    EXPECT_DOUBLE_EQ(matrix(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(values[0], 10.0);
}

TEST(MatrixBroadcastTest, AddsVectorToEveryColumnWithoutChangingInputs) {
    const linalg::Matrix matrix{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    const linalg::Vector values{10.0, 20.0, 30.0};

    const linalg::Matrix result = linalg::add_to_each_column(matrix, values);

    EXPECT_DOUBLE_EQ(result(0, 0), 11.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 12.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 23.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 24.0);
    EXPECT_DOUBLE_EQ(result(2, 0), 35.0);
    EXPECT_DOUBLE_EQ(result(2, 1), 36.0);
    EXPECT_DOUBLE_EQ(matrix(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(values[0], 10.0);
}

TEST(MatrixBroadcastTest, RejectsVectorsWithIncompatibleLengths) {
    const linalg::Matrix matrix(2, 3);
    const linalg::Vector wrong_for_rows(2);
    const linalg::Vector wrong_for_columns(3);

    EXPECT_THROW(linalg::add_to_each_row(matrix, wrong_for_rows), std::invalid_argument);
    EXPECT_THROW(
        linalg::add_to_each_column(matrix, wrong_for_columns),
        std::invalid_argument);
}

TEST(MatrixRepeatTest, RepeatsVectorAsRowsUsingIndependentStorage) {
    linalg::Vector values{1.0, 2.0, 3.0};

    const linalg::Matrix result = linalg::repeat_as_rows(values, 3);
    values[0] = 99.0;

    ASSERT_EQ(result.rows(), 3);
    ASSERT_EQ(result.cols(), 3);
    for (linalg::Matrix::size_type row = 0; row < result.rows(); ++row) {
        EXPECT_DOUBLE_EQ(result(row, 0), 1.0);
        EXPECT_DOUBLE_EQ(result(row, 1), 2.0);
        EXPECT_DOUBLE_EQ(result(row, 2), 3.0);
    }
}

TEST(MatrixRepeatTest, RepeatsVectorAsColumnsUsingIndependentStorage) {
    linalg::Vector values{1.0, 2.0, 3.0};

    const linalg::Matrix result = linalg::repeat_as_columns(values, 3);
    values[0] = 99.0;

    ASSERT_EQ(result.rows(), 3);
    ASSERT_EQ(result.cols(), 3);
    for (linalg::Matrix::size_type col = 0; col < result.cols(); ++col) {
        EXPECT_DOUBLE_EQ(result(0, col), 1.0);
        EXPECT_DOUBLE_EQ(result(1, col), 2.0);
        EXPECT_DOUBLE_EQ(result(2, col), 3.0);
    }
}

TEST(MatrixRepeatTest, SupportsEmptyVectorsAndZeroRepeatCounts) {
    const linalg::Vector empty;
    const linalg::Vector values{1.0, 2.0};

    const linalg::Matrix empty_rows = linalg::repeat_as_rows(empty, 3);
    const linalg::Matrix empty_columns = linalg::repeat_as_columns(empty, 4);
    const linalg::Matrix zero_rows = linalg::repeat_as_rows(values, 0);
    const linalg::Matrix zero_columns = linalg::repeat_as_columns(values, 0);

    EXPECT_EQ(empty_rows.rows(), 3);
    EXPECT_EQ(empty_rows.cols(), 0);
    EXPECT_TRUE(empty_rows.empty());

    EXPECT_EQ(empty_columns.rows(), 0);
    EXPECT_EQ(empty_columns.cols(), 4);
    EXPECT_TRUE(empty_columns.empty());

    EXPECT_EQ(zero_rows.rows(), 0);
    EXPECT_EQ(zero_rows.cols(), 2);
    EXPECT_TRUE(zero_rows.empty());

    EXPECT_EQ(zero_columns.rows(), 2);
    EXPECT_EQ(zero_columns.cols(), 0);
    EXPECT_TRUE(zero_columns.empty());
}

TEST(MatrixTransposeTest, TransposesRectangularMatrixWithoutChangingInput) {
    const linalg::Matrix matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    const linalg::Matrix result = linalg::transpose(matrix);

    ASSERT_EQ(result.rows(), 3);
    ASSERT_EQ(result.cols(), 2);
    EXPECT_DOUBLE_EQ(result(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(result(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(result(2, 1), 6.0);
    EXPECT_DOUBLE_EQ(matrix(0, 1), 2.0);
}

TEST(MatrixTransposeTest, ApplyingTransposeTwiceRestoresOriginalMatrix) {
    const linalg::Matrix matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    const linalg::Matrix result = linalg::transpose(linalg::transpose(matrix));

    ASSERT_EQ(result.rows(), matrix.rows());
    ASSERT_EQ(result.cols(), matrix.cols());
    for (linalg::Matrix::size_type index = 0; index < matrix.size(); ++index) {
        EXPECT_DOUBLE_EQ(result.data()[index], matrix.data()[index]);
    }
}

TEST(MatrixTransposeTest, SwapsZeroDimensions) {
    const linalg::Matrix zero_rows(0, 4);
    const linalg::Matrix zero_columns(4, 0);

    const linalg::Matrix transposed_zero_rows = linalg::transpose(zero_rows);
    const linalg::Matrix transposed_zero_columns = linalg::transpose(zero_columns);

    EXPECT_EQ(transposed_zero_rows.rows(), 4);
    EXPECT_EQ(transposed_zero_rows.cols(), 0);
    EXPECT_TRUE(transposed_zero_rows.empty());
    EXPECT_EQ(transposed_zero_columns.rows(), 0);
    EXPECT_EQ(transposed_zero_columns.cols(), 4);
    EXPECT_TRUE(transposed_zero_columns.empty());
}

TEST(MatrixTraceTest, SumsOnlyDiagonalElementsWithoutChangingInput) {
    const linalg::Matrix matrix{
        {1.5, 100.0, 200.0},
        {300.0, -2.0, 400.0},
        {500.0, 600.0, 3.25},
    };

    EXPECT_DOUBLE_EQ(linalg::trace(matrix), 2.75);
    EXPECT_DOUBLE_EQ(matrix(0, 1), 100.0);
}

TEST(MatrixTraceTest, SupportsOneByOneMatrix) {
    const linalg::Matrix matrix{{-7.5}};

    EXPECT_DOUBLE_EQ(linalg::trace(matrix), -7.5);
}

TEST(MatrixTraceTest, RejectsEmptyAndNonSquareMatrices) {
    const linalg::Matrix empty;
    const linalg::Matrix rectangular(2, 3);
    const linalg::Matrix zero_rows(0, 3);
    const linalg::Matrix zero_columns(3, 0);

    EXPECT_THROW(linalg::trace(empty), std::invalid_argument);
    EXPECT_THROW(linalg::trace(rectangular), std::invalid_argument);
    EXPECT_THROW(linalg::trace(zero_rows), std::invalid_argument);
    EXPECT_THROW(linalg::trace(zero_columns), std::invalid_argument);
}

TEST(MatrixTraceTest, PropagatesDiagonalNanAndIgnoresOffDiagonalNan) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const linalg::Matrix diagonal_nan{{nan, 1.0}, {2.0, 3.0}};
    const linalg::Matrix off_diagonal_nan{{1.0, nan}, {nan, 3.0}};

    EXPECT_TRUE(std::isnan(linalg::trace(diagonal_nan)));
    EXPECT_DOUBLE_EQ(linalg::trace(off_diagonal_nan), 4.0);
}

TEST(VectorNormTest, ComputesEachNamedAndSelectedNorm) {
    const linalg::Vector vector{-3.0, 4.0};

    EXPECT_DOUBLE_EQ(linalg::norm(vector, linalg::VectorNorm::l1), 7.0);
    EXPECT_DOUBLE_EQ(linalg::norm(vector, linalg::VectorNorm::l2), 5.0);
    EXPECT_DOUBLE_EQ(linalg::norm(vector, linalg::VectorNorm::infinity), 4.0);
}

TEST(MatrixNormTest, ComputesEachNamedAndSelectedNorm) {
    const linalg::Matrix matrix{{1.0, -2.0, 3.0}, {-4.0, 5.0, -6.0}};

    EXPECT_DOUBLE_EQ(linalg::norm(matrix, linalg::MatrixNorm::one), 9.0);
    EXPECT_NEAR(
        linalg::norm(matrix, linalg::MatrixNorm::frobenius),
        std::sqrt(91.0),
        1e-14);
    EXPECT_DOUBLE_EQ(linalg::norm(matrix, linalg::MatrixNorm::infinity), 15.0);
}

TEST(NormTest, EmptyInputsHaveZeroNorm) {
    const linalg::Vector vector;
    const linalg::Matrix matrix;

    EXPECT_DOUBLE_EQ(linalg::norm(vector, linalg::VectorNorm::l1), 0.0);
    EXPECT_DOUBLE_EQ(linalg::norm(vector, linalg::VectorNorm::l2), 0.0);
    EXPECT_DOUBLE_EQ(linalg::norm(vector, linalg::VectorNorm::infinity), 0.0);
    EXPECT_DOUBLE_EQ(linalg::norm(matrix, linalg::MatrixNorm::one), 0.0);
    EXPECT_DOUBLE_EQ(linalg::norm(matrix, linalg::MatrixNorm::frobenius), 0.0);
    EXPECT_DOUBLE_EQ(linalg::norm(matrix, linalg::MatrixNorm::infinity), 0.0);
}

TEST(NormTest, ScaledEuclideanCalculationAvoidsIntermediateOverflow) {
    const double large = std::numeric_limits<double>::max() / 2.0;
    const linalg::Vector vector{large, large};
    const linalg::Matrix matrix{{large, large}};

    EXPECT_TRUE(std::isfinite(linalg::norm(vector, linalg::VectorNorm::l2)));
    EXPECT_TRUE(std::isfinite(linalg::norm(matrix, linalg::MatrixNorm::frobenius)));
    EXPECT_NEAR(linalg::norm(vector, linalg::VectorNorm::l2) / large, std::sqrt(2.0), 1e-15);
    EXPECT_NEAR(linalg::norm(matrix, linalg::MatrixNorm::frobenius) / large, std::sqrt(2.0), 1e-15);
}

TEST(NormTest, RejectsUnknownEnumValues) {
    const linalg::Vector vector{1.0};
    const linalg::Matrix matrix{{1.0}};

    EXPECT_THROW(
        linalg::norm(vector, static_cast<linalg::VectorNorm>(999)),
        std::invalid_argument);
    EXPECT_THROW(
        linalg::norm(matrix, static_cast<linalg::MatrixNorm>(999)),
        std::invalid_argument);
}

TEST(NormTest, PropagatesNanAndInfinity) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    const linalg::Vector nan_vector{1.0, nan};
    const linalg::Vector infinite_vector{1.0, infinity, infinity};
    const linalg::Matrix nan_matrix{{1.0, nan}};
    const linalg::Matrix infinite_matrix{{infinity, infinity}};

    EXPECT_TRUE(std::isnan(linalg::norm(nan_vector, linalg::VectorNorm::l2)));
    EXPECT_TRUE(std::isnan(linalg::norm(nan_vector, linalg::VectorNorm::infinity)));
    EXPECT_TRUE(std::isinf(linalg::norm(infinite_vector, linalg::VectorNorm::l2)));
    EXPECT_TRUE(std::isnan(linalg::norm(nan_matrix, linalg::MatrixNorm::one)));
    EXPECT_TRUE(std::isnan(linalg::norm(nan_matrix, linalg::MatrixNorm::infinity)));
    EXPECT_TRUE(std::isinf(linalg::norm(infinite_matrix, linalg::MatrixNorm::frobenius)));
}
