#include <linalg/operations.hpp>

#include <gtest/gtest.h>

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
