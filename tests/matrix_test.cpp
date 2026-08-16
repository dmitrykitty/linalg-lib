#include <linalg/matrix.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

TEST(MatrixTest, DefaultConstructionCreatesAnEmptyMatrix) {
    static_assert(std::is_default_constructible_v<linalg::Matrix>);

    const linalg::Matrix matrix;

    EXPECT_EQ(matrix.rows(), 0);
    EXPECT_EQ(matrix.cols(), 0);
    EXPECT_EQ(matrix.size(), 0);
    EXPECT_TRUE(matrix.empty());
}

TEST(MatrixTest, ConstructionInitializesDimensionsAndValues) {
    const linalg::Matrix zero_matrix(2, 3);
    const linalg::Matrix filled_matrix(2, 3, 4.5);

    EXPECT_EQ(zero_matrix.rows(), 2);
    EXPECT_EQ(zero_matrix.cols(), 3);
    EXPECT_EQ(zero_matrix.size(), 6);
    EXPECT_FALSE(zero_matrix.empty());

    for (linalg::Matrix::size_type row = 0; row < 2; ++row) {
        for (linalg::Matrix::size_type col = 0; col < 3; ++col) {
            EXPECT_DOUBLE_EQ(zero_matrix(row, col), 0.0);
            EXPECT_DOUBLE_EQ(filled_matrix(row, col), 4.5);
        }
    }
}

TEST(MatrixTest, FlatInitializerListFillsMatrixInRowMajorOrder) {
    const linalg::Matrix matrix(2, 2, {1.0, 2.0, 3.0, 4.0});

    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.cols(), 2);
    EXPECT_DOUBLE_EQ(matrix(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(matrix(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(matrix(1, 1), 4.0);
}

TEST(MatrixTest, FlatInitializerListPadsMissingValuesWithZeros) {
    const linalg::Matrix matrix(2, 3, {1.0, 2.0, 3.0, 4.0});

    EXPECT_EQ(matrix.size(), 6);
    EXPECT_DOUBLE_EQ(matrix(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(matrix(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(matrix(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(matrix(1, 1), 0.0);
    EXPECT_DOUBLE_EQ(matrix(1, 2), 0.0);
}

TEST(MatrixTest, FlatSequenceAcceptsVectorAndRejectsExcessValues) {
    const std::vector<double> values{1.0, 2.0, 3.0, 4.0};
    const linalg::Matrix matrix(2, 2, values);

    EXPECT_DOUBLE_EQ(matrix(1, 1), 4.0);
    EXPECT_THROW((void)linalg::Matrix(1, 2, {1.0, 2.0, 3.0}), std::invalid_argument);
}

TEST(MatrixTest, NestedInitializerListsCreateMatrixInRowMajorOrder) {
    const linalg::Matrix matrix{{1.0, 2.0}, {3.0, 4.0}};

    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.cols(), 2);
    EXPECT_EQ(matrix.size(), 4);
    EXPECT_DOUBLE_EQ(matrix(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(matrix(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(matrix(1, 1), 4.0);
}

TEST(MatrixTest, NestedInitializerListsPadShorterRowsWithZeros) {
    const linalg::Matrix matrix{{1.0, 2.0, 3.0}, {4.0}, {5.0, 6.0}};

    ASSERT_EQ(matrix.rows(), 3);
    ASSERT_EQ(matrix.cols(), 3);
    EXPECT_DOUBLE_EQ(matrix(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(matrix(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(matrix(1, 1), 0.0);
    EXPECT_DOUBLE_EQ(matrix(1, 2), 0.0);
    EXPECT_DOUBLE_EQ(matrix(2, 0), 5.0);
    EXPECT_DOUBLE_EQ(matrix(2, 1), 6.0);
    EXPECT_DOUBLE_EQ(matrix(2, 2), 0.0);
}

TEST(MatrixTest, NestedInitializerListsSupportEmptyRows) {
    const linalg::Matrix matrix{{}, {7.0, 8.0}, {}};

    ASSERT_EQ(matrix.rows(), 3);
    ASSERT_EQ(matrix.cols(), 2);
    EXPECT_DOUBLE_EQ(matrix(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(matrix(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(matrix(1, 0), 7.0);
    EXPECT_DOUBLE_EQ(matrix(1, 1), 8.0);
    EXPECT_DOUBLE_EQ(matrix(2, 0), 0.0);
    EXPECT_DOUBLE_EQ(matrix(2, 1), 0.0);
}

TEST(MatrixTest, StoresElementsInRowMajorOrder) {
    linalg::Matrix matrix(2, 3);

    matrix(0, 0) = 10.0;
    matrix(0, 1) = 11.0;
    matrix(0, 2) = 12.0;
    matrix(1, 0) = 20.0;
    matrix(1, 1) = 21.0;
    matrix(1, 2) = 22.0;

    const double expected[] = {10.0, 11.0, 12.0, 20.0, 21.0, 22.0};
    for (linalg::Matrix::size_type index = 0; index < 6; ++index) {
        EXPECT_DOUBLE_EQ(matrix.data()[index], expected[index]);
    }
}

TEST(MatrixTest, CheckedAccessRejectsInvalidIndices) {
    linalg::Matrix matrix(2, 3);

    matrix.at(1, 2) = 7.0;
    EXPECT_DOUBLE_EQ(matrix.at(1, 2), 7.0);

    EXPECT_THROW(matrix.at(2, 0), std::out_of_range);
    EXPECT_THROW(matrix.at(0, 3), std::out_of_range);
    EXPECT_THROW(matrix.at(2, 3), std::out_of_range);
}

TEST(MatrixTest, CopyOwnsIndependentStorage) {
    linalg::Matrix original(1, 2, 3.0);
    linalg::Matrix copy = original;

    copy(0, 0) = 9.0;

    EXPECT_DOUBLE_EQ(original(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(copy(0, 0), 9.0);
    EXPECT_NE(original.data(), copy.data());
}

TEST(MatrixTest, MoveConstructionTransfersContentsAndEmptiesSource) {
    static_assert(std::is_nothrow_move_constructible_v<linalg::Matrix>);

    linalg::Matrix source(2, 2, 6.0);
    linalg::Matrix destination(std::move(source));

    EXPECT_EQ(destination.rows(), 2);
    EXPECT_EQ(destination.cols(), 2);
    EXPECT_DOUBLE_EQ(destination(1, 1), 6.0);
    EXPECT_EQ(source.rows(), 0);
    EXPECT_EQ(source.cols(), 0);
}

TEST(MatrixTest, MoveAssignmentTransfersContentsAndEmptiesSource) {
    static_assert(std::is_nothrow_move_assignable_v<linalg::Matrix>);

    linalg::Matrix source(2, 3, 8.0);
    linalg::Matrix destination(1, 1, 1.0);

    destination = std::move(source);

    EXPECT_EQ(destination.rows(), 2);
    EXPECT_EQ(destination.cols(), 3);
    EXPECT_DOUBLE_EQ(destination(1, 2), 8.0);
    EXPECT_EQ(source.rows(), 0);
    EXPECT_EQ(source.cols(), 0);
}

TEST(MatrixTest, SupportsZeroDimensionsAndRejectsImpossibleStorageSize) {
    const linalg::Matrix zero_rows(0, 4);
    const linalg::Matrix zero_cols(4, 0);

    EXPECT_EQ(zero_rows.rows(), 0);
    EXPECT_EQ(zero_rows.cols(), 4);
    EXPECT_EQ(zero_rows.size(), 0);
    EXPECT_TRUE(zero_rows.empty());
    EXPECT_EQ(zero_cols.rows(), 4);
    EXPECT_EQ(zero_cols.cols(), 0);
    EXPECT_EQ(zero_cols.size(), 0);
    EXPECT_TRUE(zero_cols.empty());

    constexpr auto maximum = std::numeric_limits<linalg::Matrix::size_type>::max();
    EXPECT_THROW((void)linalg::Matrix(maximum, 2), std::length_error);
}
