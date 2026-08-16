#include <linalg/vector.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

TEST(VectorTest, DefaultConstructionCreatesAnEmptyVector) {
    const linalg::Vector vector;

    EXPECT_EQ(vector.size(), 0);
    EXPECT_TRUE(vector.empty());
}

TEST(VectorTest, SizeConstructionZeroInitializesValues) {
    const linalg::Vector vector(3);

    ASSERT_EQ(vector.size(), 3);
    EXPECT_FALSE(vector.empty());
    EXPECT_DOUBLE_EQ(vector[0], 0.0);
    EXPECT_DOUBLE_EQ(vector[1], 0.0);
    EXPECT_DOUBLE_EQ(vector[2], 0.0);
}

TEST(VectorTest, FillConstructionStoresValuesContiguously) {
    linalg::Vector vector(4, 2.5);
    vector[2] = 7.0;

    ASSERT_EQ(vector.size(), 4);
    EXPECT_DOUBLE_EQ(vector[0], 2.5);
    EXPECT_DOUBLE_EQ(vector[2], 7.0);
    EXPECT_EQ(&vector[0], vector.data());
    EXPECT_EQ(&vector[2], vector.data() + 2);
}

TEST(VectorTest, InitializerListConstructionStoresProvidedValues) {
    const linalg::Vector vector{1.0, 2.0, 3.0};

    ASSERT_EQ(vector.size(), 3);
    EXPECT_DOUBLE_EQ(vector[0], 1.0);
    EXPECT_DOUBLE_EQ(vector[1], 2.0);
    EXPECT_DOUBLE_EQ(vector[2], 3.0);
}

TEST(VectorTest, SpanConstructionCopiesValuesFromContiguousSequence) {
    std::vector<double> values{4.0, 5.0, 6.0};
    linalg::Vector vector(values);

    values[0] = 99.0;

    ASSERT_EQ(vector.size(), 3);
    EXPECT_DOUBLE_EQ(vector[0], 4.0);
    EXPECT_DOUBLE_EQ(vector[1], 5.0);
    EXPECT_DOUBLE_EQ(vector[2], 6.0);
    EXPECT_NE(vector.data(), values.data());
}

TEST(VectorTest, BracesRepresentValuesWhileParenthesesRepresentSize) {
    const linalg::Vector one_value{3.0};
    const linalg::Vector three_zeroes(3);

    ASSERT_EQ(one_value.size(), 1);
    EXPECT_DOUBLE_EQ(one_value[0], 3.0);
    EXPECT_EQ(three_zeroes.size(), 3);
    EXPECT_DOUBLE_EQ(three_zeroes[0], 0.0);
}

TEST(VectorTest, AtChecksBoundsForMutableAndConstVectors) {
    linalg::Vector vector(2, 1.0);
    vector.at(1) = 3.0;
    const auto& const_vector = vector;

    EXPECT_DOUBLE_EQ(const_vector.at(1), 3.0);
    EXPECT_THROW((void)vector.at(2), std::out_of_range);
    EXPECT_THROW((void)const_vector.at(2), std::out_of_range);
}

TEST(VectorTest, CopyConstructionCreatesIndependentStorage) {
    linalg::Vector original(3, 4.0);
    linalg::Vector copy(original);
    copy[0] = 9.0;

    EXPECT_DOUBLE_EQ(original[0], 4.0);
    EXPECT_DOUBLE_EQ(copy[0], 9.0);
    EXPECT_NE(original.data(), copy.data());
}

TEST(VectorTest, CopyAssignmentCreatesIndependentStorage) {
    linalg::Vector original(2, 5.0);
    linalg::Vector copy(1, 1.0);
    copy = original;
    copy[1] = 8.0;

    EXPECT_DOUBLE_EQ(original[1], 5.0);
    EXPECT_DOUBLE_EQ(copy[1], 8.0);
    EXPECT_NE(original.data(), copy.data());
}

TEST(VectorTest, MoveConstructionTransfersContentsAndEmptiesSource) {
    static_assert(std::is_nothrow_move_constructible_v<linalg::Vector>);

    linalg::Vector source(3, 6.0);
    linalg::Vector destination(std::move(source));

    EXPECT_EQ(destination.size(), 3);
    EXPECT_DOUBLE_EQ(destination[2], 6.0);
    EXPECT_TRUE(source.empty());
}

TEST(VectorTest, MoveAssignmentTransfersContentsAndEmptiesSource) {
    static_assert(std::is_nothrow_move_assignable_v<linalg::Vector>);

    linalg::Vector source(4, 8.0);
    linalg::Vector destination(1, 1.0);
    destination = std::move(source);

    EXPECT_EQ(destination.size(), 4);
    EXPECT_DOUBLE_EQ(destination[3], 8.0);
    EXPECT_TRUE(source.empty());
}

TEST(VectorTest, SupportsZeroSizeAndRejectsImpossibleStorageSize) {
    const linalg::Vector empty(0);
    EXPECT_TRUE(empty.empty());

    constexpr auto maximum = std::numeric_limits<linalg::Vector::size_type>::max();
    EXPECT_THROW((void)linalg::Vector(maximum), std::length_error);
}

