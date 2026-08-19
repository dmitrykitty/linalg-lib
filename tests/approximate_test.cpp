#include "support/approximate.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

TEST(ApproximateComparisonTest, UsesAbsoluteAndRelativeTolerance) {
    EXPECT_TRUE(linalg::test::almost_equal(0.0, 1.0e-10, 1.0e-9, 0.0));
    EXPECT_TRUE(linalg::test::almost_equal(1.0e9, 1.0e9 + 1.0, 0.0, 1.0e-9));
    EXPECT_FALSE(linalg::test::almost_equal(1.0, 1.1, 1.0e-3, 1.0e-3));
}

TEST(ApproximateComparisonTest, HandlesInfinityAndNanExplicitly) {
    constexpr double infinity = std::numeric_limits<double>::infinity();
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();

    EXPECT_TRUE(linalg::test::almost_equal(infinity, infinity, 0.0, 0.0));
    EXPECT_FALSE(linalg::test::almost_equal(infinity, -infinity, 0.0, 0.0));
    EXPECT_FALSE(linalg::test::almost_equal(nan, nan, 0.0, 0.0));
}

TEST(ApproximateComparisonTest, RejectsInvalidTolerances) {
    EXPECT_THROW(
        (void)linalg::test::almost_equal(1.0, 1.0, -1.0, 0.0),
        std::invalid_argument);
    EXPECT_THROW(
        (void)linalg::test::almost_equal(
            1.0,
            1.0,
            0.0,
            std::numeric_limits<double>::infinity()),
        std::invalid_argument);
}

TEST(ApproximateComparisonTest, ComparesVectorsAndRequiresMatchingSizes) {
    const linalg::Vector actual{1.0, 2.0, 3.0};
    const linalg::Vector close{1.0 + 1.0e-10, 2.0, 3.0};
    const linalg::Vector far{1.1, 2.0, 3.0};
    const linalg::Vector wrong_size{1.0, 2.0};

    EXPECT_TRUE(linalg::test::almost_equal(actual, close, 1.0e-9, 1.0e-9));
    EXPECT_FALSE(linalg::test::almost_equal(actual, far, 1.0e-9, 1.0e-9));
    EXPECT_FALSE(linalg::test::almost_equal(actual, wrong_size, 1.0e-9, 1.0e-9));
}

TEST(ApproximateComparisonTest, ComparesMatricesAndRequiresMatchingShapes) {
    const linalg::Matrix actual{{1.0, 2.0}, {3.0, 4.0}};
    const linalg::Matrix close{{1.0, 2.0 + 1.0e-10}, {3.0, 4.0}};
    const linalg::Matrix far{{1.0, 2.1}, {3.0, 4.0}};
    const linalg::Matrix wrong_shape(1, 4, 1.0);

    EXPECT_TRUE(linalg::test::almost_equal(actual, close, 1.0e-9, 1.0e-9));
    EXPECT_FALSE(linalg::test::almost_equal(actual, far, 1.0e-9, 1.0e-9));
    EXPECT_FALSE(linalg::test::almost_equal(actual, wrong_shape, 1.0e-9, 1.0e-9));
}
