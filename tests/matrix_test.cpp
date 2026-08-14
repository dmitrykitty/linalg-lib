#include <linalg/matrix.hpp>

#include <gtest/gtest.h>

#include <type_traits>

TEST(MatrixSkeletonTest, IsAnEmptyDefaultConstructibleType) {
    static_assert(std::is_default_constructible_v<linalg::Matrix>);
    static_assert(std::is_empty_v<linalg::Matrix>);

    [[maybe_unused]] const linalg::Matrix matrix;
    SUCCEED();
}

