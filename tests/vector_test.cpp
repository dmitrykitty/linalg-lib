#include <linalg/vector.hpp>

#include <gtest/gtest.h>

#include <type_traits>

TEST(VectorSkeletonTest, IsAnEmptyDefaultConstructibleType) {
    static_assert(std::is_default_constructible_v<linalg::Vector>);
    static_assert(std::is_empty_v<linalg::Vector>);

    [[maybe_unused]] const linalg::Vector vector;
    SUCCEED();
}

