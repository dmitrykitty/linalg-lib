#include <linalg/linalg.hpp>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename T>
constexpr bool construction_and_access_work_in_constant_expressions() {
    using Vector = linalg::StaticVector<T, 3>;
    const Vector zero;
    const Vector filled(T{7});
    const std::array<T, 2> values{T{4}, T{5}};
    const Vector from_span{std::span<const T>{values}};
    const Vector empty_input(std::initializer_list<T>{});
    Vector list{T{1}, T{2}};
    list.at(2) = T{3};
    list[0] = T{9};
    list.data()[1] = T{8};
    Vector copy = list;
    copy[0] = T{6};
    const Vector moved = std::move(copy);
    const linalg::StaticVector<T, 0> empty;

    return zero.size() == 3 && !zero.empty() && zero[2] == T{} && filled.at(2) == T{7} &&
           from_span[1] == T{5} && from_span[2] == T{} && empty_input[0] == T{} &&
           list.at(2) == T{3} && list[0] == T{9} && list.data()[1] == T{8} &&
           moved[0] == T{6} && copy[0] == T{6} && empty.size() == 0 && empty.empty();
}

static_assert(construction_and_access_work_in_constant_expressions<float>());
static_assert(construction_and_access_work_in_constant_expressions<double>());

template <typename T>
concept StaticVectorAvailable = requires { typename linalg::StaticVector<T, 3>; };

static_assert(StaticVectorAvailable<float>);
static_assert(StaticVectorAvailable<double>);
static_assert(!StaticVectorAvailable<int>);
static_assert(!StaticVectorAvailable<long double>);

using FixedVector = linalg::StaticVector<double, 3>;
static_assert(std::is_trivially_copyable_v<FixedVector>);
static_assert(std::is_nothrow_move_constructible_v<FixedVector>);
static_assert(std::is_nothrow_move_assignable_v<FixedVector>);
static_assert(!std::is_convertible_v<double, FixedVector>);
static_assert(!std::is_convertible_v<std::span<const double>, FixedVector>);
static_assert(!std::is_constructible_v<FixedVector, std::size_t, double>);
static_assert(!std::is_assignable_v<FixedVector&, linalg::StaticVector<double, 2>>);
static_assert(!std::is_assignable_v<FixedVector&, linalg::StaticVector<float, 3>>);

constexpr auto maximum_size = std::numeric_limits<std::size_t>::max();

template <typename T>
class StaticVectorTest : public ::testing::Test {};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(StaticVectorTest, ScalarTypes);

TYPED_TEST(StaticVectorTest, DefaultAndFillConstructionPreserveFixedLength) {
    using T = TypeParam;
    using Vector = linalg::StaticVector<T, 3>;
    const Vector zero;
    const Vector filled(T{4});

    EXPECT_EQ(zero.size(), 3U);
    EXPECT_FALSE(zero.empty());
    EXPECT_EQ(filled.size(), 3U);
    for (std::size_t index = 0; index < zero.size(); ++index) {
        EXPECT_EQ(zero[index], T{});
        EXPECT_EQ(filled[index], T{4});
    }
}

TYPED_TEST(StaticVectorTest, BracesContainValuesAndZeroPadMissingElements) {
    using T = TypeParam;
    using Vector = linalg::StaticVector<T, 3>;
    const Vector full{T{1}, T{2}, T{3}};
    const Vector partial{T{1}, T{2}};
    const Vector one_value{T{4}};
    const Vector empty_input(std::initializer_list<T>{});

    EXPECT_EQ(full[0], T{1});
    EXPECT_EQ(full[1], T{2});
    EXPECT_EQ(full[2], T{3});
    EXPECT_EQ(partial[1], T{2});
    EXPECT_EQ(partial[2], T{});
    EXPECT_EQ(one_value[0], T{4});
    EXPECT_EQ(one_value[1], T{});
    EXPECT_EQ(one_value[2], T{});
    for (std::size_t index = 0; index < empty_input.size(); ++index) {
        EXPECT_EQ(empty_input[index], T{});
    }
}

TYPED_TEST(StaticVectorTest, SpanConstructionCopiesSequencesAndPadsMissingElements) {
    using T = TypeParam;
    using Vector = linalg::StaticVector<T, 3>;
    std::array<T, 2> values{T{1}, T{2}};
    const Vector from_array{std::span<const T>{values}};
    std::vector<T> sequence{T{4}, T{5}, T{6}};
    const Vector from_sequence(sequence);
    const Vector empty_input{std::span<const T>{}};
    values[0] = T{9};
    sequence[0] = T{9};

    EXPECT_EQ(from_array[0], T{1});
    EXPECT_EQ(from_array[1], T{2});
    EXPECT_EQ(from_array[2], T{});
    EXPECT_NE(from_array.data(), values.data());
    EXPECT_EQ(from_sequence[0], T{4});
    EXPECT_EQ(from_sequence[2], T{6});
    EXPECT_NE(from_sequence.data(), sequence.data());
    for (std::size_t index = 0; index < empty_input.size(); ++index) {
        EXPECT_EQ(empty_input[index], T{});
    }
}

TYPED_TEST(StaticVectorTest, ConstructorsRejectInputsThatExceedTheFixedLength) {
    using T = TypeParam;
    using Vector = linalg::StaticVector<T, 3>;
    const std::array<T, 4> values{};

    EXPECT_THROW((void)(Vector{T{1}, T{2}, T{3}, T{4}}), std::invalid_argument);
    EXPECT_THROW((void)(Vector{std::span<const T>{values}}), std::invalid_argument);
}

TYPED_TEST(StaticVectorTest, MutableAndConstAccessMatchTheDynamicVectorCoreApi) {
    using T = TypeParam;
    using Vector = linalg::StaticVector<T, 3>;
    static_assert(std::is_same_v<typename Vector::value_type, T>);
    static_assert(std::is_same_v<typename Vector::size_type, std::size_t>);
    static_assert(std::is_same_v<decltype(std::declval<Vector&>().data()), T*>);
    static_assert(std::is_same_v<decltype(std::declval<const Vector&>().data()), const T*>);
    static_assert(std::is_same_v<decltype(std::declval<Vector&>()[0]), T&>);
    static_assert(std::is_same_v<decltype(std::declval<const Vector&>()[0]), const T&>);
    static_assert(std::is_same_v<decltype(std::declval<Vector&>().at(0)), T&>);
    static_assert(std::is_same_v<decltype(std::declval<const Vector&>().at(0)), const T&>);

    Vector vector;
    vector[0] = T{1};
    vector.at(1) = T{2};
    vector.data()[2] = T{3};
    const Vector& read_only = vector;
    EXPECT_EQ(read_only[0], T{1});
    EXPECT_EQ(read_only.at(1), T{2});
    EXPECT_EQ(read_only.data()[2], T{3});
    EXPECT_EQ(&read_only[0], read_only.data());
    EXPECT_EQ(&read_only[2], read_only.data() + 2);
}

TYPED_TEST(StaticVectorTest, CheckedAccessRejectsOutOfRangeIndices) {
    using T = TypeParam;
    linalg::StaticVector<T, 3> vector;
    const auto& read_only = vector;

    EXPECT_THROW(vector.at(3), std::out_of_range);
    EXPECT_THROW(vector.at(maximum_size), std::out_of_range);
    EXPECT_THROW(read_only.at(3), std::out_of_range);
    EXPECT_THROW(read_only.at(maximum_size), std::out_of_range);
}

TYPED_TEST(StaticVectorTest, ZeroLengthSupportsEmptyInputsAndRejectsElements) {
    using T = TypeParam;
    using Vector = linalg::StaticVector<T, 0>;
    Vector empty;
    const Vector filled(T{4});
    const Vector empty_list(std::initializer_list<T>{});
    const Vector empty_span{std::span<const T>{}};
    const std::array<T, 1> values{T{1}};

    EXPECT_EQ(empty.size(), 0U);
    EXPECT_TRUE(empty.empty());
    EXPECT_TRUE(filled.empty());
    EXPECT_TRUE(empty_list.empty());
    EXPECT_TRUE(empty_span.empty());
    EXPECT_THROW(empty.at(0), std::out_of_range);
    EXPECT_THROW(empty_span.at(0), std::out_of_range);
    EXPECT_THROW((void)(Vector{T{1}}), std::invalid_argument);
    EXPECT_THROW((void)(Vector{std::span<const T>{values}}), std::invalid_argument);
}

TYPED_TEST(StaticVectorTest, CopyConstructionAndAssignmentOwnIndependentStorage) {
    using T = TypeParam;
    using Vector = linalg::StaticVector<T, 3>;
    const Vector original{T{1}, T{2}, T{3}};
    Vector copy = original;
    copy[0] = T{9};

    EXPECT_EQ(original[0], T{1});
    EXPECT_EQ(copy[0], T{9});
    EXPECT_NE(original.data(), copy.data());

    Vector assigned;
    assigned = original;
    assigned[2] = T{8};
    EXPECT_EQ(original[2], T{3});
    EXPECT_EQ(assigned[2], T{8});
    EXPECT_NE(original.data(), assigned.data());
}

TYPED_TEST(StaticVectorTest, MovesKeepIndependentInlineStorageAndFixedLength) {
    using T = TypeParam;
    using Vector = linalg::StaticVector<T, 3>;
    Vector source{T{1}, T{2}, T{3}};
    Vector moved = std::move(source);

    EXPECT_EQ(moved[2], T{3});
    EXPECT_EQ(source.size(), 3U);
    EXPECT_FALSE(source.empty());
    EXPECT_EQ(source[2], T{3});
    EXPECT_NE(source.data(), moved.data());
    moved[2] = T{8};
    EXPECT_EQ(source[2], T{3});

    Vector assigned;
    assigned = std::move(moved);
    EXPECT_EQ(assigned[2], T{8});
    EXPECT_EQ(moved.size(), 3U);
    EXPECT_EQ(moved[2], T{8});
    EXPECT_NE(assigned.data(), moved.data());
}

TYPED_TEST(StaticVectorTest, FullInitializationMatchesDynamicVectorValues) {
    using T = TypeParam;
    const linalg::StaticVector<T, 3> fixed{T{1}, T{2}, T{3}};
    const linalg::DynamicVector<T> dynamic{T{1}, T{2}, T{3}};

    ASSERT_EQ(fixed.size(), dynamic.size());
    for (std::size_t index = 0; index < fixed.size(); ++index) {
        EXPECT_EQ(fixed[index], dynamic[index]);
    }
}

} // namespace
