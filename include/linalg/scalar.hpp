#pragma once

#include <concepts>

namespace linalg {

// The first scalar-generic milestone supports these two floating-point types.
template <typename T>
concept FloatingScalar = std::same_as<T, float> || std::same_as<T, double>;

} // namespace linalg
