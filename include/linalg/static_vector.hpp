#pragma once

#include <linalg/internal/storage.hpp>
#include <linalg/scalar.hpp>

#include <array>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <span>

namespace linalg {

// Fixed length and inline storage. Intended primarily for small vectors.
template <FloatingScalar T, std::size_t Size>
class StaticVector {
public:
    using size_type = std::size_t;
    using value_type = T;

    static_assert(Size <= std::numeric_limits<size_type>::max() / sizeof(T),
                  "StaticVector size exceeds representable storage size");

    constexpr StaticVector() = default;

    // Parentheses fill the fixed-size buffer; they do not change its length.
    explicit constexpr StaticVector(T value) noexcept {
        data_.fill(value);
    }

    // Braces contain values; missing elements remain zero.
    constexpr StaticVector(std::initializer_list<T> values)
        : StaticVector(std::span<const T>{values.begin(), values.size()}) {}

    explicit constexpr StaticVector(std::span<const T> values) {
        internal::validate_input_fits(values.size(), size());
        internal::copy_values(values, data_.data());
    }

    // std::array supplies copy/move/destruction; every object keeps its fixed length.

    // A valid index is a precondition in Release builds.
    constexpr const T& operator[](size_type index) const noexcept {
        return internal::vector_element(data_.data(), size(), index);
    }

    constexpr T& operator[](size_type index) noexcept {
        return internal::vector_element(data_.data(), size(), index);
    }

    constexpr const T& at(size_type index) const {
        return internal::checked_vector_element(data_.data(), size(), index);
    }

    constexpr T& at(size_type index) {
        return internal::checked_vector_element(data_.data(), size(), index);
    }

    constexpr size_type size() const noexcept {
        return data_.size();
    }

    constexpr bool empty() const noexcept {
        return data_.empty();
    }

    constexpr const T* data() const noexcept {
        return data_.data();
    }

    constexpr T* data() noexcept {
        return data_.data();
    }

private:
    std::array<T, Size> data_{};
};

} // namespace linalg
