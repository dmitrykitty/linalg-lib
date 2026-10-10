#pragma once

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <stdexcept>

namespace linalg::internal {

// Storage-independent helpers; these are implementation details, not public API.
constexpr void check_matrix_value_count(std::size_t count, std::size_t capacity) {
    if (count > capacity) {
        throw std::invalid_argument("too many values for matrix dimensions");
    }
}

// The owner checks capacity and zero-initializes storage before copying values.
template <typename T>
constexpr void copy_matrix_values(std::span<const T> values, T* destination) noexcept {
    for (std::size_t index = 0; index < values.size(); ++index) {
        destination[index] = values[index];
    }
}

// The owner validates the shape and zero-initializes storage before calling this.
template <typename T>
constexpr void copy_matrix_rows(
    std::initializer_list<std::initializer_list<T>> rows, std::size_t cols,
    T* destination) noexcept {
    std::size_t row_index = 0;
    for (const auto& row : rows) {
        std::size_t col_index = 0;
        for (const T value : row) {
            destination[row_index * cols + col_index] = value;
            ++col_index;
        }
        ++row_index;
    }
}

template <typename T>
constexpr T& matrix_element(T* data, std::size_t rows, std::size_t cols,
                           std::size_t row, std::size_t col) noexcept {
    // Keep rows used when assertions are disabled in Release builds.
    (void)rows;
    assert(row < rows);
    assert(col < cols);
    return data[row * cols + col];
}

template <typename T>
constexpr T& checked_matrix_element(T* data, std::size_t rows, std::size_t cols,
                                   std::size_t row, std::size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("matrix index is out of range");
    }
    return data[row * cols + col];
}

} // namespace linalg::internal
