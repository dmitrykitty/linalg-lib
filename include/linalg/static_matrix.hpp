#pragma once

#include <linalg/internal/matrix_storage.hpp>
#include <linalg/scalar.hpp>

#include <array>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <span>

namespace linalg {

// Fixed dimensions and inline row-major storage. Intended primarily for small matrices.
template <FloatingScalar T, std::size_t Rows, std::size_t Cols>
class StaticMatrix {
public:
    using size_type = std::size_t;
    using value_type = T;

    // Check before multiplying, including the number of bytes required by the array.
    static_assert(Rows == 0 || Cols <= (std::numeric_limits<size_type>::max() / sizeof(T)) / Rows,
                  "StaticMatrix dimensions exceed representable storage size");

    constexpr StaticMatrix() = default;

    explicit constexpr StaticMatrix(T value) noexcept {
        data_.fill(value);
    }

    // A named flat-list factory avoids ambiguity with single-element/empty rows.
    static constexpr StaticMatrix from_values(std::initializer_list<T> values) {
        return StaticMatrix(std::span<const T>{values.begin(), values.size()});
    }

    explicit constexpr StaticMatrix(std::span<const T> values) {
        internal::validate_input_fits(values.size(), size());
        internal::copy_matrix_values(values, data_.data());
    }

    // Missing rows/elements stay zero; supplied rows must fit the fixed shape.
    constexpr StaticMatrix(std::initializer_list<std::initializer_list<T>> rows) {
        internal::validate_input_fits(rows.size(), Rows);
        for (const auto& row : rows) {
            internal::validate_input_fits(row.size(), Cols);
        }
        internal::copy_matrix_rows(rows, Cols, data_.data());
    }

    // std::array supplies copy/move/destruction; every object keeps its fixed shape.

    // Valid row and column indices are a precondition in Release builds.
    constexpr const T& operator()(size_type row, size_type col) const noexcept {
        return internal::matrix_element(data_.data(), Rows, Cols, row, col);
    }

    constexpr T& operator()(size_type row, size_type col) noexcept {
        return internal::matrix_element(data_.data(), Rows, Cols, row, col);
    }

    constexpr const T& at(size_type row, size_type col) const {
        return internal::checked_matrix_element(data_.data(), Rows, Cols, row, col);
    }

    constexpr T& at(size_type row, size_type col) {
        return internal::checked_matrix_element(data_.data(), Rows, Cols, row, col);
    }

    constexpr size_type rows() const noexcept {
        return Rows;
    }

    constexpr size_type cols() const noexcept {
        return Cols;
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
    std::array<T, Rows * Cols> data_{};
};

} // namespace linalg
