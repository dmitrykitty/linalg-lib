#pragma once

#include <linalg/scalar.hpp>

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace linalg {

template <FloatingScalar T>
class DynamicMatrix {
public:
    using size_type = std::size_t;
    using value_type = T;

    DynamicMatrix() = default;
    DynamicMatrix(size_type rows, size_type cols);
    DynamicMatrix(size_type rows, size_type cols, T value);
    DynamicMatrix(size_type rows, size_type cols, std::initializer_list<T> values);
    DynamicMatrix(size_type rows, size_type cols, std::span<const T> values);
    DynamicMatrix(std::initializer_list<std::initializer_list<T>> rows);
    DynamicMatrix(const DynamicMatrix&) = default;
    DynamicMatrix(DynamicMatrix&& other) noexcept;
    ~DynamicMatrix() = default;

    DynamicMatrix& operator=(DynamicMatrix&& other) noexcept;
    DynamicMatrix& operator=(const DynamicMatrix&) = default;

    // Valid row and column indices are a precondition in Release builds.
    const T& operator()(size_type row, size_type col) const noexcept {
        assert(row < rows_);
        assert(col < cols_);
        return data_[row * cols_ + col];
    }

    T& operator()(size_type row, size_type col) noexcept {
        assert(row < rows_);
        assert(col < cols_);
        return data_[row * cols_ + col];
    }

    DynamicMatrix& operator+=(const DynamicMatrix&);
    DynamicMatrix& operator-=(const DynamicMatrix&);
    DynamicMatrix& operator+=(T scalar) noexcept;
    DynamicMatrix& operator-=(T scalar) noexcept;
    DynamicMatrix& operator*=(T scalar) noexcept;

    size_type rows() const noexcept {
        return rows_;
    }

    size_type cols() const noexcept {
        return cols_;
    }

    size_type size() const noexcept {
        return data_.size();
    }

    bool empty() const noexcept {
        return data_.empty();
    }

    const T* data() const noexcept {
        return data_.data();
    }

    T* data() noexcept {
        return data_.data();
    }

    const T& at(size_type row, size_type col) const;
    T& at(size_type row, size_type col);

private:
    static size_type checked_element_count(size_type rows, size_type cols);

    size_type rows_ = 0;
    size_type cols_ = 0;
    std::vector<T> data_;
};

// Preserve the existing double-precision source API.
using Matrix = DynamicMatrix<double>;

template <FloatingScalar T>
DynamicMatrix<T> operator+(DynamicMatrix<T> left, const DynamicMatrix<T>& right) {
    left += right;
    return left;
}

template <FloatingScalar T>
DynamicMatrix<T> operator-(DynamicMatrix<T> left, const DynamicMatrix<T>& right) {
    left -= right;
    return left;
}

template <FloatingScalar T>
DynamicMatrix<T> operator*(DynamicMatrix<T> matrix, T scalar) {
    matrix *= scalar;
    return matrix;
}

template <FloatingScalar T>
DynamicMatrix<T> operator*(T scalar, DynamicMatrix<T> matrix) {
    matrix *= scalar;
    return matrix;
}

// Template definitions stay in this header so every user can instantiate the container.
template <FloatingScalar T>
DynamicMatrix<T>::DynamicMatrix(size_type rows, size_type cols)
    : DynamicMatrix(rows, cols, T{}) {}

template <FloatingScalar T>
DynamicMatrix<T>::DynamicMatrix(size_type rows, size_type cols, T value)
    : rows_(rows), cols_(cols), data_(checked_element_count(rows, cols), value) {}

template <FloatingScalar T>
DynamicMatrix<T>::DynamicMatrix(
    size_type rows, size_type cols, std::initializer_list<T> values)
    : DynamicMatrix(rows, cols, std::span<const T>{values.begin(), values.size()}) {}

template <FloatingScalar T>
DynamicMatrix<T>::DynamicMatrix(size_type rows, size_type cols, std::span<const T> values)
    : rows_(rows), cols_(cols) {
    const size_type element_count = checked_element_count(rows, cols);
    if (values.size() > element_count) {
        throw std::invalid_argument("too many values for matrix dimensions");
    }

    data_.assign(element_count, T{});
    size_type index = 0;
    for (const T value : values) {
        data_[index] = value;
        ++index;
    }
}

template <FloatingScalar T>
DynamicMatrix<T>::DynamicMatrix(std::initializer_list<std::initializer_list<T>> rows)
    : rows_(rows.size()) {
    for (const auto& row : rows) {
        if (row.size() > cols_) {
            cols_ = row.size();
        }
    }

    data_.assign(checked_element_count(rows_, cols_), T{});
    size_type row_index = 0;
    for (const auto& row : rows) {
        size_type col_index = 0;
        for (const T value : row) {
            data_[row_index * cols_ + col_index] = value;
            ++col_index;
        }
        ++row_index;
    }
}

template <FloatingScalar T>
DynamicMatrix<T>::DynamicMatrix(DynamicMatrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_)) {
    other.rows_ = 0;
    other.cols_ = 0;
    other.data_.clear();
}

template <FloatingScalar T>
DynamicMatrix<T>& DynamicMatrix<T>::operator=(DynamicMatrix&& other) noexcept {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = std::move(other.data_);
        other.rows_ = 0;
        other.cols_ = 0;
        other.data_.clear();
    }
    return *this;
}

template <FloatingScalar T>
DynamicMatrix<T>& DynamicMatrix<T>::operator+=(const DynamicMatrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("matrix has wrong size");
    }
    for (size_type index = 0; index < data_.size(); ++index) {
        data_[index] += other.data_[index];
    }
    return *this;
}

template <FloatingScalar T>
DynamicMatrix<T>& DynamicMatrix<T>::operator-=(const DynamicMatrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("matrix has wrong size");
    }
    for (size_type index = 0; index < data_.size(); ++index) {
        data_[index] -= other.data_[index];
    }
    return *this;
}

template <FloatingScalar T>
DynamicMatrix<T>& DynamicMatrix<T>::operator+=(T scalar) noexcept {
    for (T& value : data_) {
        value += scalar;
    }
    return *this;
}

template <FloatingScalar T>
DynamicMatrix<T>& DynamicMatrix<T>::operator-=(T scalar) noexcept {
    for (T& value : data_) {
        value -= scalar;
    }
    return *this;
}

template <FloatingScalar T>
DynamicMatrix<T>& DynamicMatrix<T>::operator*=(T scalar) noexcept {
    for (T& value : data_) {
        value *= scalar;
    }
    return *this;
}

template <FloatingScalar T>
const T& DynamicMatrix<T>::at(size_type row, size_type col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("matrix index is out of range");
    }
    return data_[row * cols_ + col];
}

template <FloatingScalar T>
T& DynamicMatrix<T>::at(size_type row, size_type col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("matrix index is out of range");
    }
    return data_[row * cols_ + col];
}

template <FloatingScalar T>
DynamicMatrix<T>::size_type DynamicMatrix<T>::checked_element_count(
    size_type rows, size_type cols) {
    const size_type max_size = std::vector<T>{}.max_size();
    if (rows != 0 && cols > max_size / rows) {
        throw std::length_error("matrix dimensions are too large");
    }
    return rows * cols;
}

} // namespace linalg
