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
class DynamicVector {
public:
    using size_type = std::size_t;
    using value_type = T;

    DynamicVector() = default;
    explicit DynamicVector(size_type size);
    DynamicVector(size_type size, T value);
    DynamicVector(std::initializer_list<T> values);
    explicit DynamicVector(std::span<const T> values);
    DynamicVector(const DynamicVector&) = default;
    DynamicVector(DynamicVector&& other) noexcept;
    ~DynamicVector() = default;

    DynamicVector& operator=(const DynamicVector&) = default;
    DynamicVector& operator=(DynamicVector&& other) noexcept;

    DynamicVector& operator+=(const DynamicVector& other);
    DynamicVector& operator-=(const DynamicVector& other);
    DynamicVector& operator+=(T scalar) noexcept;
    DynamicVector& operator-=(T scalar) noexcept;
    DynamicVector& operator*=(T scalar) noexcept;

    const T& operator[](size_type index) const noexcept {
        assert(index < size());
        return data_[index];
    }

    T& operator[](size_type index) noexcept {
        assert(index < size());
        return data_[index];
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

    const T& at(size_type index) const;
    T& at(size_type index);

private:
    static size_type checked_size(size_type size);

    std::vector<T> data_;
};

// Preserve the existing double-precision source API.
using Vector = DynamicVector<double>;

template <FloatingScalar T>
DynamicVector<T> operator+(DynamicVector<T> left, const DynamicVector<T>& right) {
    left += right;
    return left;
}

template <FloatingScalar T>
DynamicVector<T> operator-(DynamicVector<T> left, const DynamicVector<T>& right) {
    left -= right;
    return left;
}

template <FloatingScalar T>
DynamicVector<T> operator*(DynamicVector<T> vector, T scalar) {
    vector *= scalar;
    return vector;
}

template <FloatingScalar T>
DynamicVector<T> operator*(T scalar, DynamicVector<T> vector) {
    vector *= scalar;
    return vector;
}

template <FloatingScalar T>
DynamicVector<T>::DynamicVector(size_type size) : DynamicVector(size, T{}) {}

template <FloatingScalar T>
DynamicVector<T>::DynamicVector(size_type size, T value)
    : data_(checked_size(size), value) {}

template <FloatingScalar T>
DynamicVector<T>::DynamicVector(std::initializer_list<T> values) : data_(values) {}

template <FloatingScalar T>
DynamicVector<T>::DynamicVector(std::span<const T> values)
    : data_(values.begin(), values.end()) {}

template <FloatingScalar T>
DynamicVector<T>::DynamicVector(DynamicVector&& other) noexcept
    : data_(std::move(other.data_)) {
    other.data_.clear();
}

template <FloatingScalar T>
DynamicVector<T>& DynamicVector<T>::operator=(DynamicVector&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        other.data_.clear();
    }
    return *this;
}

template <FloatingScalar T>
DynamicVector<T>& DynamicVector<T>::operator+=(const DynamicVector& other) {
    if (size() != other.size()) {
        throw std::invalid_argument("vector sizes must match for addition");
    }

    for (size_type index = 0; index < size(); ++index) {
        data_[index] += other.data_[index];
    }
    return *this;
}

template <FloatingScalar T>
DynamicVector<T>& DynamicVector<T>::operator-=(const DynamicVector& other) {
    if (size() != other.size()) {
        throw std::invalid_argument("vector sizes must match for subtraction");
    }

    for (size_type index = 0; index < size(); ++index) {
        data_[index] -= other.data_[index];
    }
    return *this;
}

template <FloatingScalar T>
DynamicVector<T>& DynamicVector<T>::operator+=(T scalar) noexcept {
    for (T& value : data_) {
        value += scalar;
    }
    return *this;
}

template <FloatingScalar T>
DynamicVector<T>& DynamicVector<T>::operator-=(T scalar) noexcept {
    for (T& value : data_) {
        value -= scalar;
    }
    return *this;
}

template <FloatingScalar T>
DynamicVector<T>& DynamicVector<T>::operator*=(T scalar) noexcept {
    for (T& value : data_) {
        value *= scalar;
    }
    return *this;
}

template <FloatingScalar T>
const T& DynamicVector<T>::at(size_type index) const {
    if (index >= size()) {
        throw std::out_of_range("vector index is out of range");
    }
    return data_[index];
}

template <FloatingScalar T>
T& DynamicVector<T>::at(size_type index) {
    if (index >= size()) {
        throw std::out_of_range("vector index is out of range");
    }
    return data_[index];
}

template <FloatingScalar T>
DynamicVector<T>::size_type DynamicVector<T>::checked_size(size_type size) {
    if (size > std::vector<T>{}.max_size()) {
        throw std::length_error("vector size is too large");
    }
    return size;
}

} // namespace linalg
