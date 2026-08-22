#include <linalg/vector.hpp>

#include <cassert>
#include <stdexcept>
#include <utility>

namespace linalg {

Vector::Vector(size_type size) : Vector(size, 0.0) {}

Vector::Vector(size_type size, double value) : data_(checked_size(size), value) {}

Vector::Vector(std::initializer_list<double> values) : data_(values) {}

Vector::Vector(std::span<const double> values) : data_(values.begin(), values.end()) {}

Vector::Vector(Vector&& other) noexcept : data_(std::move(other.data_)) {
    other.data_.clear();
}

Vector& Vector::operator=(Vector&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    data_ = std::move(other.data_);
    other.data_.clear();
    return *this;
}

Vector& Vector::operator+=(const Vector& other) {
    if (size() != other.size()) {
        throw std::invalid_argument("vector sizes must match for addition");
    }

    for (size_type index = 0; index < size(); ++index) {
        data_[index] += other.data_[index];
    }
    return *this;
}

Vector& Vector::operator-=(const Vector& other) {
    if (size() != other.size()) {
        throw std::invalid_argument("vector sizes must match for subtraction");
    }

    for (size_type index = 0; index < size(); ++index) {
        data_[index] -= other.data_[index];
    }
    return *this;
}

Vector& Vector::operator+=(double scalar) noexcept {
    for (double& value : data_) {
        value += scalar;
    }
    return *this;
}

Vector& Vector::operator-=(double scalar) noexcept {
    for (double& value : data_) {
        value -= scalar;
    }
    return *this;
}

Vector& Vector::operator*=(double scalar) noexcept {
    for (double& value : data_) {
        value *= scalar;
    }
    return *this;
}

const double& Vector::at(size_type index) const {
    if (index >= size()) {
        throw std::out_of_range("vector index is out of range");
    }
    return data_[index];
}

double& Vector::at(size_type index) {
    if (index >= size()) {
        throw std::out_of_range("vector index is out of range");
    }
    return data_[index];
}

Vector::size_type Vector::checked_size(size_type size) {
    if (size > std::vector<double>{}.max_size()) {
        throw std::length_error("vector size is too large");
    }
    return size;
}

Vector operator+(Vector left, const Vector& right) {
    left += right;
    return left;
}

Vector operator-(Vector left, const Vector& right) {
    left -= right;
    return left;
}

Vector operator*(Vector vector, double scalar) {
    vector *= scalar;
    return vector;
}

Vector operator*(double scalar, Vector vector) {
    vector *= scalar;
    return vector;
}

} // namespace linalg

