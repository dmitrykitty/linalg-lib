#include <linalg/vector.hpp>

#include <cassert>
#include <stdexcept>
#include <utility>

namespace linalg {

Vector::Vector(size_type size) : Vector(size, 0.0) {}

Vector::Vector(size_type size, double value) : data_(checked_size(size), value) {}

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

const double& Vector::operator[](size_type index) const noexcept {
    assert(index < size());
    return data_[index];
}

double& Vector::operator[](size_type index) noexcept {
    assert(index < size());
    return data_[index];
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

} // namespace linalg

