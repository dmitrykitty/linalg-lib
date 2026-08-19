#pragma once

#include <cstddef>
#include <initializer_list>
#include <span>
#include <vector>

namespace linalg {

class Vector {
public:
    using size_type = std::size_t;

    Vector() = default;
    explicit Vector(size_type size);
    Vector(size_type size, double value);
    Vector(std::initializer_list<double> values);
    explicit Vector(std::span<const double> values);
    Vector(const Vector&) = default;
    Vector(Vector&& other) noexcept;
    ~Vector() = default;

    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&& other) noexcept;

    Vector& operator+=(const Vector& other);
    Vector& operator-=(const Vector& other);
    Vector& operator+=(double scalar) noexcept;
    Vector& operator-=(double scalar) noexcept;
    Vector& operator*=(double scalar) noexcept;

    const double& operator[](size_type index) const noexcept;
    double& operator[](size_type index) noexcept;

    size_type size() const noexcept {
        return data_.size();
    }

    bool empty() const noexcept {
        return data_.empty();
    }

    const double* data() const noexcept {
        return data_.data();
    }

    double* data() noexcept {
        return data_.data();
    }

    const double& at(size_type index) const;
    double& at(size_type index);

private:
    static size_type checked_size(size_type size);

    std::vector<double> data_;
};

Vector operator+(Vector left, const Vector& right);
Vector operator-(Vector left, const Vector& right);
Vector operator*(Vector vector, double scalar);
Vector operator*(double scalar, Vector vector);

} // namespace linalg

