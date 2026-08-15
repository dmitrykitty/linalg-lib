#pragma once

#include <cstddef>
#include <vector>

namespace linalg {

class Matrix {
public:
    using size_type = std::size_t;

    Matrix() = default;
    Matrix(size_type rows, size_type cols);
    Matrix(size_type rows, size_type cols, double value);
    Matrix(const Matrix&) = default;
    Matrix(Matrix&& other) noexcept;
    ~Matrix() = default;

    Matrix& operator=(Matrix&& other) noexcept;
    Matrix& operator=(const Matrix&) = default;

    const double& operator()(size_type row, size_type col) const noexcept;
    double& operator()(size_type row, size_type col) noexcept;

    size_type rows() const noexcept {
        return rows_;
    }

    size_type cols() const noexcept {
        return cols_;
    }

    const double* data() const noexcept {
        return data_.data();
    }

    double* data() noexcept {
        return data_.data();
    }

    const double& at(size_type row, size_type col) const;
    double& at(size_type row, size_type col);

private:
    static size_type checked_element_count(size_type rows, size_type cols);

    size_type rows_ = 0;
    size_type cols_ = 0;
    std::vector<double> data_;
};

} // namespace linalg

