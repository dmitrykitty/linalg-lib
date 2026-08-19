#include <linalg/matrix.hpp>
#include <stdexcept>
#include <utility>
#include <cassert>
#include <cstddef>


namespace linalg { 

Matrix::Matrix(size_type rows, size_type cols): Matrix(rows, cols, 0.0) {} 

Matrix::Matrix(size_type rows, size_type cols, double value)
    : rows_(rows), cols_(cols), data_(checked_element_count(rows, cols), value) {}

Matrix::Matrix(size_type rows, size_type cols, std::initializer_list<double> values)
    : Matrix(rows, cols, std::span<const double>{values.begin(), values.size()}) {}

Matrix::Matrix(size_type rows, size_type cols, std::span<const double> values)
    : rows_(rows), cols_(cols) {
    const auto element_count = checked_element_count(rows, cols);
    if (values.size() > element_count) {
        throw std::invalid_argument("too many values for matrix dimensions");
    }

    data_.assign(element_count, 0.0);
    size_type index = 0;
    for (const double value : values) {
        data_[index] = value;
        ++index;
    }
}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> rows)
    : rows_(rows.size()) {
    for (const auto& row : rows) {
        if (row.size() > cols_) {
            cols_ = row.size();
        }
    }

    data_.assign(checked_element_count(rows_, cols_), 0.0);

    size_type row_index = 0;
    for (const auto& row : rows) {
        size_type col_index = 0;
        for (const double value : row) {
            data_[row_index * cols_ + col_index] = value;
            ++col_index;
        }
        ++row_index;
    }
}

Matrix::Matrix(Matrix&& o) noexcept: 
    rows_(o.rows_), 
    cols_(o.cols_), 
    data_(std::move(o.data_)) {
    o.rows_ = 0;
    o.cols_ = 0; 
    o.data_.clear(); 
}

Matrix& Matrix::operator=(Matrix&& o) noexcept {
    rows_ = o.rows_; 
    if (this == &o) {
        return *this;
    }

    cols_ = o.cols_; 
    data_ = std::move(o.data_);

    o.rows_ = 0;
    o.cols_ = 0; 
    o.data_.clear(); 
    return *this;
}

const double& Matrix::operator()(size_type row, size_type col) const noexcept {
    assert(row < rows_);
    assert(col < cols_);
    return data_[row * cols_ + col];
}

double& Matrix::operator()(size_type row, size_type col) noexcept {
    assert(row < rows_);
    assert(col < cols_);
    return data_[row * cols_ + col];
}

Matrix& Matrix::operator+=(const Matrix& o) {
    if(rows_ != o.rows_ || cols_ != o.cols_) {
        throw std::invalid_argument("matrix has wrong size");
    }

    for(size_type i = 0; i < data_.size(); ++i) {
        data_[i] += o.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& o) {
    if(rows_ != o.rows_ || cols_ != o.cols_) {
        throw std::invalid_argument("matrix has wrong size");
    }

    for(size_type i = 0; i < data_.size(); ++i) {
        data_[i] -= o.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator+=(double scalar) noexcept {
    for (double& value : data_) {
        value += scalar;
    }
    return *this;
}

Matrix& Matrix::operator-=(double scalar) noexcept {
    for (double& value : data_) {
        value -= scalar;
    }
    return *this;
}

Matrix& Matrix::operator*=(double scalar) noexcept {
    for(size_type i = 0; i < data_.size(); ++i) {
        data_[i] *= scalar;
    }
    return *this;
}

const double& Matrix::at(size_type row, size_type col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("matrix index is out of range");
    }
    return data_[row * cols_ + col];
}

double& Matrix::at(size_type row, size_type col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("matrix index is out of range");
    }
    return data_[row * cols_ + col];
} 

Matrix::size_type Matrix::checked_element_count(size_type rows, size_type cols) {
    const auto max_size = std::vector<double>{}.max_size();

    if (rows != 0 && cols > max_size / rows) {
        throw std::length_error("matrix dimensions are too large");
    }
    return rows * cols;
}

//==========================FUNCTION OPERATORS====================================

Matrix operator+(Matrix left, const Matrix& right) {
    left += right;
    return left;
}

Matrix operator-(Matrix left, const Matrix& right) {
    left -= right;
    return left;
}

Matrix operator*(Matrix matrix, double scalar) {
    matrix *= scalar;
    return matrix;
}

Matrix operator*(double scalar, Matrix matrix) {
    matrix *= scalar;
    return matrix;
}

} // namespace linalg