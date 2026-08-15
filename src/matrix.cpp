#include <linalg/matrix.hpp>
#include <stdexcept>
#include <utility>
#include <cassert>
#include <cstddef>


namespace linalg { 

Matrix::Matrix(size_type rows, size_type cols): Matrix(rows, cols, 0.0) {} 

Matrix::Matrix(size_type rows, size_type cols, double val): 
    rows_(rows), 
    cols_(cols),
    data_((std::vector<double>(checked_element_count(rows_, cols_), val))) {}

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
    cols_ = o.cols_; 
    data_ = std::move(o.data_);

    o.rows_ = 0;
    o.cols_ = 0; 
    o.data_.clear(); 
    return *this;
}

const double& Matrix::operator()(size_type row, size_type col) const noexcept {
    assert(row <= rows_);
    assert(col <= cols_);
    return data_[row * cols_ + col];
}

double& Matrix::operator()(size_type row, size_type col) noexcept {
    assert(row <= rows_);
    assert(col <= cols_);
    return data_[row * cols_ + col];
}

const double& Matrix::at(size_type row, size_type col) const {
    if(rows_ <= row || cols_ <= col) {
        throw std::out_of_range("wrong index");
    }
    return data_[row * cols_ + col];
}

double& Matrix::at(size_type row, size_type col) {
    if(rows_ <= row || cols_ <= col) {
        throw std::out_of_range("Wrong index");
    }
    return data_[row * cols_ + col];
} 

Matrix::size_type Matrix::checked_element_count(size_type rows, size_type cols) {
    const auto max_size = std::vector<double>{}.max_size();

    if(rows != 0 && max_size / rows < cols) {
        throw std::length_error("matrix dimensions are too large");
    }
    return rows * cols;
}

} // namespace linalg