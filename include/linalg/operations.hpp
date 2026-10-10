#pragma once

#include <linalg/internal/kernels.hpp>
#include <linalg/matrix.hpp>
#include <linalg/static_matrix.hpp>
#include <linalg/static_vector.hpp>
#include <linalg/vector.hpp>

#include <stdexcept>

namespace linalg {

enum class VectorNorm {
    /// Sum of absolute values: |x0| + |x1| + ...
    l1,
    /// Euclidean length: sqrt(x0^2 + x1^2 + ...)
    l2,
    /// Largest absolute element: max(|xi|)
    infinity,
};

enum class MatrixNorm {
    /// Largest absolute column sum.
    one,
    /// Square root of the sum of squares of every matrix element.
    frobenius,
    /// Largest absolute row sum.
    infinity,
};

Matrix add_to_each_row(const Matrix& matrix, const Vector& values);
Matrix add_to_each_column(const Matrix& matrix, const Vector& values);

Matrix repeat_as_rows(const Vector& values, Matrix::size_type row_count);
Matrix repeat_as_columns(const Vector& values, Matrix::size_type column_count);

// Overloads choose the result's ownership/shape; calculation stays in one kernel.
template <FloatingScalar T>
DynamicMatrix<T> transpose(const DynamicMatrix<T>& matrix) {
    DynamicMatrix<T> result(matrix.cols(), matrix.rows());
    internal::transpose_into(matrix, result);
    return result;
}

template <FloatingScalar T, std::size_t Rows, std::size_t Cols>
constexpr StaticMatrix<T, Cols, Rows> transpose(const StaticMatrix<T, Rows, Cols>& matrix) {
    StaticMatrix<T, Cols, Rows> result;
    internal::transpose_into(matrix, result);
    return result;
}

double trace(const Matrix&);

double norm(const Vector& vector, VectorNorm type);
double norm(const Matrix& matrix, MatrixNorm type);

/// Returns a new vector whose L2 norm is 1. Scaled arithmetic is used instead of
/// dividing by norm() so extreme finite values avoid overflow or underflow.
Vector normalize(const Vector& vector);

template <FloatingScalar T>
DynamicVector<T> multiply(const DynamicMatrix<T>& matrix, const DynamicVector<T>& vector) {
    if (matrix.cols() != vector.size()) {
        throw std::invalid_argument("vector size must equal matrix column count");
    }
    DynamicVector<T> result(matrix.rows());
    internal::matvec_into(matrix, vector, result);
    return result;
}

template <FloatingScalar T, std::size_t Rows, std::size_t Cols>
constexpr StaticVector<T, Rows> multiply(
    const StaticMatrix<T, Rows, Cols>& matrix, const StaticVector<T, Cols>& vector) {
    StaticVector<T, Rows> result;
    internal::matvec_into(matrix, vector, result);
    return result;
}

template <FloatingScalar T>
DynamicMatrix<T> multiply(const DynamicMatrix<T>& left, const DynamicMatrix<T>& right) {
    if (left.cols() != right.rows()) {
        throw std::invalid_argument("left column count must equal right row count");
    }
    DynamicMatrix<T> result(left.rows(), right.cols());
    internal::matmul_into(left, right, result);
    return result;
}

template <FloatingScalar T, std::size_t Rows, std::size_t Inner, std::size_t Cols>
constexpr StaticMatrix<T, Rows, Cols> multiply(
    const StaticMatrix<T, Rows, Inner>& left, const StaticMatrix<T, Inner, Cols>& right) {
    StaticMatrix<T, Rows, Cols> result;
    internal::matmul_into(left, right, result);
    return result;
}

} // namespace linalg
