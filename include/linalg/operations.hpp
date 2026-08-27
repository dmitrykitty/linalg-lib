#pragma once

#include <linalg/matrix.hpp>
#include <linalg/vector.hpp>

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

Matrix transpose(const Matrix&);
double trace(const Matrix&);

double norm(const Vector& vector, VectorNorm type);
double norm(const Matrix& matrix, MatrixNorm type);

/// Returns a new vector whose L2 norm is 1. Scaled arithmetic is used instead of
/// dividing by norm() so extreme finite values avoid overflow or underflow.
Vector normalize(const Vector& vector);

Vector multiply(const Matrix& matrix, const Vector& vector);
Matrix multiply(const Matrix& left, const Matrix& right);

} // namespace linalg
