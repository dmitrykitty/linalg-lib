#pragma once

#include <linalg/matrix.hpp>
#include <linalg/vector.hpp>

namespace linalg {

enum class VectorNorm {
    l1,
    l2,
    infinity,
};

enum class MatrixNorm {
    one,
    frobenius,
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

} // namespace linalg
