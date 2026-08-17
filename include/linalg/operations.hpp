#pragma once

#include <linalg/matrix.hpp>
#include <linalg/vector.hpp>

namespace linalg {

Matrix add_to_each_row(const Matrix& matrix, const Vector& values);
Matrix add_to_each_column(const Matrix& matrix, const Vector& values);

Matrix repeat_as_rows(const Vector& values, Matrix::size_type row_count);
Matrix repeat_as_columns(const Vector& values, Matrix::size_type column_count);

} // namespace linalg
