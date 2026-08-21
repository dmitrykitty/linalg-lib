#include <linalg/operations.hpp>

#include <stdexcept>

namespace linalg {

Matrix add_to_each_row(const Matrix& matrix, const Vector& values) {
    if (values.size() != matrix.cols()) {
        throw std::invalid_argument("row vector size must equal matrix column count");
    }

    Matrix result(matrix);
    for (Matrix::size_type row = 0; row < result.rows(); ++row) {
        for (Matrix::size_type col = 0; col < result.cols(); ++col) {
            result(row, col) += values[col];
        }
    }
    return result;
}

Matrix add_to_each_column(const Matrix& matrix, const Vector& values) {
    if (values.size() != matrix.rows()) {
        throw std::invalid_argument("column vector size must equal matrix row count");
    }

    Matrix result(matrix);
    for (Matrix::size_type row = 0; row < result.rows(); ++row) {
        for (Matrix::size_type col = 0; col < result.cols(); ++col) {
            result(row, col) += values[row];
        }
    }
    return result;
}

Matrix repeat_as_rows(const Vector& values, Matrix::size_type row_count) {
    Matrix result(row_count, values.size());
    for (Matrix::size_type row = 0; row < result.rows(); ++row) {
        for (Matrix::size_type col = 0; col < result.cols(); ++col) {
            result(row, col) = values[col];
        }
    }
    return result;
}

Matrix repeat_as_columns(const Vector& values, Matrix::size_type column_count) {
    Matrix result(values.size(), column_count);
    for (Matrix::size_type row = 0; row < result.rows(); ++row) {
        for (Matrix::size_type col = 0; col < result.cols(); ++col) {
            result(row, col) = values[row];
        }
    }
    return result;
}

Matrix transpose(const Matrix& other) {
    Matrix result(other.cols(), other.rows());
    if (other.empty()) {
        return result;
    }

    for (Matrix::size_type row = 0; row < other.rows(); ++row) {
        for (Matrix::size_type col = 0; col < other.cols(); ++col) {
            result(col, row) = other(row, col);
        }
    }
    return result;
}

long double trace(const Matrix& other) {
    if(other.empty() || other.cols() != other.rows()) {
        throw std::invalid_argument("matrix should be square and not empty"); 
    }

    const double* data = other.data();
    const Matrix::size_type cols = other.cols(); 
    const Matrix::size_type rows = other.rows();
    long double res = 0.0;

    for(Matrix::size_type i = 0 ; i < rows; i++) {
        res += data[cols * i + i];
    }

    return res;
}

} // namespace linalg
