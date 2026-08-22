#include <linalg/operations.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>


//===============================PRIVATE HELPERS ===========================================
namespace {

using size_type = linalg::Matrix::size_type;
using Matrix = linalg::Matrix;
using Vector = linalg::Vector;

double scaled_euclidean_norm(const double* values, std::size_t size) {
    double scale = 0.0;
    double sum_of_squares = 1.0;

    for (std::size_t index = 0; index < size; ++index) {
        const double magnitude = std::abs(values[index]);
        if (!std::isfinite(magnitude)) {
            return magnitude;
        }
        if (magnitude == 0.0) {
            continue;
        }

        if (scale < magnitude) {
            const double ratio = scale / magnitude;
            sum_of_squares = 1.0 + sum_of_squares * ratio * ratio;
            scale = magnitude;
        } else {
            const double ratio = magnitude / scale;
            sum_of_squares += ratio * ratio;
        }
    }

    return scale == 0.0 ? 0.0 : scale * std::sqrt(sum_of_squares);
}

double frobenius_norm(const Matrix& matrix) {
    return scaled_euclidean_norm(matrix.data(), matrix.size());
}

double one_norm(const Matrix& matrix) {
    long double largest_sum = 0.0L;
    for (size_type col = 0; col < matrix.cols(); ++col) {
        long double column_sum = 0.0L;
        for (size_type row = 0; row < matrix.rows(); ++row) {
            const double magnitude = std::abs(matrix(row, col));
            if (std::isnan(magnitude)) {
                return magnitude;
            }
            column_sum += magnitude;
        }
        largest_sum = std::max(largest_sum, column_sum);
    }
    return static_cast<double>(largest_sum);
}

double infinity_norm(const Matrix& matrix) {
    long double largest_sum = 0.0L;
    for (size_type row = 0; row < matrix.rows(); ++row) {
        long double row_sum = 0.0L;
        for (size_type col = 0; col < matrix.cols(); ++col) {
            const double magnitude = std::abs(matrix(row, col));
            if (std::isnan(magnitude)) {
                return magnitude;
            }
            row_sum += magnitude;
        }
        largest_sum = std::max(largest_sum, row_sum);
    }
    return static_cast<double>(largest_sum);
}

double l1_norm(const Vector& vector) {
    long double result = 0.0L;
    for (size_type index = 0; index < vector.size(); ++index) {
        result += std::abs(static_cast<long double>(vector[index]));
    }
    return static_cast<double>(result);
}

double l2_norm(const Vector& vector) {
    return scaled_euclidean_norm(vector.data(), vector.size());
}

double infinity_norm(const Vector& vector) {
    double result = 0.0;
    for (size_type index = 0; index < vector.size(); ++index) {
        const double magnitude = std::abs(vector[index]);
        if (std::isnan(magnitude)) {
            return magnitude;
        }
        result = std::max(result, magnitude);
    }
    return result;
}

} // namespace

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

double trace(const Matrix& other) {
    if (other.empty() || other.cols() != other.rows()) {
        throw std::invalid_argument("matrix must be square and non-empty");
    }

    const double* data = other.data();
    const Matrix::size_type diagonal_stride = other.cols() + 1;
    long double result = 0.0L;

    for (Matrix::size_type index = 0; index < other.rows(); ++index) {
        result += data[index * diagonal_stride];
    }

    return static_cast<double>(result);
}

double norm(const Vector& vector, VectorNorm type) {
    switch (type) {
    case VectorNorm::l1:
        return l1_norm(vector);
    case VectorNorm::l2:
        return l2_norm(vector);
    case VectorNorm::infinity:
        return infinity_norm(vector);
    }
    throw std::invalid_argument("unknown vector norm type");
}

double norm(const Matrix& matrix, MatrixNorm type) {
    switch (type) {
    case MatrixNorm::one:
        return one_norm(matrix);
    case MatrixNorm::frobenius:
        return frobenius_norm(matrix);
    case MatrixNorm::infinity:
        return infinity_norm(matrix);
    }
    throw std::invalid_argument("unknown matrix norm type");
}

} // namespace linalg
