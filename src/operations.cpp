#include <linalg/operations.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>


//===============================PRIVATE HELPERS ===========================================
namespace {

using size_type = linalg::Matrix::size_type;
using Matrix = linalg::Matrix;
using Vector = linalg::Vector;

/**
 * It is possible to have a matrix or vector consisting of numbers close to DOUBLE_MAX
 * (let's call it just DM).
 *
 * So sqrt(DM^2 + DM^2 + ... + DM^2) is a problem because DM^2 itself can
 * overflow the double limit, even if the final norm could still be representable.
 *
 * What we do instead:
 *     - let's find the maximum absolute value among all numbers; assume it is MAX,
 *       which will be our scale
 *     - let's take MAX out of the square root:
 *           MAX * sqrt(SCALED_1^2 + SCALED_2^2 + ... + SCALED_N^2)
 *       where each SCALED_i = abs(Aij) / MAX and therefore belongs to [0, 1]
 *     - so our final formula is:
 *           scale * sqrt(sum((Aij / scale)^2))
 *       or equivalently:
 *           sqrt(scale^2 * sum((Aij / scale)^2))
 *
 * So we could make two passes: the first one to find MAX and the second one
 * to calculate the scaled sum. But we want to finish it in just one pass.
 *
 * During the pass, `scale` is the largest absolute value seen so far, and
 * `sum_of_squares` stores the sum relative to this scale.
 *
 * We maintain the following invariant:
 *           real_sum_of_squares = scale^2 * sum_of_squares
 * 
 * If the next magnitude is smaller than or equal to the current scale, nothing
 * changes about the scale. We just add its scaled contribution:
 *           sum_of_squares += (magnitude / scale)^2
 *
 * If the next magnitude is larger than the current scale, we have found a new
 * scale. All previously accumulated values were expressed relative to the old
 * scale, so we have to rescale them:
 *
 *           ratio = old_scale / new_scale
 *
 *           sum_of_squares =
 *               1 + old_sum_of_squares * ratio^2 or (new_scale^2 + old_sum_of_squares * old_scale^2) / new scale^2
 *
 * So 1 represents the new maximum itself because:
 *           new_scale / new_scale = 1
 *
 * After processing all values, we restore the real norm:
 *           scale * sqrt(sum_of_squares)
 *
 * The important part is that we always divide the smaller value by the larger
 * one before squaring, so the ratio stays in [0, 1] and we avoid overflowing
 * because of squaring very large input values.
 */
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

Vector normalize(const Vector& vector) {
    double scale = 0.0;
    for (size_type index = 0; index < vector.size(); ++index) {
        const double magnitude = std::abs(vector[index]);
        if (!std::isfinite(magnitude)) {
            throw std::invalid_argument(
                "cannot normalize a vector with non-finite elements");
        }
        scale = std::max(scale, magnitude);
    }

    if (scale == 0.0) {
        throw std::invalid_argument("cannot normalize a zero vector");
    }

    long double scaled_sum_of_squares = 0.0L;
    for (size_type index = 0; index < vector.size(); ++index) {
        const long double scaled_value = static_cast<long double>(vector[index]) / scale;
        scaled_sum_of_squares += scaled_value * scaled_value;
    }

    const long double scaled_length = std::sqrt(scaled_sum_of_squares);
    Vector result(vector.size());
    for (size_type index = 0; index < vector.size(); ++index) {
        result[index] = static_cast<double>((static_cast<long double>(vector[index]) / scale) / scaled_length);
    }
    return result;
}

Vector multiply(const Matrix& matrix, const Vector& vector) {
    if (matrix.cols() != vector.size()) {
        throw std::invalid_argument("vector size must equal matrix column count");
    }

    Vector result(matrix.rows(), 0.0);
    for (size_type row = 0; row < matrix.rows(); ++row) {
        for (size_type col = 0; col < matrix.cols(); ++col) {
            result[row] += matrix(row, col) * vector[col];
        }
    }

    return result;
}

} // namespace linalg
