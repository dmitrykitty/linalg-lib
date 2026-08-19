#pragma once

#include <linalg/matrix.hpp>
#include <linalg/vector.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace linalg::test {

namespace detail {

inline void validate_tolerances(
    double absolute_tolerance,
    double relative_tolerance) {
    if (!std::isfinite(absolute_tolerance) || absolute_tolerance < 0.0 ||
        !std::isfinite(relative_tolerance) || relative_tolerance < 0.0) {
        throw std::invalid_argument("tolerances must be finite and non-negative");
    }
}

inline bool almost_equal_unchecked(
    double actual,
    double expected,
    double absolute_tolerance,
    double relative_tolerance) {
    if (actual == expected) {
        return true;
    }

    if (!std::isfinite(actual) || !std::isfinite(expected)) {
        return false;
    }

    const double difference = std::abs(actual - expected);
    const double scale = std::max(std::abs(actual), std::abs(expected));
    return difference <= absolute_tolerance + relative_tolerance * scale;
}

} // namespace detail

inline bool almost_equal(
    double actual,
    double expected,
    double absolute_tolerance,
    double relative_tolerance) {
    detail::validate_tolerances(absolute_tolerance, relative_tolerance);
    return detail::almost_equal_unchecked(
        actual, expected, absolute_tolerance, relative_tolerance);
}

inline bool almost_equal(
    const Vector& actual,
    const Vector& expected,
    double absolute_tolerance,
    double relative_tolerance) {
    detail::validate_tolerances(absolute_tolerance, relative_tolerance);

    if (actual.size() != expected.size()) {
        return false;
    }

    for (Vector::size_type index = 0; index < actual.size(); ++index) {
        if (!detail::almost_equal_unchecked(
                actual[index],
                expected[index],
                absolute_tolerance,
                relative_tolerance)) {
            return false;
        }
    }
    return true;
}

inline bool almost_equal(
    const Matrix& actual,
    const Matrix& expected,
    double absolute_tolerance,
    double relative_tolerance) {
    detail::validate_tolerances(absolute_tolerance, relative_tolerance);

    if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
        return false;
    }

    for (Matrix::size_type index = 0; index < actual.size(); ++index) {
        if (!detail::almost_equal_unchecked(
                actual.data()[index],
                expected.data()[index],
                absolute_tolerance,
                relative_tolerance)) {
            return false;
        }
    }
    return true;
}

} // namespace linalg::test
