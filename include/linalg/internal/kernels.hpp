#pragma once

#include <linalg/scalar.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>

namespace linalg::internal {

// The common row-major interface of the current owning matrix types.
template <typename M>
concept MatrixLike = requires(M& matrix, const M& read_only, std::size_t index) {
    typename M::value_type;
    requires FloatingScalar<typename M::value_type>;
    { read_only.rows() } -> std::same_as<std::size_t>;
    { read_only.cols() } -> std::same_as<std::size_t>;
    { read_only.size() } -> std::same_as<std::size_t>;
    { read_only.data() } -> std::same_as<const typename M::value_type*>;
    { matrix.data() } -> std::same_as<typename M::value_type*>;
    { read_only(index, index) } -> std::same_as<const typename M::value_type&>;
    { matrix(index, index) } -> std::same_as<typename M::value_type&>;
};

template <typename V>
concept VectorLike = requires(V& vector, const V& read_only, std::size_t index) {
    typename V::value_type;
    requires FloatingScalar<typename V::value_type>;
    { read_only.size() } -> std::same_as<std::size_t>;
    { read_only.data() } -> std::same_as<const typename V::value_type*>;
    { vector.data() } -> std::same_as<typename V::value_type*>;
    { read_only[index] } -> std::same_as<const typename V::value_type&>;
    { vector[index] } -> std::same_as<typename V::value_type&>;
};

// Internal kernels overwrite preallocated output. Compatible shapes and output
// storage distinct from the inputs are preconditions, asserted in Debug builds.
template <MatrixLike Input, MatrixLike Output>
    requires std::same_as<typename Input::value_type, typename Output::value_type>
constexpr void transpose_into(const Input& input, Output& output) noexcept {
    assert(output.rows() == input.cols() && output.cols() == input.rows());
    assert(output.size() == 0 || output.data() != input.data());
    if (input.size() == 0) {
        return;
    }

    // Destination-contiguous traversal: Experiment 1 in docs/performance.md.
    for (std::size_t col = 0; col < input.cols(); ++col) {
        for (std::size_t row = 0; row < input.rows(); ++row) {
            output(col, row) = input(row, col);
        }
    }
}

template <MatrixLike Left, MatrixLike Right, MatrixLike Output>
    requires std::same_as<typename Left::value_type, typename Right::value_type> &&
             std::same_as<typename Left::value_type, typename Output::value_type>
constexpr void matmul_into(const Left& left, const Right& right, Output& output) noexcept {
    assert(left.cols() == right.rows());
    assert(output.rows() == left.rows() && output.cols() == right.cols());
    assert(output.size() == 0 ||
           (output.data() != left.data() && output.data() != right.data()));
    using T = Output::value_type;
    for (std::size_t index = 0; index < output.size(); ++index) {
        output.data()[index] = T{};
    }
    if (output.size() == 0) {
        return;
    }

    // ikj keeps B reads and C updates contiguous in the inner j loop.
    // Default selected from Experiment 4; no unmeasured shape-dispatch heuristic.
    for (std::size_t row = 0; row < left.rows(); ++row) {
        for (std::size_t inner = 0; inner < left.cols(); ++inner) {
            const T left_value = left(row, inner);
            for (std::size_t col = 0; col < right.cols(); ++col) {
                output(row, col) += left_value * right(inner, col);
            }
        }
    }
}

template <MatrixLike Matrix, VectorLike Input, VectorLike Output>
    requires std::same_as<typename Matrix::value_type, typename Input::value_type> &&
             std::same_as<typename Matrix::value_type, typename Output::value_type>
constexpr void matvec_into(const Matrix& matrix, const Input& input, Output& output) noexcept {
    assert(matrix.cols() == input.size() && output.size() == matrix.rows());
    assert(output.size() == 0 ||
           (output.data() != input.data() && output.data() != matrix.data()));
    using T = Output::value_type;
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        T sum{};
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            sum += matrix(row, col) * input[col];
        }
        output[row] = sum;
    }
}

} // namespace linalg::internal
