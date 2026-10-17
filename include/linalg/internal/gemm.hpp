#pragma once

#include <cstddef>

namespace linalg::internal {

// Selects the register micro-kernel used inside the blocked GEMM driver.
enum class GemmKernel {
    /// Fastest kernel supported by the running CPU.
    automatic,
    /// Plain C++ micro-kernel; always available.
    portable,
    /// x86-64 AVX2 + FMA micro-kernel; requires runtime CPU support.
    avx2_fma,
};

/// Reports whether `kernel` can run on this CPU. `automatic` and `portable` are always available.
bool gemm_kernel_available(GemmKernel kernel) noexcept;

// Row-major C = A * B with cache blocking and packing (GotoBLAS/BLIS loop structure).
// A is m x k with row stride lda, B is k x n with row stride ldb, and C is m x n with row
// stride ldc. C is overwritten; elements beyond its n columns are left untouched.
// Preconditions (asserted in Debug builds): lda >= k, ldb >= n, ldc >= n, and C does not
// overlap A or B. Throws std::invalid_argument if `kernel` is not available, and
// std::bad_alloc if the packing buffers cannot be allocated.
void gemm(std::size_t m, std::size_t n, std::size_t k, const double* a, std::size_t lda,
          const double* b, std::size_t ldb, double* c, std::size_t ldc,
          GemmKernel kernel = GemmKernel::automatic);

void gemm(std::size_t m, std::size_t n, std::size_t k, const float* a, std::size_t lda,
          const float* b, std::size_t ldb, float* c, std::size_t ldc,
          GemmKernel kernel = GemmKernel::automatic);

} // namespace linalg::internal
