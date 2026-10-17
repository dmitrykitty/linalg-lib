#include <linalg/internal/gemm.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>

// The AVX2/FMA micro-kernel uses per-function target attributes, so the rest of the library is
// still compiled for the baseline ISA and the kernel is only called after a runtime CPU check.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define LINALG_HAS_AVX2_FMA_KERNEL 1
#include <immintrin.h>
#else
#define LINALG_HAS_AVX2_FMA_KERNEL 0
#endif

namespace linalg::internal {
namespace {

using size_type = std::size_t;

/*
Blocking parameters (i7-10750H: 32 KiB L1d, 256 KiB L2 per core, 12 MiB shared L3).

mr x nr     register tile of C held in accumulators by the micro-kernel;
kc x nr     packed B micro-panel, reused for every A micro-panel -> L1 (256 * 8 * 8 B = 16 KiB);
mc x kc     packed A block, reused for every B micro-panel      -> L2 (72 * 256 * 8 B = 144 KiB);
kc x nc     packed B panel, reused for every A block            -> L3 (256 * 4080 * 8 B ~ 8 MiB).

These are starting values from the cache sizes, not yet tuned by measurement.
*/
template <typename T>
struct Blocking;

template <>
struct Blocking<double> {
    // 6 x 8 doubles = 12 ymm accumulators + 2 B loads + 1 A broadcast <= 16 ymm registers.
    static constexpr size_type mr = 6;
    static constexpr size_type nr = 8;
    static constexpr size_type kc = 256;
    static constexpr size_type mc = 72;
    static constexpr size_type nc = 4080;
};

template <>
struct Blocking<float> {
    // Same register budget as double: 8 floats per ymm register.
    static constexpr size_type mr = 6;
    static constexpr size_type nr = 16;
    static constexpr size_type kc = 256;
    static constexpr size_type mc = 144;
    static constexpr size_type nc = 4080;
};

template <typename T>
constexpr bool blocking_is_consistent() {
    using B = Blocking<T>;
    return B::mc % B::mr == 0 && B::nc % B::nr == 0 && (B::nr * sizeof(T)) % 32 == 0;
}
static_assert(blocking_is_consistent<double>() && blocking_is_consistent<float>());

// Packed micro-panels start at multiples of nr elements, so 64-byte buffers keep every B
// micro-panel aligned for vector loads and on a cache-line boundary.
constexpr std::align_val_t buffer_alignment{64};

struct AlignedDelete {
    void operator()(void* pointer) const noexcept {
        ::operator delete(pointer, buffer_alignment);
    }
};

template <typename T>
using AlignedBuffer = std::unique_ptr<T[], AlignedDelete>;

template <typename T>
AlignedBuffer<T> allocate_aligned(size_type count) {
    return AlignedBuffer<T>(static_cast<T*>(::operator new(count * sizeof(T), buffer_alignment)));
}

constexpr size_type round_up(size_type value, size_type multiple) noexcept {
    return (value + multiple - 1) / multiple * multiple;
}

// Copies an mc x kc block of A into consecutive mr-row micro-panels. Inside a micro-panel the
// layout is column-major (mr values per k step), the order the micro-kernel reads it in.
// Missing rows of the last micro-panel are zero-filled.
template <typename T>
void pack_a(size_type mc, size_type kc, const T* a, size_type lda, T* packed) noexcept {
    constexpr size_type mr = Blocking<T>::mr;
    for (size_type panel_row = 0; panel_row < mc; panel_row += mr) {
        const size_type rows = std::min(mr, mc - panel_row);
        const T* panel = a + panel_row * lda;
        for (size_type inner = 0; inner < kc; ++inner) {
            for (size_type row = 0; row < rows; ++row) {
                packed[row] = panel[row * lda + inner];
            }
            for (size_type row = rows; row < mr; ++row) {
                packed[row] = T{};
            }
            packed += mr;
        }
    }
}

// Copies a kc x nc panel of B into consecutive nr-column micro-panels, each stored row-major
// (nr values per k step). Missing columns of the last micro-panel are zero-filled.
template <typename T>
void pack_b(size_type kc, size_type nc, const T* b, size_type ldb, T* packed) noexcept {
    constexpr size_type nr = Blocking<T>::nr;
    for (size_type panel_col = 0; panel_col < nc; panel_col += nr) {
        const size_type cols = std::min(nr, nc - panel_col);
        for (size_type inner = 0; inner < kc; ++inner) {
            const T* row = b + inner * ldb + panel_col;
            for (size_type col = 0; col < cols; ++col) {
                packed[col] = row[col];
            }
            for (size_type col = cols; col < nr; ++col) {
                packed[col] = T{};
            }
            packed += nr;
        }
    }
}

// A micro-kernel computes C[mr x nr] += packed_a[mr x kc] * packed_b[kc x nr], where C has row
// stride ldc. Edge tiles are handled by the driver, so kernels always see full tiles.
template <typename T>
using MicroKernel = void (*)(size_type kc, const T* a, const T* b, T* c, size_type ldc) noexcept;

template <typename T>
void microkernel_portable(size_type kc, const T* a, const T* b, T* c, size_type ldc) noexcept {
    constexpr size_type mr = Blocking<T>::mr;
    constexpr size_type nr = Blocking<T>::nr;
    T accumulators[mr][nr] = {};
    for (size_type inner = 0; inner < kc; ++inner) {
        for (size_type row = 0; row < mr; ++row) {
            const T a_value = a[row];
            for (size_type col = 0; col < nr; ++col) {
                accumulators[row][col] += a_value * b[col];
            }
        }
        a += mr;
        b += nr;
    }
    for (size_type row = 0; row < mr; ++row) {
        for (size_type col = 0; col < nr; ++col) {
            c[row * ldc + col] += accumulators[row][col];
        }
    }
}

#if LINALG_HAS_AVX2_FMA_KERNEL

bool cpu_supports_avx2_fma() noexcept {
    static const bool supported =
        __builtin_cpu_supports("avx2") != 0 && __builtin_cpu_supports("fma") != 0;
    return supported;
}

// 6 x 8 double tile: each k step broadcasts one A value per row and issues two FMAs against
// the two 4-wide halves of the B row.
[[gnu::target("avx2,fma")]] void microkernel_avx2_fma(
    size_type kc, const double* a, const double* b, double* c, size_type ldc) noexcept {
    constexpr size_type mr = Blocking<double>::mr;
    static_assert(Blocking<double>::nr == 8);

    __m256d accumulators[mr][2];
#pragma GCC unroll 6
    for (size_type row = 0; row < mr; ++row) {
        accumulators[row][0] = _mm256_setzero_pd();
        accumulators[row][1] = _mm256_setzero_pd();
    }

    for (size_type inner = 0; inner < kc; ++inner) {
        const __m256d b_low = _mm256_load_pd(b);
        const __m256d b_high = _mm256_load_pd(b + 4);
#pragma GCC unroll 6
        for (size_type row = 0; row < mr; ++row) {
            const __m256d a_value = _mm256_broadcast_sd(a + row);
            accumulators[row][0] = _mm256_fmadd_pd(a_value, b_low, accumulators[row][0]);
            accumulators[row][1] = _mm256_fmadd_pd(a_value, b_high, accumulators[row][1]);
        }
        a += mr;
        b += 8;
    }

#pragma GCC unroll 6
    for (size_type row = 0; row < mr; ++row) {
        double* c_row = c + row * ldc;
        _mm256_storeu_pd(c_row, _mm256_add_pd(_mm256_loadu_pd(c_row), accumulators[row][0]));
        _mm256_storeu_pd(c_row + 4,
                         _mm256_add_pd(_mm256_loadu_pd(c_row + 4), accumulators[row][1]));
    }
}

// 6 x 16 float tile with the same register budget as the double kernel.
[[gnu::target("avx2,fma")]] void microkernel_avx2_fma(
    size_type kc, const float* a, const float* b, float* c, size_type ldc) noexcept {
    constexpr size_type mr = Blocking<float>::mr;
    static_assert(Blocking<float>::nr == 16);

    __m256 accumulators[mr][2];
#pragma GCC unroll 6
    for (size_type row = 0; row < mr; ++row) {
        accumulators[row][0] = _mm256_setzero_ps();
        accumulators[row][1] = _mm256_setzero_ps();
    }

    for (size_type inner = 0; inner < kc; ++inner) {
        const __m256 b_low = _mm256_load_ps(b);
        const __m256 b_high = _mm256_load_ps(b + 8);
#pragma GCC unroll 6
        for (size_type row = 0; row < mr; ++row) {
            const __m256 a_value = _mm256_broadcast_ss(a + row);
            accumulators[row][0] = _mm256_fmadd_ps(a_value, b_low, accumulators[row][0]);
            accumulators[row][1] = _mm256_fmadd_ps(a_value, b_high, accumulators[row][1]);
        }
        a += mr;
        b += 16;
    }

#pragma GCC unroll 6
    for (size_type row = 0; row < mr; ++row) {
        float* c_row = c + row * ldc;
        _mm256_storeu_ps(c_row, _mm256_add_ps(_mm256_loadu_ps(c_row), accumulators[row][0]));
        _mm256_storeu_ps(c_row + 8,
                         _mm256_add_ps(_mm256_loadu_ps(c_row + 8), accumulators[row][1]));
    }
}

#else

bool cpu_supports_avx2_fma() noexcept {
    return false;
}

#endif

template <typename T>
MicroKernel<T> select_kernel(GemmKernel kernel) {
    switch (kernel) {
    case GemmKernel::automatic:
#if LINALG_HAS_AVX2_FMA_KERNEL
        if (cpu_supports_avx2_fma()) {
            return microkernel_avx2_fma;
        }
#endif
        return microkernel_portable<T>;
    case GemmKernel::portable:
        return microkernel_portable<T>;
    case GemmKernel::avx2_fma:
#if LINALG_HAS_AVX2_FMA_KERNEL
        if (cpu_supports_avx2_fma()) {
            return microkernel_avx2_fma;
        }
#endif
        throw std::invalid_argument("the AVX2/FMA GEMM kernel is not available on this CPU");
    }
    throw std::invalid_argument("unknown GEMM kernel");
}

template <typename T>
void gemm_blocked(size_type m, size_type n, size_type k, const T* a, size_type lda, const T* b,
                  size_type ldb, T* c, size_type ldc, GemmKernel kernel_choice) {
    assert(lda >= k && ldb >= n && ldc >= n);
    using B = Blocking<T>;
    const MicroKernel<T> kernel = select_kernel<T>(kernel_choice);

    if (m == 0 || n == 0) {
        return;
    }
    assert(c != a && c != b);

    // Micro-kernels accumulate into C, so C starts from zero.
    for (size_type row = 0; row < m; ++row) {
        std::fill_n(c + row * ldc, n, T{});
    }
    if (k == 0) {
        return;
    }

    // Buffers are sized for the largest block this call will pack, not the global maxima.
    const size_type kc_max = std::min(B::kc, k);
    const AlignedBuffer<T> packed_a =
        allocate_aligned<T>(round_up(std::min(B::mc, m), B::mr) * kc_max);
    const AlignedBuffer<T> packed_b =
        allocate_aligned<T>(round_up(std::min(B::nc, n), B::nr) * kc_max);
    alignas(64) T edge_tile[B::mr * B::nr];

    for (size_type jc = 0; jc < n; jc += B::nc) {
        const size_type nc = std::min(B::nc, n - jc);
        for (size_type pc = 0; pc < k; pc += B::kc) {
            const size_type kc = std::min(B::kc, k - pc);
            pack_b(kc, nc, b + pc * ldb + jc, ldb, packed_b.get());

            for (size_type ic = 0; ic < m; ic += B::mc) {
                const size_type mc = std::min(B::mc, m - ic);
                pack_a(mc, kc, a + ic * lda + pc, lda, packed_a.get());

                // jr outside ir: one B micro-panel stays in L1 while A micro-panels stream from L2.
                for (size_type jr = 0; jr < nc; jr += B::nr) {
                    const size_type nr = std::min(B::nr, nc - jr);
                    const T* b_panel = packed_b.get() + jr * kc;
                    for (size_type ir = 0; ir < mc; ir += B::mr) {
                        const size_type mr = std::min(B::mr, mc - ir);
                        const T* a_panel = packed_a.get() + ir * kc;
                        T* c_tile = c + (ic + ir) * ldc + jc + jr;

                        if (mr == B::mr && nr == B::nr) {
                            kernel(kc, a_panel, b_panel, c_tile, ldc);
                            continue;
                        }
                        // Partial tile: compute the full zero-padded tile, then add the valid part.
                        std::fill_n(edge_tile, B::mr * B::nr, T{});
                        kernel(kc, a_panel, b_panel, edge_tile, B::nr);
                        for (size_type row = 0; row < mr; ++row) {
                            for (size_type col = 0; col < nr; ++col) {
                                c_tile[row * ldc + col] += edge_tile[row * B::nr + col];
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace

bool gemm_kernel_available(GemmKernel kernel) noexcept {
    return kernel != GemmKernel::avx2_fma || cpu_supports_avx2_fma();
}

void gemm(size_type m, size_type n, size_type k, const double* a, size_type lda, const double* b,
          size_type ldb, double* c, size_type ldc, GemmKernel kernel) {
    gemm_blocked(m, n, k, a, lda, b, ldb, c, ldc, kernel);
}

void gemm(size_type m, size_type n, size_type k, const float* a, size_type lda, const float* b,
          size_type ldb, float* c, size_type ldc, GemmKernel kernel) {
    gemm_blocked(m, n, k, a, lda, b, ldb, c, ldc, kernel);
}

} // namespace linalg::internal
