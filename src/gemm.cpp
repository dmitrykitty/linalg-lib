#include <linalg/internal/gemm.hpp>

#include "thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
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

The values come from the cache sizes; sweeps of KC and MC confirmed them for double (Experiment 8
in docs/performance.md). The float values follow the same byte budget and were not swept.
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

constexpr size_type ceil_div(size_type value, size_type divisor) noexcept {
    return (value + divisor - 1) / divisor;
}

constexpr size_type round_up(size_type value, size_type multiple) noexcept {
    return ceil_div(value, multiple) * multiple;
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

// C is only touched after the k loop, and for large matrices its tile rows (one per row, ldc
// apart) usually miss all caches. Requesting them before the loop lets the ~kc * 6 cycles of FMAs
// hide the memory latency instead of stalling on it at the end. A row of nr elements may straddle
// two cache lines when C is not 64-byte aligned, so both ends are prefetched.
template <typename T>
[[gnu::always_inline]] inline void prefetch_c_tile(const T* c, size_type ldc) noexcept {
#pragma GCC unroll 6
    for (size_type row = 0; row < Blocking<T>::mr; ++row) {
        _mm_prefetch(reinterpret_cast<const char*>(c + row * ldc), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(c + row * ldc + Blocking<T>::nr - 1),
                     _MM_HINT_T0);
    }
}

// How many k steps ahead the micro-kernels prefetch the packed A and B micro-panels. A packed
// micro-panel spans several 4 KiB pages, and the L2 hardware prefetcher stops at page boundaries,
// so without help the kernel stalls on demand misses at every page crossing. Measured 8..24 as a
// flat optimum (double, i7-10750H); 12 had the best geometric mean. See notes/gemm-theory.md.
constexpr size_type prefetch_distance = 12;

// 6 x 8 double tile: each k step broadcasts one A value per row and issues two FMAs against
// the two 4-wide halves of the B row.
[[gnu::target("avx2,fma")]] void microkernel_avx2_fma(
    size_type kc, const double* a, const double* b, double* c, size_type ldc) noexcept {
    constexpr size_type mr = Blocking<double>::mr;
    static_assert(Blocking<double>::nr == 8);

    prefetch_c_tile(c, ldc);

    __m256d accumulators[mr][2];
#pragma GCC unroll 6
    for (size_type row = 0; row < mr; ++row) {
        accumulators[row][0] = _mm256_setzero_pd();
        accumulators[row][1] = _mm256_setzero_pd();
    }

    for (size_type inner = 0; inner < kc; ++inner) {
        // One B row (8 doubles) is exactly one cache line per k step. Prefetching past the end of
        // a panel is harmless: prefetches never fault.
        _mm_prefetch(reinterpret_cast<const char*>(b + 8 * prefetch_distance), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a + mr * prefetch_distance), _MM_HINT_T0);
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

    prefetch_c_tile(c, ldc);

    __m256 accumulators[mr][2];
#pragma GCC unroll 6
    for (size_type row = 0; row < mr; ++row) {
        accumulators[row][0] = _mm256_setzero_ps();
        accumulators[row][1] = _mm256_setzero_ps();
    }

    for (size_type inner = 0; inner < kc; ++inner) {
        // One B row (16 floats) is one cache line per k step, as in the double kernel.
        _mm_prefetch(reinterpret_cast<const char*>(b + 16 * prefetch_distance), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a + mr * prefetch_distance), _MM_HINT_T0);
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

// The AVX2/FMA kernel for T, or nullptr when it was not compiled in or the CPU lacks AVX2/FMA.
template <typename T>
MicroKernel<T> avx2_fma_kernel() noexcept {
#if LINALG_HAS_AVX2_FMA_KERNEL
    if (cpu_supports_avx2_fma()) {
        return microkernel_avx2_fma;
    }
#endif
    return nullptr;
}

template <typename T>
MicroKernel<T> select_kernel(GemmKernel kernel) {
    if (kernel == GemmKernel::portable) {
        return microkernel_portable<T>;
    }
    if (const MicroKernel<T> fast = avx2_fma_kernel<T>()) {
        return fast;
    }
    if (kernel == GemmKernel::automatic) {
        return microkernel_portable<T>;
    }
    throw std::invalid_argument("the AVX2/FMA GEMM kernel is not available on this CPU");
}

struct Range {
    size_type begin;
    size_type end;
};

// Splits `count` items into `parts` contiguous ranges whose sizes differ by at most one.
constexpr Range split_evenly(size_type count, size_type parts, size_type part) noexcept {
    const size_type base = count / parts;
    const size_type extra = count % parts;
    const size_type begin = part * base + std::min(part, extra);
    return {begin, begin + base + (part < extra ? 1 : 0)};
}

// The compute work of one (jc, pc) step is cut into chunks of one mc-row block of A times a range
// of nr-column panels of B. Threads take chunks dynamically from a shared counter, so a thread that
// runs slower (preempted, sharing a core, lower clock) simply finishes fewer chunks instead of
// holding everyone else at the next barrier.
struct ChunkLayout {
    size_type row_blocks;
    size_type panels_per_chunk;
    size_type count;
};

// Chunks per thread trade load balance (more chunks: a shorter tail while others wait at the barrier)
// against per-chunk overhead (re-packing the A block, grabbing the counter, cold B micro-panels).
// Measured on the i7-10750H (double, 6 threads, 256..2048 cubed): 4 beat 8 and 16 by 7% in geometric
// mean and by 23% at 256^3, with 2048^3 within noise. See notes/gemm-theory.md.
constexpr size_type chunks_per_thread = 4;

ChunkLayout chunk_layout(size_type row_blocks, size_type col_panels, size_type threads) noexcept {
    // One thread gets one column chunk per row block: exactly the serial loop nest.
    const size_type target = threads == 1 ? 1 : chunks_per_thread * threads;
    const size_type col_chunks = std::clamp<size_type>(ceil_div(target, row_blocks), 1, col_panels);
    const size_type panels_per_chunk = ceil_div(col_panels, col_chunks);
    return {row_blocks, panels_per_chunk, row_blocks * ceil_div(col_panels, panels_per_chunk)};
}

template <typename T>
struct GemmArguments {
    size_type m;
    size_type n;
    size_type k;
    const T* a;
    size_type lda;
    const T* b;
    size_type ldb;
    T* c;
    size_type ldc;
};

/*
One blocked GEMM executed by `threads` cooperating threads (one thread is the serial case).

For every (jc, pc) panel the threads first pack disjoint slices of the shared B panel and wait at a
barrier. Then they repeatedly take the next chunk (an mc-row block times a range of nr-column
panels, see ChunkLayout) from an atomic counter, pack that A block into a private buffer, and
compute its tiles of C. Consecutive chunks walk down the row blocks first, so threads working at
the same time read the same B micro-panels from the shared L3.

Every element of C is still accumulated by one micro-kernel call per pc, in pc order, so the result
is bitwise identical for any thread count.
*/
template <typename T>
class BlockedGemm {
public:
    BlockedGemm(const GemmArguments<T>& arguments, MicroKernel<T> kernel, size_type threads)
        : arguments_(arguments), kernel_(kernel), threads_(threads),
          kc_max_(std::min(B::kc, arguments.k)),
          // Round each private slice to whole cache lines so threads never share one.
          a_slice_(round_up(round_up(std::min(B::mc, arguments.m), B::mr) * kc_max_,
                            64 / sizeof(T))),
          packed_a_(allocate_aligned<T>(a_slice_ * threads)),
          packed_b_(allocate_aligned<T>(round_up(std::min(B::nc, arguments.n), B::nr) * kc_max_)),
          sync_(static_cast<std::ptrdiff_t>(threads)) {}

    void operator()(size_type thread) noexcept {
        const GemmArguments<T>& args = arguments_;
        const size_type row_blocks = ceil_div(args.m, B::mc);
        T* const packed_a = packed_a_.get() + thread * a_slice_;
        T* const packed_b = packed_b_.get();

        for (size_type jc = 0; jc < args.n; jc += B::nc) {
            const size_type nc = std::min(B::nc, args.n - jc);
            const size_type col_panel_count = ceil_div(nc, B::nr);
            const ChunkLayout chunks = chunk_layout(row_blocks, col_panel_count, threads_);

            for (size_type pc = 0; pc < args.k; pc += B::kc) {
                const size_type kc = std::min(B::kc, args.k - pc);

                // Nobody may still be reading the previous B panel when it is overwritten, and
                // every thread has left the previous chunk loop, so the counter can be reset.
                sync_.arrive_and_wait();
                if (thread == 0) {
                    next_chunk_.store(0, std::memory_order_relaxed);
                }
                const Range share = split_evenly(col_panel_count, threads_, thread);
                if (share.begin < share.end) {
                    const size_type first_col = share.begin * B::nr;
                    pack_b(kc, std::min(share.end * B::nr, nc) - first_col,
                           args.b + pc * args.ldb + jc + first_col, args.ldb,
                           packed_b + first_col * kc);
                }
                // B is complete and the counter reset is visible to every thread.
                sync_.arrive_and_wait();

                // Relaxed is enough: the counter only hands out unique chunk numbers; all data
                // dependencies are ordered by the barriers.
                for (size_type chunk = next_chunk_.fetch_add(1, std::memory_order_relaxed);
                     chunk < chunks.count;
                     chunk = next_chunk_.fetch_add(1, std::memory_order_relaxed)) {
                    const size_type ic = (chunk % chunks.row_blocks) * B::mc;
                    const size_type mc = std::min(B::mc, args.m - ic);
                    const size_type col_begin =
                        (chunk / chunks.row_blocks) * chunks.panels_per_chunk * B::nr;
                    const size_type col_end =
                        std::min(col_begin + chunks.panels_per_chunk * B::nr, nc);
                    T* const c_block = args.c + ic * args.ldc + jc;

                    // Micro-kernels accumulate into C, so a chunk's C is zeroed on its first pc.
                    if (pc == 0) {
                        for (size_type row = 0; row < mc; ++row) {
                            std::fill(c_block + row * args.ldc + col_begin,
                                      c_block + row * args.ldc + col_end, T{});
                        }
                    }
                    pack_a(mc, kc, args.a + ic * args.lda + pc, args.lda, packed_a);
                    compute_block(mc, kc, packed_a, packed_b, col_begin, col_end, c_block);
                }
            }
        }
    }

private:
    using B = Blocking<T>;

    // Multiplies one packed A block by packed B micro-panels [col_begin, col_end) into C.
    void compute_block(size_type mc, size_type kc, const T* packed_a, const T* packed_b,
                       size_type col_begin, size_type col_end, T* c) const noexcept {
        const size_type ldc = arguments_.ldc;
        alignas(64) T edge_tile[B::mr * B::nr];

        // jr outside ir: one B micro-panel stays in L1 while A micro-panels stream from L2.
        for (size_type jr = col_begin; jr < col_end; jr += B::nr) {
            const size_type nr = std::min(B::nr, col_end - jr);
            const T* b_panel = packed_b + jr * kc;
            for (size_type ir = 0; ir < mc; ir += B::mr) {
                const size_type mr = std::min(B::mr, mc - ir);
                const T* a_panel = packed_a + ir * kc;
                T* c_tile = c + ir * ldc + jr;

                if (mr == B::mr && nr == B::nr) {
                    kernel_(kc, a_panel, b_panel, c_tile, ldc);
                    continue;
                }
                // Partial tile: compute the full zero-padded tile, then add the valid part.
                std::fill_n(edge_tile, B::mr * B::nr, T{});
                kernel_(kc, a_panel, b_panel, edge_tile, B::nr);
                for (size_type row = 0; row < mr; ++row) {
                    for (size_type col = 0; col < nr; ++col) {
                        c_tile[row * ldc + col] += edge_tile[row * B::nr + col];
                    }
                }
            }
        }
    }

    GemmArguments<T> arguments_;
    MicroKernel<T> kernel_;
    size_type threads_;
    size_type kc_max_;
    size_type a_slice_;
    AlignedBuffer<T> packed_a_;
    AlignedBuffer<T> packed_b_;
    std::barrier<> sync_;
    // On its own cache line: every chunk grab writes it, and it must not invalidate the line that
    // holds the read-mostly fields above.
    alignas(64) std::atomic<size_type> next_chunk_{0};
};

// Below this many floating-point operations (2mnk) the automatic thread count stays at one:
// waking the pool and the per-panel barriers cost more than the parallel speedup. Measured with
// BM_MatMulBlockedAvx2FmaThreads (double, 6 threads vs 1, i7-10750H): serial still wins at 96^3,
// parallel wins from 128^3. Re-measure under the controlled protocol before tuning further.
constexpr double parallel_flop_threshold = 2.0 * 128.0 * 128.0 * 128.0;

size_type resolve_thread_count(size_type requested, size_type m, size_type n, size_type k) {
    if (requested != 0) {
        return requested;
    }
    const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                         static_cast<double>(k);
    return flops < parallel_flop_threshold ? 1 : gemm_default_thread_count();
}

template <typename T>
void gemm_blocked(const GemmArguments<T>& args, GemmKernel kernel_choice,
                  size_type thread_count) {
    assert(args.lda >= args.k && args.ldb >= args.n && args.ldc >= args.n);
    using B = Blocking<T>;
    const MicroKernel<T> kernel = select_kernel<T>(kernel_choice);

    if (args.m == 0 || args.n == 0) {
        return;
    }
    assert(args.c != args.a && args.c != args.b);
    if (args.k == 0) {
        for (size_type row = 0; row < args.m; ++row) {
            std::fill_n(args.c + row * args.ldc, args.n, T{});
        }
        return;
    }

    // More threads than micro-tiles in the first panel would only add idle barrier participants.
    const size_type tiles =
        ceil_div(args.m, B::mr) * ceil_div(std::min(B::nc, args.n), B::nr);
    size_type threads =
        std::min(resolve_thread_count(thread_count, args.m, args.n, args.k), tiles);
    if (threads > 1) {
        ThreadPool& pool = shared_thread_pool();
        threads = std::min(threads, pool.max_threads());
        BlockedGemm<T> parallel(args, kernel, threads);
        if (pool.try_run(threads, parallel)) {
            return;
        }
        // Another thread owns the pool (e.g. concurrent multiply calls): run serially instead.
    }
    BlockedGemm<T> serial(args, kernel, 1);
    serial(0);
}

} // namespace

bool gemm_kernel_available(GemmKernel kernel) noexcept {
    return kernel != GemmKernel::avx2_fma || cpu_supports_avx2_fma();
}

size_type gemm_default_thread_count() {
    static const size_type count = physical_core_count();
    return count;
}

void gemm(size_type m, size_type n, size_type k, const double* a, size_type lda, const double* b,
          size_type ldb, double* c, size_type ldc, GemmKernel kernel, size_type thread_count) {
    gemm_blocked(GemmArguments{m, n, k, a, lda, b, ldb, c, ldc}, kernel, thread_count);
}

void gemm(size_type m, size_type n, size_type k, const float* a, size_type lda, const float* b,
          size_type ldb, float* c, size_type ldc, GemmKernel kernel, size_type thread_count) {
    gemm_blocked(GemmArguments{m, n, k, a, lda, b, ldb, c, ldc}, kernel, thread_count);
}

} // namespace linalg::internal
