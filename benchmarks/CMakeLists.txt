add_executable(
    linalg_benchmarks
    benchmark_main.cpp
    broadcast_benchmark.cpp
    gemm_comparison_benchmark.cpp
    smoke_benchmark.cpp
    trace_benchmark.cpp
    transpose_benchmark.cpp
    multiply_benchmark.cpp
)

target_link_libraries(linalg_benchmarks PRIVATE linalg::linalg benchmark::benchmark)
target_compile_features(linalg_benchmarks PRIVATE cxx_std_20)

linalg_set_project_warnings(linalg_benchmarks)
linalg_enable_sanitizers(linalg_benchmarks)

# OpenBLAS is optional and benchmark-only: the library itself never links a BLAS. Distribution
# packages (for example Ubuntu's libopenblas-dev) ship an `openblas` pkg-config module.
if(LINALG_BENCHMARK_OPENBLAS)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(LINALG_OPENBLAS QUIET IMPORTED_TARGET openblas)
    endif()
    if(LINALG_OPENBLAS_FOUND)
        message(STATUS "linalg: GEMM benchmarks include OpenBLAS ${LINALG_OPENBLAS_VERSION}")
        target_link_libraries(linalg_benchmarks PRIVATE PkgConfig::LINALG_OPENBLAS)
        target_compile_definitions(linalg_benchmarks PRIVATE LINALG_HAS_OPENBLAS=1)
    else()
        message(STATUS "linalg: OpenBLAS not found; GEMM benchmarks run without BLAS baselines")
    endif()
endif()
