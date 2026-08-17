add_executable(
    linalg_benchmarks
    operations_benchmark.cpp
    smoke_benchmark.cpp
)

target_link_libraries(linalg_benchmarks PRIVATE linalg::linalg benchmark::benchmark)
target_compile_features(linalg_benchmarks PRIVATE cxx_std_20)

linalg_set_project_warnings(linalg_benchmarks)
linalg_enable_sanitizers(linalg_benchmarks)

