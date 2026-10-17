include(GoogleTest)

add_executable(
    linalg_tests
    approximate_test.cpp
    gemm_test.cpp
    matrix_test.cpp
    operations_test.cpp
    shared_operations_test.cpp
    static_matrix_test.cpp
    static_vector_test.cpp
    vector_test.cpp
)

target_link_libraries(linalg_tests PRIVATE linalg::linalg GTest::gtest_main)
target_compile_features(linalg_tests PRIVATE cxx_std_20)

linalg_set_project_warnings(linalg_tests)
linalg_enable_sanitizers(linalg_tests)

gtest_discover_tests(linalg_tests)

