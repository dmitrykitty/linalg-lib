include(FetchContent)

function(linalg_setup_googletest)
    find_package(GTest CONFIG QUIET)
    if(TARGET GTest::gtest_main)
        return()
    endif()

    if(NOT LINALG_FETCH_DEPENDENCIES)
        message(FATAL_ERROR "GoogleTest was not found and dependency fetching is disabled")
    endif()

    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(googletest)
endfunction()

function(linalg_setup_google_benchmark)
    find_package(benchmark CONFIG QUIET)
    if(TARGET benchmark::benchmark_main)
        return()
    endif()

    if(NOT LINALG_FETCH_DEPENDENCIES)
        message(FATAL_ERROR "Google Benchmark was not found and dependency fetching is disabled")
    endif()

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        google_benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.5
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(google_benchmark)
endfunction()

