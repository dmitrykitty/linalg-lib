cmake_minimum_required(VERSION 3.20)

project(
    linear_algebra_lib
    VERSION 0.1.0
    DESCRIPTION "A learning-focused high-performance numerical linear algebra library"
    LANGUAGES CXX
)

include(CTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CompilerWarnings)
include(Dependencies)
include(Sanitizers)

option(LINALG_BUILD_TESTS "Build the linalg test suite" ${PROJECT_IS_TOP_LEVEL})
option(LINALG_BUILD_BENCHMARKS "Build the linalg benchmark suite" ${PROJECT_IS_TOP_LEVEL})
option(LINALG_BUILD_EXAMPLES "Build the linalg examples" ${PROJECT_IS_TOP_LEVEL})
option(LINALG_FETCH_DEPENDENCIES "Fetch missing development dependencies" ON)
option(LINALG_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)
option(LINALG_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(LINALG_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(linalg STATIC)
add_library(linalg::linalg ALIAS linalg)

target_sources(
    linalg
    PRIVATE
        src/matrix.cpp
        src/operations.cpp
        src/vector.cpp
    PUBLIC
        FILE_SET public_headers
        TYPE HEADERS
        BASE_DIRS include
        FILES
            include/linalg/linalg.hpp
            include/linalg/matrix.hpp
            include/linalg/operations.hpp
            include/linalg/vector.hpp
)

target_compile_features(linalg PUBLIC cxx_std_20)

target_include_directories(
    linalg
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

linalg_set_project_warnings(linalg)
linalg_enable_sanitizers(linalg)

if(BUILD_TESTING AND LINALG_BUILD_TESTS)
    linalg_setup_googletest()
    add_subdirectory(tests)
endif()

if(LINALG_BUILD_BENCHMARKS)
    linalg_setup_google_benchmark()
    add_subdirectory(benchmarks)
endif()

if(LINALG_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

