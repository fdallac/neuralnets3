cmake_minimum_required(VERSION 3.10)
project(MatrixProject CXX)
set(CMAKE_CXX_STANDARD 17)

include_directories(include)

# --- OpenMP ---
find_package(OpenMP REQUIRED)

# --- OpenBLAS ---
find_library(OPENBLAS_LIB NAMES openblas REQUIRED)


add_library(
    matrix STATIC 
    src/matrix.cpp
    src/matmul.cpp
)

# ---- GoogleTest ----
add_subdirectory(third_party/googletest)

# ---- Test executable ----
add_executable(
    test_matrix_construct
    tests/unit/test_matrix_construct.cpp
)
add_executable(
    test_matrix_ops
    tests/unit/test_matrix_ops.cpp
)
add_executable(
    test_matrix_mult
    tests/unit/test_matrix_mult.cpp
)
add_executable(
    test_benchmark
    tests/benchmark/test_benchmark.cpp
)

target_link_libraries(
    matrix PUBLIC
    OpenMP::OpenMP_CXX
)

target_link_libraries(
    test_matrix_construct PRIVATE
    matrix
    gtest
    gtest_main
)
target_link_libraries(
    test_matrix_ops PRIVATE
    matrix
    gtest
    gtest_main
)
target_link_libraries(
    test_matrix_mult PRIVATE
    matrix
    gtest
    gtest_main
    OpenMP::OpenMP_CXX
)
target_link_libraries(
    test_benchmark PRIVATE
    matrix
    ${OPENBLAS_LIB}
)

enable_testing()
add_test(NAME MatrixConstructTest   COMMAND test_matrix_construct)
add_test(NAME MatrixOpsTest         COMMAND test_matrix_ops)
add_test(NAME MatrixMultTest        COMMAND test_matrix_mult)   