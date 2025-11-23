cmake_minimum_required(VERSION 3.10)
project(MatrixProject CXX)
set(CMAKE_CXX_STANDARD 17)

include_directories(include)

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
    tests/test_matrix_construct.cpp
)
add_executable(
    test_matrix_ops
    tests/test_matrix_ops.cpp
)
add_executable(
    test_matrix_mult
    tests/test_matrix_mult.cpp
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
)

enable_testing()
add_test(NAME MatrixConstructTest COMMAND test_matrix_construct)
add_test(NAME MatrixOpsTest COMMAND test_matrix_ops)
add_test(NAME MatrixMultTest COMMAND test_matrix_mult)