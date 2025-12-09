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

add_library(
    neuralnets STATIC
    src/neuralnets.cpp
    src/activation.cpp
    src/loss.cpp
)

# ---- Enable AVX-512 ----
target_compile_options(matrix PUBLIC -mavx512f)

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
    test_neural_nets
    tests/unit/test_neural_nets.cpp
)
add_executable(
    test_benchmark
    tests/benchmark/test_benchmark.cpp
)
add_executable(
    test_run_nn_train
    tests/unit/test_run_nn_train.cpp
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
target_link_libraries(
    test_neural_nets PRIVATE
    matrix
    gtest 
    gtest_main
)   
target_link_libraries(
    test_run_nn_train PRIVATE
    matrix
    neuralnets
)

enable_testing()
add_test(NAME MatrixConstructTest   COMMAND test_matrix_construct)
add_test(NAME MatrixOpsTest         COMMAND test_matrix_ops)
add_test(NAME MatrixMultTest        COMMAND test_matrix_mult)   
add_test(NAME NeuralNetsTest        COMMAND test_neural_nets)