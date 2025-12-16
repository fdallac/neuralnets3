cmake_minimum_required(VERSION 3.16)
project(neuralnets LANGUAGES CXX)

# -------------------------------------------------
# Global configuration
# -------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------------------------------------
# Dependencies
# -------------------------------------------------

# OpenMP
find_package(OpenMP REQUIRED)

# OpenBLAS (optional but expected)
find_library(OPENBLAS_LIB NAMES openblas REQUIRED)

# GoogleTest
add_subdirectory(third_party/googletest)

# -------------------------------------------------
# Header-only libraries
# -------------------------------------------------

# ---------- Matrix library ----------
add_library(matrix INTERFACE)

target_include_directories(matrix INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(matrix INTERFACE cxx_std_17)

target_compile_options(matrix INTERFACE
    -mavx512f
)

target_link_libraries(matrix INTERFACE
    OpenMP::OpenMP_CXX
    ${OPENBLAS_LIB}
)

target_sources(matrix INTERFACE
    include/matrix/matrix.hpp
    include/matrix/matmul.hpp
    include/matrix/iohelper.hpp
)

# ---------- Neural networks ----------
add_library(neuralnets INTERFACE)

target_include_directories(neuralnets INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(neuralnets INTERFACE cxx_std_17)

target_link_libraries(neuralnets INTERFACE
    matrix
)

target_sources(neuralnets INTERFACE
    include/neuralnets/neuralnets.hpp
    include/neuralnets/activation.hpp
    include/neuralnets/loss.hpp
    include/neuralnets/optimizer.hpp
    include/neuralnets/metrics.hpp
)

# ---------- Utilities ----------
add_library(utils INTERFACE)

target_include_directories(utils INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(utils INTERFACE cxx_std_17)

target_sources(utils INTERFACE
    include/utils/bench.hpp
)

# -------------------------------------------------
# Test executables
# -------------------------------------------------

# Unit tests
add_executable(test_matrix_construct
    tests/unit/test_matrix_construct.cpp
)

add_executable(test_matrix_ops
    tests/unit/test_matrix_ops.cpp
)

add_executable(test_matrix_mult
    tests/unit/test_matrix_mult.cpp
)

add_executable(test_neural_nets
    tests/unit/test_neural_nets.cpp
)

add_executable(test_io_helper
    tests/unit/test_io_helper.cpp
)

add_executable(test_optimizer
    tests/unit/test_optimizer.cpp
)

# Benchmarks
add_executable(test_benchmark
    tests/benchmark/test_benchmark.cpp
)

# End-to-end tests
add_executable(test_classification_neuralnet
    tests/test_classification_neuralnet.cpp
)

add_executable(test_regression_neuralnet
    tests/test_regression_neuralnet.cpp
)

# -------------------------------------------------
# Link targets
# -------------------------------------------------

# Unit tests
foreach(t
    test_matrix_construct
    test_matrix_ops
    test_matrix_mult
    test_neural_nets
    test_io_helper
    test_optimizer
)
    target_link_libraries(${t} PRIVATE
        matrix
        neuralnets
        gtest
        gtest_main
    )
endforeach()

# Neural net tests
target_link_libraries(test_classification_neuralnet PRIVATE
    neuralnets
)

target_link_libraries(test_regression_neuralnet PRIVATE
    neuralnets
)

# Benchmarks
target_link_libraries(test_benchmark PRIVATE
    matrix
    utils
)

# -------------------------------------------------
# Enable testing
# -------------------------------------------------
enable_testing()

add_test(NAME MatrixConstructTest      COMMAND test_matrix_construct)
add_test(NAME MatrixOpsTest            COMMAND test_matrix_ops)
add_test(NAME MatrixMultTest           COMMAND test_matrix_mult)
add_test(NAME IOHelperTest             COMMAND test_io_helper)
add_test(NAME NeuralNetsTest           COMMAND test_neural_nets)
add_test(NAME OptimizerTest            COMMAND test_optimizer)
