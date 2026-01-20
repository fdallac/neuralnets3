cmake_minimum_required(VERSION 3.18)
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

# CUDA (optional)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CUDA_ENABLED TRUE)
    message(STATUS "CUDA support enabled")
else()
    set(CUDA_ENABLED FALSE)
    message(STATUS "CUDA not found - matrix multiplication will not include GPU support")
endif()


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

# If CUDA is available, create a CUDA-enabled library
if(CUDA_ENABLED)
    add_library(matrix_cuda STATIC
        include/matrix/matmul_cuda.cu
        include/matrix/matmul_cuda.cuh
        include/matrix/matmul_cuda_openmp.cuh
    )
    
    target_include_directories(matrix_cuda PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    )
    
    set_target_properties(matrix_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "61;70;75;80;86"  # Common architectures: Pascal (61), Volta, Turing, Ampere
        POSITION_INDEPENDENT_CODE ON
    )
    
    # Enable OpenMP for CUDA+OpenMP hybrid implementation
    target_compile_options(matrix_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC,-fopenmp>
    )
    
    # Link CUDA runtime library and OpenMP
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(matrix_cuda PUBLIC CUDA::cudart CUDA::cublas OpenMP::OpenMP_CXX)
    
    # Link matrix library with CUDA support
    target_link_libraries(matrix INTERFACE matrix_cuda)
    target_compile_definitions(matrix INTERFACE CUDA_AVAILABLE)
    
    message(STATUS "CUDA matrix library configured for architectures: 60;70;75;80;86")
endif()

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
    include/neuralnets/preprocessing.hpp
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

add_executable(test_sgd_optimizer
    tests/unit/test_optimizer.cpp
)

add_executable(test_adam_optimizer
    tests/unit/test_adam_optimizer.cpp
)

# CUDA tests (only built if CUDA is available)
if(CUDA_ENABLED)
    add_executable(test_matrix_cuda_mult
        tests/unit/test_matrix_cuda_mult.cpp
    )
endif()

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

add_executable(test_multi_classification_neuralnet
    tests/test_multi_classification_neuralnet.cpp
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
    test_sgd_optimizer
    test_adam_optimizer
)
    target_link_libraries(${t} PRIVATE
        matrix
        neuralnets
        gtest
        gtest_main
    )
endforeach()

# CUDA tests
if(CUDA_ENABLED)
    target_link_libraries(test_matrix_cuda_mult PRIVATE
        matrix
        neuralnets
        gtest
        gtest_main
    )
endif()

# Neural net tests
target_link_libraries(test_classification_neuralnet PRIVATE
    neuralnets
)

target_link_libraries(test_regression_neuralnet PRIVATE
    neuralnets
)

target_link_libraries(test_multi_classification_neuralnet PRIVATE
    neuralnets
)

# Benchmarks
target_link_libraries(test_benchmark PRIVATE
    matrix
    utils
)

# -------------------------------------------------
# Python Bindings (Optional)
# -------------------------------------------------
option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)

if(BUILD_PYTHON_BINDINGS)
    # Find Python and pybind11
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG QUIET)
    
    if(NOT pybind11_FOUND)
        # Try to find pybind11 via Python
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(pybind11_DIR)
            find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})
        else()
            message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
        endif()
    endif()
    
    # Create the Python module
    pybind11_add_module(pynn3
        include/bindings/pynn3.cpp
    )
    
    # No need to set OUTPUT_NAME - module is already named 'pynn3'
    set_target_properties(pynn3 PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    )
    
    target_include_directories(pynn3 PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )
    
    target_link_libraries(pynn3 PRIVATE
        matrix
    )
    
    # Copy to bin for easy testing
    add_custom_command(TARGET pynn3 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:pynn3>
            ${CMAKE_BINARY_DIR}/bin/
        COMMENT "Copying pynn3 Python module to bin/"
    )
    
    message(STATUS "Python bindings enabled - module will be built as 'pynn3'")
endif()


# -------------------------------------------------
# Enable testing
# -------------------------------------------------
enable_testing()

add_test(NAME MatrixConstructTest      COMMAND test_matrix_construct)
add_test(NAME MatrixOpsTest            COMMAND test_matrix_ops)
add_test(NAME MatrixMultTest           COMMAND test_matrix_mult)
add_test(NAME IOHelperTest             COMMAND test_io_helper)
add_test(NAME NeuralNetsTest           COMMAND test_neural_nets)
add_test(NAME SGDOptimizerTest          COMMAND test_sgd_optimizer)
add_test(NAME AdamOptimizerTest         COMMAND test_adam_optimizer)
if(CUDA_ENABLED)
    add_test(NAME MatrixCudaMultTest       COMMAND test_matrix_cuda_mult)
endif()