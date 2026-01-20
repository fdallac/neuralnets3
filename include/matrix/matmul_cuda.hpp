/**
 * @file matmul_cuda.hpp
 * @brief C++ interface declarations for CUDA matrix multiplication
 */

#pragma once

#include <cstddef>
#include <string>

namespace cuda_matmul {

/**
 * @brief Check if CUDA is available on this system
 * @return true if a CUDA-capable device is available
 */
bool isCudaAvailable();

/**
 * @brief Get information about the current CUDA device
 * @return String with device name and compute capability
 */
std::string getCudaDeviceInfo();

/**
 * @brief Host wrapper for CUDA matrix multiplication (float)
 * @param h_A Host pointer to matrix A (M x K)
 * @param h_B Host pointer to matrix B (K x N)
 * @param h_C Host pointer to result matrix C (M x N)
 * @param M Number of rows in A
 * @param N Number of columns in B
 * @param K Inner dimension (cols of A, rows of B)
 */
void matmul_cuda_wrapper_float(const float* h_A, const float* h_B, float* h_C,
                                size_t M, size_t N, size_t K);

/**
 * @brief Host wrapper for CUDA matrix multiplication (double)
 * @param h_A Host pointer to matrix A (M x K)
 * @param h_B Host pointer to matrix B (K x N)
 * @param h_C Host pointer to result matrix C (M x N)
 * @param M Number of rows in A
 * @param N Number of columns in B
 * @param K Inner dimension (cols of A, rows of B)
 */
void matmul_cuda_wrapper_double(const double* h_A, const double* h_B, double* h_C,
                                 size_t M, size_t N, size_t K);

/**
 * @brief Template wrapper that dispatches to type-specific implementations
 * @tparam T Data type (float or double)
 * 
 * This is a convenience template that calls the appropriate type-specific
 * wrapper function. Only float and double are supported.
 */
template<typename T>
inline void matmul_cuda_wrapper(const T* h_A, const T* h_B, T* h_C,
                                size_t M, size_t N, size_t K) {
    // Default implementation - will cause linker error for unsupported types
    static_assert(sizeof(T) == 0, "matmul_cuda_wrapper only supports float and double");
}

// Explicit specialization declarations (defined in matmul_cuda.cu)
template<>
inline void matmul_cuda_wrapper<float>(const float* h_A, const float* h_B, float* h_C,
                                       size_t M, size_t N, size_t K) {
    matmul_cuda_wrapper_float(h_A, h_B, h_C, M, N, K);
}

template<>
inline void matmul_cuda_wrapper<double>(const double* h_A, const double* h_B, double* h_C,
                                        size_t M, size_t N, size_t K) {
    matmul_cuda_wrapper_double(h_A, h_B, h_C, M, N, K);
}

// ============================================================================
// CUDA + OpenMP Hybrid Interface
// ============================================================================

/**
 * @brief Host wrapper for hybrid CUDA+OpenMP matrix multiplication (float)
 * @param h_A Host pointer to matrix A (M x K)
 * @param h_B Host pointer to matrix B (K x N)
 * @param h_C Host pointer to result matrix C (M x N)
 * @param M Number of rows in A
 * @param N Number of columns in B
 * @param K Inner dimension (cols of A, rows of B)
 * 
 * Uses OpenMP for task-level parallelism (managing CUDA streams)
 * and CUDA for fine-grained GPU computation.
 */
void matmul_cuda_openmp_wrapper_float(const float* h_A, const float* h_B, float* h_C,
                                       size_t M, size_t N, size_t K);

/**
 * @brief Host wrapper for hybrid CUDA+OpenMP matrix multiplication (double)
 * @param h_A Host pointer to matrix A (M x K)
 * @param h_B Host pointer to matrix B (K x N)
 * @param h_C Host pointer to result matrix C (M x N)
 * @param M Number of rows in A
 * @param N Number of columns in B
 * @param K Inner dimension (cols of A, rows of B)
 */
void matmul_cuda_openmp_wrapper_double(const double* h_A, const double* h_B, double* h_C,
                                        size_t M, size_t N, size_t K);

/**
 * @brief Template wrapper for hybrid CUDA+OpenMP matrix multiplication
 * @tparam T Data type (float or double)
 * 
 * This convenience template dispatches to the appropriate type-specific
 * wrapper function. Only float and double are supported.
 */
template<typename T>
inline void matmul_cuda_openmp_wrapper(const T* h_A, const T* h_B, T* h_C,
                                        size_t M, size_t N, size_t K) {
    static_assert(sizeof(T) == 0, "matmul_cuda_openmp_wrapper only supports float and double");
}

template<>
inline void matmul_cuda_openmp_wrapper<float>(const float* h_A, const float* h_B, float* h_C,
                                               size_t M, size_t N, size_t K) {
    matmul_cuda_openmp_wrapper_float(h_A, h_B, h_C, M, N, K);
}

template<>
inline void matmul_cuda_openmp_wrapper<double>(const double* h_A, const double* h_B, double* h_C,
                                                size_t M, size_t N, size_t K) {
    matmul_cuda_openmp_wrapper_double(h_A, h_B, h_C, M, N, K);
}

}
