/**
 * @file matmul_cuda.cu
 * @brief CUDA compilation unit - provides linkable symbols for C++ code
 * 
 * This minimal file exists because CMake requires a .cu extension to compile with nvcc.
 * All implementation logic is in matmul_cuda.cuh.
 */

#include "matrix/matmul_cuda.cuh"
#include "matrix/matmul_cuda_openmp.cuh"

namespace cuda_matmul {

// Non-template entry points callable from regular C++ code.
// These simply forward to the implementations in matmul_cuda.cuh.

bool isCudaAvailable() { return isCudaAvailableImpl(); }
std::string getCudaDeviceInfo() { return getCudaDeviceInfoImpl(); }

void matmul_cuda_wrapper_float(const float* A, const float* B, float* C,
                                size_t M, size_t N, size_t K) {
    matmul_cuda_impl<float>(A, B, C, M, N, K);
}

void matmul_cuda_wrapper_double(const double* A, const double* B, double* C,
                                 size_t M, size_t N, size_t K) {
    matmul_cuda_impl<double>(A, B, C, M, N, K);
}

// CUDA+OpenMP hybrid wrappers
void matmul_cuda_openmp_wrapper_float(const float* A, const float* B, float* C,
                                       size_t M, size_t N, size_t K) {
    cuda_openmp_matmul::matmul_cuda_openmp_impl<float>(A, B, C, M, N, K);
}

void matmul_cuda_openmp_wrapper_double(const double* A, const double* B, double* C,
                                        size_t M, size_t N, size_t K) {
    cuda_openmp_matmul::matmul_cuda_openmp_impl<double>(A, B, C, M, N, K);
}

} // namespace cuda_matmul
