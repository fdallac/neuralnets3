/**
 * @file matmul_cuda.cuh
 * @brief CUDA implementation of matrix multiplication
 * 
 * This file provides GPU-accelerated matrix multiplication using CUDA.
 * The implementation uses a tiled approach with shared memory for efficiency.
 * 
 * @note Requires NVIDIA CUDA Toolkit and compatible GPU
 * @note Compile with nvcc and link with CUDA runtime
 */

#pragma once

#ifdef __CUDACC__  // Only compile with NVCC

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cuda_matmul {

/**
 * @brief Tile size for shared memory blocking
 * 
 * This value determines the size of tiles loaded into shared memory.
 * Typical values: 16 or 32. Must match thread block dimensions.
 * 
 * Trade-offs:
 * - Larger tiles: Better data reuse, but more shared memory usage
 * - Smaller tiles: Less shared memory, but more global memory accesses
 * 
 * 16x16 = 256 threads per block (good occupancy on most GPUs)
 */
constexpr int TILE_SIZE = 16;


/**
 * @brief Check CUDA errors and throw exception on failure
 * @param err CUDA error code
 * @param msg Context message for the error
 * @throws std::runtime_error if err != cudaSuccess
 */
inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}


/**
 * @brief Check if CUDA is available on this system
 * @return true if a CUDA-capable device is available
 */
inline bool isCudaAvailableImpl() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
}


/**
 * @brief Get information about the current CUDA device
 * @return String with device name and compute capability
 */
inline std::string getCudaDeviceInfoImpl() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return std::string(prop.name) + " (Compute " + 
           std::to_string(prop.major) + "." + std::to_string(prop.minor) + ")";
}


/**
 * @brief CUDA kernel for tiled matrix multiplication
 * 
 * Computes C = A * B using shared memory tiling.
 * Each thread computes one element of the result matrix C.
 * 
 * @tparam T Data type (float or double)
 * @param A Input matrix A (M x K), row-major in global memory
 * @param B Input matrix B (K x N), row-major in global memory
 * @param C Output matrix C (M x N), row-major in global memory
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A / rows in B (inner dimension)
 */
template<typename T>
__global__ void matmul_tiled_kernel(const T* __restrict__ A, 
                                     const T* __restrict__ B, 
                                     T* __restrict__ C,
                                     int M, int N, int K) {
    
    // Shared memory tiles for A and B
    // Each tile is TILE_SIZE x TILE_SIZE elements
    __shared__ T tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ T tile_B[TILE_SIZE][TILE_SIZE];
    
    // Calculate global row and column indices for this thread
    // Each thread computes one element C[row][col]
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Accumulator for the dot product
    T sum = T(0);
    
    // Number of tiles needed to cover the K dimension
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Iterate over all tiles
    for (int t = 0; t < num_tiles; ++t) {
        
        // Load tile from A into shared memory
        // Each thread loads one element
        int a_col = t * TILE_SIZE + threadIdx.x;  // Column in A
        if (row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = T(0);  // Zero-padding for boundary
        }
        
        // Load tile from B into shared memory
        // Each thread loads one element
        int b_row = t * TILE_SIZE + threadIdx.y;  // Row in B
        if (b_row < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = T(0);  // Zero-padding for boundary
        }
        
        // Synchronize to ensure all threads have loaded their elements
        // This barrier is essential for correctness!
        __syncthreads();
        
        // Compute partial dot product for this tile
        // Each thread accumulates TILE_SIZE products
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        // Synchronize before loading next tile
        // Prevents overwriting shared memory while others are still reading
        __syncthreads();
    }
    
    // Write result to global memory
    // Only threads within matrix bounds write their result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


/**
 * @brief Host wrapper for CUDA matrix multiplication
 * 
 * Handles memory allocation, data transfer, and kernel launch.
 * 
 * @tparam T Data type (float or double)
 * @param h_A Host pointer to matrix A (M x K)
 * @param h_B Host pointer to matrix B (K x N)
 * @param h_C Host pointer to result matrix C (M x N)
 * @param M Number of rows in A
 * @param N Number of columns in B
 * @param K Inner dimension (cols of A, rows of B)
 * @throws std::runtime_error on CUDA errors
 * 
 * @note This function is named differently from the public API in matmul_cuda.hpp
 *       to avoid ODR issues when linking CUDA and non-CUDA compilation units.
 */
template<typename T>
void matmul_cuda_impl(const T* h_A, const T* h_B, T* h_C, 
                      size_t M, size_t N, size_t K) {
    
    // Validate input pointers
    if (h_A == nullptr || h_B == nullptr || h_C == nullptr) {
        throw std::runtime_error("Null pointer passed to CUDA matrix multiplication");
    }
    
    // Validate dimensions
    if (M == 0 || N == 0 || K == 0) {
        throw std::runtime_error("Zero dimension passed to CUDA matrix multiplication");
    }
    
    // Calculate sizes in bytes
    size_t size_A = M * K * sizeof(T);
    size_t size_B = K * N * sizeof(T);
    size_t size_C = M * N * sizeof(T);
    
    // Device pointers
    T *d_A = nullptr;
    T *d_B = nullptr;
    T *d_C = nullptr;
    
    // Allocate device memory
    checkCudaError(cudaMalloc(&d_A, size_A), "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc(&d_B, size_B), "Failed to allocate device memory for B");
    checkCudaError(cudaMalloc(&d_C, size_C), "Failed to allocate device memory for C");
    
    // Initialize C to zero on device
    checkCudaError(cudaMemset(d_C, 0, size_C), "Failed to initialize device memory for C");
    
    // Copy input matrices from host to device
    checkCudaError(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice), 
                   "Failed to copy matrix A to device");
    checkCudaError(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice), 
                   "Failed to copy matrix B to device");
    
    // Configure kernel launch parameters
    // Grid dimensions: enough blocks to cover the entire output matrix
    // Block dimensions: TILE_SIZE x TILE_SIZE threads
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,  // Blocks in x (columns)
        (M + TILE_SIZE - 1) / TILE_SIZE   // Blocks in y (rows)
    );
    
    // Launch the kernel
    matmul_tiled_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, 
                                                   static_cast<int>(M), 
                                                   static_cast<int>(N), 
                                                   static_cast<int>(K));
    
    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    // Wait for kernel to complete
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");
    
    // Copy result from device to host
    checkCudaError(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost), 
                   "Failed to copy matrix C to host");
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

} // namespace cuda_matmul

#endif // __CUDACC__
