/**
 * @file matmul_cuda_openmp.cuh
 * @brief Hybrid CUDA + OpenMP matrix multiplication implementation
 * 
 * This file implements a heterogeneous computing approach for matrix multiplication
 * that leverages both GPU (CUDA) and multi-core CPU (OpenMP) capabilities.
 * 
 * ## Design Philosophy
 * 
 * 1. **Register Blocking**: Each thread computes a TM×TN micro-tile, performing
 *    64 multiplications but only 16 shared memory loads per K-tile iteration.
 *    This drastically reduces memory controller pressure.
 * 
 * 2. **Block-Based Decomposition**: Matrix A is divided into horizontal strips.
 *    Each strip is processed by a separate CUDA stream.
 * 
 * 3. **OpenMP for Task Parallelism**: OpenMP threads manage multiple CUDA streams,
 *    enabling concurrent kernel execution and memory transfers.
 * 
 * 4. **PCIe Overlap**: Using cudaMemcpyAsync within the OpenMP loop allows the
 *    driver to schedule results from Block N to be copied back while the GPU
 *    is already computing Block N+1.
 * 
 */

#pragma once

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

namespace cuda_openmp_matmul {


/**
 * @brief Tile dimensions for register blocking optimization
 * 
 * BM × BN = output tile size per thread block
 * BK = depth of shared memory tile (K dimension)
 * TM × TN = output tile size per thread (register blocking)
 * 
 * Thread count = (BM/TM) × (BN/TN) = 16 × 16 = 256 threads
 * Each thread computes TM × TN = 64 output elements
 */
constexpr int BM = 128;  ///< Block tile size in M dimension
constexpr int BN = 128;  ///< Block tile size in N dimension  
constexpr int BK = 8;    ///< Block tile size in K dimension (shared memory depth)
constexpr int TM = 8;    ///< Thread tile size in M (each thread computes 8 rows)
constexpr int TN = 8;    ///< Thread tile size in N (each thread computes 8 cols)

/**
 * @brief Number of threads per block
 * With BM=128, BN=128, TM=8, TN=8: (128/8) × (128/8) = 256 threads
 */
constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

/**
 * @brief Stride for loading shared memory tiles
 * sA[BM][BK] = 128×8 = 1024 elements, 256 threads → 4 loads per thread
 * sB[BK][BN] = 8×128 = 1024 elements, 256 threads → 4 loads per thread
 */
constexpr int STRIDE_A = NUM_THREADS / BK;   ///< 256/8 = 32 rows per load iteration
constexpr int STRIDE_B = NUM_THREADS / BN;   ///< 256/128 = 2 rows per load iteration

/**
 * @brief Block size for high-level stream decomposition (row strips)
 */
constexpr int BLOCK_SIZE = 1024;

/**
 * @brief Maximum number of concurrent CUDA streams
 */
constexpr int MAX_STREAMS = 8;


// Error Handling
inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}


/**
 * @brief Matrix multiplication kernel with register blocking
 * 
 * This kernel implements a 2D register-blocked matrix multiplication:
 * - Each thread block computes a BM×BN tile of C
 * - Each thread computes a TM×TN micro-tile using registers
 * - Shared memory tiles reduce global memory bandwidth
 * 
 * Memory hierarchy:
 * 1. Global → Shared: Coalesced loads of BM×BK (A) and BK×BN (B)
 * 2. Shared → Registers: Each thread loads TM + TN elements
 * 3. Registers: Compute TM×TN outer product
 * 
 * @tparam T Data type (float or double)
 */
template<typename T>
__global__ void matmul_optimized_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc)
{
    // Shared memory tiles with padding to avoid bank conflicts
    __shared__ T sA[BM][BK];
    __shared__ T sB[BK][BN];
    
    // Thread's micro-tile accumulator in registers (TM × TN = 64 values)
    T threadResults[TM][TN] = {};
    
    // Registers for the current K-iteration
    T regA[TM];
    T regB[TN];
    
    // Block coordinates
    const int bx = blockIdx.x;  // Column block
    const int by = blockIdx.y;  // Row block
    
    // Thread coordinates for computing (16×16 thread arrangement)
    const int threadCol = threadIdx.x % (BN / TN);  // 0-15 (column within block)
    const int threadRow = threadIdx.x / (BN / TN);  // 0-15 (row within block)
    
    // Thread coordinates for loading sA (coalesced along K dimension)
    const int loadA_row = threadIdx.x / BK;   // 0-31 (row within sA)
    const int loadA_col = threadIdx.x % BK;   // 0-7  (column within sA)
    
    // Thread coordinates for loading sB (coalesced along N dimension)  
    const int loadB_row = threadIdx.x / BN;   // 0-1  (row within sB)
    const int loadB_col = threadIdx.x % BN;   // 0-127 (column within sB)
    
    // Base pointers for this block
    const int A_row_base = by * BM;
    const int B_col_base = bx * BN;
    
    // Main loop over K dimension in BK-sized tiles
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        
        // Load shared memory tiles (coalesced global memory access)
        // Load sA[BM][BK] - need BM/STRIDE_A = 128/32 = 4 iterations
        #pragma unroll
        for (int offset = 0; offset < BM; offset += STRIDE_A) {
            int row = loadA_row + offset;
            int globalRow = A_row_base + row;
            int globalCol = k_tile + loadA_col;
            
            if (globalRow < M && globalCol < K) {
                sA[row][loadA_col] = A[globalRow * lda + globalCol];
            } else {
                sA[row][loadA_col] = T(0);
            }
        }
        
        // Load sB[BK][BN] - need BK/STRIDE_B = 8/2 = 4 iterations
        #pragma unroll
        for (int offset = 0; offset < BK; offset += STRIDE_B) {
            int row = loadB_row + offset;
            int globalRow = k_tile + row;
            int globalCol = B_col_base + loadB_col;
            
            if (globalRow < K && globalCol < N) {
                sB[row][loadB_col] = B[globalRow * ldb + globalCol];
            } else {
                sB[row][loadB_col] = T(0);
            }
        }
        
        __syncthreads();
        
        // Compute micro-tile using register blocking
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load column of A from shared memory to registers
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = sA[threadRow * TM + i][dotIdx];
            }
            
            // Load row of B from shared memory to registers
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = sB[dotIdx][threadCol * TN + j];
            }
            
            // Compute outer product and accumulate (TM × TN = 64 FMAs)
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results from registers to global memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int globalRow = A_row_base + threadRow * TM + i;
        
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int globalCol = B_col_base + threadCol * TN + j;
            
            if (globalRow < M && globalCol < N) {
                C[globalRow * ldc + globalCol] = threadResults[i][j];
            }
        }
    }
}


constexpr int SIMPLE_TILE = 16;

template<typename T>
__global__ void matmul_simple_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc)
{
    __shared__ T tileA[SIMPLE_TILE][SIMPLE_TILE];
    __shared__ T tileB[SIMPLE_TILE][SIMPLE_TILE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * SIMPLE_TILE + ty;
    int col = blockIdx.x * SIMPLE_TILE + tx;
    
    T sum = T(0);
    
    for (int t = 0; t < (K + SIMPLE_TILE - 1) / SIMPLE_TILE; ++t) {
        int aCol = t * SIMPLE_TILE + tx;
        int bRow = t * SIMPLE_TILE + ty;
        
        tileA[ty][tx] = (row < M && aCol < K) ? A[row * lda + aCol] : T(0);
        tileB[ty][tx] = (bRow < K && col < N) ? B[bRow * ldb + col] : T(0);
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < SIMPLE_TILE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * ldc + col] = sum;
    }
}


/**
 * @brief Matrix multiplication using CUDA streams and OpenMP
 * 
 * Key features:
 * 1. Register-blocked kernel for high arithmetic intensity
 * 2. Multiple CUDA streams for PCIe overlap
 * 3. OpenMP for stream management
 * 4. Asynchronous memory transfers
 * 
 * @tparam T Data type (float or double)
 */
template<typename T>
void matmul_cuda_openmp_streams(
    const T* h_A, const T* h_B, T* h_C,
    size_t M, size_t N, size_t K)
{
    // Determine parallelism
    int num_threads = omp_get_max_threads();
    int num_streams = std::min(num_threads, MAX_STREAMS);
    size_t num_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Use simple kernel for small matrices
    bool use_simple_kernel = (M < 128 || N < 128 || K < 128);
    
    // Allocate and copy matrix B (shared across all streams)
    T* d_B;
    size_t size_B = K * N * sizeof(T);
    checkCudaError(cudaMalloc(&d_B, size_B), "B allocation failed");
    checkCudaError(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice), "B copy failed");
    
    // Create streams and per-stream buffers
    std::vector<cudaStream_t> streams(num_streams);
    std::vector<T*> d_A_blocks(num_streams);
    std::vector<T*> d_C_blocks(num_streams);
    
    size_t size_A_block = BLOCK_SIZE * K * sizeof(T);
    size_t size_C_block = BLOCK_SIZE * N * sizeof(T);
    
    for (int i = 0; i < num_streams; ++i) {
        checkCudaError(cudaStreamCreate(&streams[i]), "Stream creation failed");
        checkCudaError(cudaMalloc(&d_A_blocks[i], size_A_block), "A block allocation failed");
        checkCudaError(cudaMalloc(&d_C_blocks[i], size_C_block), "C block allocation failed");
    }
    
    // Process blocks in parallel using OpenMP
    #pragma omp parallel num_threads(num_streams)
    {
        int tid = omp_get_thread_num();
        cudaStream_t stream = streams[tid];
        T* d_A = d_A_blocks[tid];
        T* d_C = d_C_blocks[tid];
        
        #pragma omp for schedule(dynamic)
        for (size_t b = 0; b < num_blocks; ++b) {
            size_t row_start = b * BLOCK_SIZE;
            size_t row_count = std::min(static_cast<size_t>(BLOCK_SIZE), M - row_start);
            
            if (row_count == 0) continue;
            
            // Async H2D transfer of A block
            cudaMemcpyAsync(d_A, h_A + row_start * K, 
                           row_count * K * sizeof(T),
                           cudaMemcpyHostToDevice, stream);
            
            // Launch kernel
            if (use_simple_kernel) {
                dim3 blockDim(SIMPLE_TILE, SIMPLE_TILE);
                dim3 gridDim(
                    (static_cast<unsigned int>(N) + SIMPLE_TILE - 1) / SIMPLE_TILE,
                    (static_cast<unsigned int>(row_count) + SIMPLE_TILE - 1) / SIMPLE_TILE
                );
                matmul_simple_kernel<T><<<gridDim, blockDim, 0, stream>>>(
                    d_A, d_B, d_C,
                    static_cast<int>(row_count), static_cast<int>(N), static_cast<int>(K),
                    static_cast<int>(K), static_cast<int>(N), static_cast<int>(N)
                );
            } else {
                // Optimized kernel: 256 threads, each computing 8×8 output
                dim3 blockDim(NUM_THREADS);
                dim3 gridDim(
                    (static_cast<unsigned int>(N) + BN - 1) / BN,
                    (static_cast<unsigned int>(row_count) + BM - 1) / BM
                );
                matmul_optimized_kernel<T><<<gridDim, blockDim, 0, stream>>>(
                    d_A, d_B, d_C,
                    static_cast<int>(row_count), static_cast<int>(N), static_cast<int>(K),
                    static_cast<int>(K), static_cast<int>(N), static_cast<int>(N)
                );
            }
            
            // Async D2H transfer of C block (overlaps with next block's H2D + compute)
            cudaMemcpyAsync(h_C + row_start * N, d_C,
                           row_count * N * sizeof(T),
                           cudaMemcpyDeviceToHost, stream);
            
            // Sync before reusing this stream's buffers
            cudaStreamSynchronize(stream);
        }
    }
    
    // Final sync and cleanup
    cudaDeviceSynchronize();
    
    for (int i = 0; i < num_streams; ++i) {
        cudaFree(d_A_blocks[i]);
        cudaFree(d_C_blocks[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_B);
}



/**
 * @brief Hybrid CUDA+OpenMP matrix multiplication
 * 
 * Key features:
 * - Small matrices (<128): Uses simple 16×16 tiled kernel
 * - Large matrices: Uses register-blocked kernel with 8×8 output per thread
 * - Multiple streams for PCIe overlap
 * 
 * @tparam T Data type (float or double)
 */
template<typename T>
void matmul_cuda_openmp_impl(
    const T* h_A, const T* h_B, T* h_C,
    size_t M, size_t N, size_t K)
{
    if (h_A == nullptr || h_B == nullptr || h_C == nullptr) {
        throw std::runtime_error("Null pointer passed to CUDA+OpenMP matrix multiplication");
    }
    
    if (M == 0 || N == 0 || K == 0) {
        throw std::runtime_error("Zero dimension passed to CUDA+OpenMP matrix multiplication");
    }
    
    matmul_cuda_openmp_streams<T>(h_A, h_B, h_C, M, N, K);
}

} // namespace cuda_openmp_matmul

#endif // __CUDACC__