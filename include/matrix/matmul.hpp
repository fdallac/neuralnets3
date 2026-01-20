/**
 * @file matmul.hpp
 * @brief Multiple implementations of matrix multiplication
 * 
 * Provides various optimized matrix multiplication algorithms:
 * - Vanilla: Simple triple-nested loop
 * - LoopUnrolled: Unrolled inner loops for better instruction pipelining
 * - Tiled: Cache-friendly blocked multiplication
 * - SIMD_AVX512: Vectorized using AVX-512 intrinsics (float/double only)
 * - OpenMP: Parallelized using OpenMP
 * - SIMD_OpenMP_Tile: Combines tiling, SIMD, and OpenMP
 * - CUDA: GPU-accelerated multiplication using CUDA
 * - CUDA_OpenMP: Hybrid GPU + CPU parallelism
 * @tparam T Numeric type (float, double, int, etc.)
 */

#pragma once

#include "matrix/matrix.hpp"
#if defined(_OPENMP)
    #include <omp.h>
#endif
#include <immintrin.h>

// Include CUDA interface when CUDA is available
// matmul_cuda.hpp is a C++ compatible header (no CUDA syntax)
// The actual CUDA code is compiled separately in matmul_cuda.cu
#ifdef CUDA_AVAILABLE
    #include "matrix/matmul_cuda.hpp"
#endif


/**
 * @brief Matrix multiplication method selection
 * 
 * Different methods offer tradeoffs between portability, simplicity, and performance.
 */
enum class MatMulMethod {
    Vanilla,        ///< Simple O(n³) algorithm
    LoopUnrolled4,  ///< 4-way loop unrolling
    LoopUnrolled8,  ///< 8-way loop unrolling
    LoopUnrolled,   ///< Template-parameterized unrolling
    Tiled,          ///< Cache-blocking/tiling
    SIMD_AVX2,      ///< AVX2 vectorization (not implemented)
    SIMD_AVX512,    ///< AVX-512 vectorization (float/double only)
    OpenMP,         ///< Multi-threaded with OpenMP
    SIMD_OpenMP_Tile, ///< Best combination: SIMD + OpenMP + Tiling
    CUDA,           ///< GPU acceleration with CUDA
    CUDA_OpenMP,    ///< Hybrid GPU + CPU parallelism
    Auto            ///< Automatic selection based on matrix sizes
};

/**
 * @brief Matrix multiplication dispatcher and implementations
 * @tparam T Numeric type (float, double, int, etc.)
 * 
 * All methods are static - no instantiation needed.
 * Use MatMul<T>::mm(A, B, method) to compute A * B.
 */
template<typename T>
class MatMul {
public:
    /**
     * @brief Main interface for matrix multiplication
     * @param A Left-hand matrix (m x k)
     * @param B Right-hand matrix (k x n)
     * @param method Algorithm to use (default: Auto)
     * @return Result matrix C = A * B (m x n)
     * @throws std::invalid_argument if A.cols() != B.rows()
     * 
     * Dispatches to specific implementation based on method parameter.
     * Recommended: MatMulMethod::Auto for automatic selection.
     */
    static Matrix<T> mm(const Matrix<T>& A, const Matrix<T>& B,
                 MatMulMethod method = MatMulMethod::Auto) { 

        if (method == MatMulMethod::Auto) {
            // Heuristic for automatic method selection
            std::size_t volume = A.rows() * A.cols() * B.cols();
            
            if (volume <= 128 * 128 * 128) {
                // Small matrices: use SIMD implementation
                method = MatMulMethod::SIMD_AVX512;
            } else if (volume <= 1024 * 1024 * 1024) {        
                // Medium matrices: use pure CUDA if available, else SIMD + OpenMP
#ifdef CUDA_AVAILABLE
                method = MatMulMethod::CUDA;
#else
                method = MatMulMethod::SIMD_OpenMP_Tile;
#endif
            }
        } else {
                // Large matrices: use hybrid CUDA + OpenMP if available, else SIMD + OpenMP
#ifdef CUDA_AVAILABLE
                method = MatMulMethod::CUDA_OpenMP;
#else
                method = MatMulMethod::SIMD_OpenMP_Tile;
#endif
        }
        
        // Dispatch to selected method
        switch (method) {
            case MatMulMethod::Vanilla:
                return mm_vanilla(A, B);

            case MatMulMethod::LoopUnrolled4:
                return mm_unrolled4(A, B);

            case MatMulMethod::LoopUnrolled8:
                return mm_unrolled8(A, B);

            case MatMulMethod::LoopUnrolled:
                return mm_unrolled<8>(A, B);

            case MatMulMethod::Tiled:
                return mm_tiled<16>(A, B);

            case MatMulMethod::OpenMP:
                return mm_openmp(A, B);

            case MatMulMethod::SIMD_AVX512:
                return mm_avx512(A, B);

            case MatMulMethod::SIMD_OpenMP_Tile:
                return mm_simd_openmp_tile(A, B);

            case MatMulMethod::CUDA:
#ifdef CUDA_AVAILABLE
                return mm_cuda(A, B);
#else
                throw std::invalid_argument("CUDA method selected but not compiled with nvcc");
#endif

            case MatMulMethod::CUDA_OpenMP:
#ifdef CUDA_AVAILABLE
                return mm_cuda_openmp(A, B);
#else
                throw std::invalid_argument("CUDA_OpenMP method selected but not compiled with nvcc");
#endif
            
            // Future methods can be added here
            default:
                throw std::invalid_argument("Unknown matrix multiplication method");
        }
    }


    
    /// Specific implementations


    /**
     * @brief Vanilla (naive) matrix multiplication using triple nested loops
     * @param A Left-hand matrix (M x K)
     * @param B Right-hand matrix (K x N)
     * @return Result matrix C = A * B (M x N)
     * @throws std::invalid_argument if A.cols() != B.rows()
     * 
     * Algorithm: Simple O(MNK) triple-nested loop implementation.
     * 
     * Performance characteristics:
     * - Time complexity: O(M * N * K)
     * - Space complexity: O(M * N) for result
     * - No optimizations applied
     * - Poor cache locality (accessing B column-wise)
     * - Baseline reference implementation
     * 
     * Use case: Small matrices, educational purposes, correctness verification.
     */
    static Matrix<T> mm_vanilla(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }
        Matrix<T> C(A.rows(), B.cols());
        for (std::size_t i = 0; i < A.rows(); ++i) {
            for (std::size_t j = 0; j < B.cols(); ++j) {
                T sum = T{};
                for (std::size_t k = 0; k < A.cols(); ++k) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }
        return C;
    }


    /**
     * @brief Matrix multiplication with 4-way loop unrolling
     * @param A Left-hand matrix (M x K)
     * @param B Right-hand matrix (K x N)
     * @return Result matrix C = A * B (M x N)
     * @throws std::invalid_argument if A.cols() != B.rows()
     * 
     * Algorithm: Unrolls the innermost k-loop by factor of 4 using separate accumulators.
     * 
     * Performance characteristics:
     * - Time complexity: O(M * N * K)
     * - Improved instruction-level parallelism (ILP)
     * - Reduces loop overhead by ~75%
     * - Uses 4 accumulator variables to reduce data dependencies
     * - Compiler can better schedule independent operations
     * - Handles remaining elements (K % 4) with scalar loop
     */
    static Matrix<T> mm_unrolled4(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }
        Matrix<T> C(A.rows(), B.cols());
        const std::size_t N = A.cols();
        for (std::size_t i = 0; i < A.rows(); ++i) {
            for (std::size_t j = 0; j < B.cols(); ++j) {
                T acc0 = T{};
                T acc1 = T{};
                T acc2 = T{};
                T acc3 = T{};
                std::size_t k = 0;
                // Process 4 elements at a time
                for (; k + 3 < N; k += 4) {
                    acc0 += A(i, k    ) * B(k,     j);
                    acc1 += A(i, k + 1) * B(k + 1, j);
                    acc2 += A(i, k + 2) * B(k + 2, j);
                    acc3 += A(i, k + 3) * B(k + 3, j);
                }
                // Add the partial sums together
                T sum = acc0 + acc1 + acc2 + acc3;
                // Handle leftover values
                for (; k < N; ++k) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }
        return C;
    }


    /**
     * @brief Matrix multiplication with 8-way loop unrolling
     * @param A Left-hand matrix (M x K)
     * @param B Right-hand matrix (K x N)
     * @return Result matrix C = A * B (M x N)
     * @throws std::invalid_argument if A.cols() != B.rows()
     * 
     * Algorithm: Unrolls the innermost k-loop by factor of 8 using separate accumulators.
     * 
     * Performance characteristics:
     * - Time complexity: O(M * N * K)
     * - Higher instruction-level parallelism than 4-way unrolling
     * - Reduces loop overhead by ~87.5%
     * - Uses 8 accumulator variables for maximum ILP
     * - Better utilization of CPU execution units
     * - Handles remaining elements (K % 8) with scalar loop
     */
    static Matrix<T> mm_unrolled8(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }
        Matrix<T> C(A.rows(), B.cols());
        const std::size_t N = A.cols();
        for (std::size_t i = 0; i < A.rows(); ++i) {
            for (std::size_t j = 0; j < B.cols(); ++j) {
                T acc0 = T{};
                T acc1 = T{};
                T acc2 = T{};
                T acc3 = T{};
                T acc4 = T{};
                T acc5 = T{};
                T acc6 = T{};
                T acc7 = T{};
                std::size_t k = 0;
                // Process 8 elements at a time
                for (; k + 7 < N; k += 8) {
                    acc0 += A(i, k    ) * B(k,     j);
                    acc1 += A(i, k + 1) * B(k + 1, j);
                    acc2 += A(i, k + 2) * B(k + 2, j);
                    acc3 += A(i, k + 3) * B(k + 3, j);
                    acc4 += A(i, k + 4) * B(k + 4, j);
                    acc5 += A(i, k + 5) * B(k + 5, j);
                    acc6 += A(i, k + 6) * B(k + 6, j);
                    acc7 += A(i, k + 7) * B(k + 7, j);
                }
                // Add the partial sums together
                T sum = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
                // Handle leftover values
                for (; k < N; ++k) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }
        return C;
    }


    /**
     * @brief Template-parameterized loop unrolling for matrix multiplication
     * @tparam N_UNROLL Unrolling factor (number of iterations per unrolled block)
     * @param A Left-hand matrix (M x K)
     * @param B Right-hand matrix (K x N)
     * @return Result matrix C = A * B (M x N)
     * @throws std::invalid_argument if A.cols() != B.rows()
     * 
     * Algorithm: Compile-time configurable loop unrolling using template parameter.
     * 
     * Performance characteristics:
     * - Time complexity: O(M * N * K)
     * - Reduces loop overhead by (N_UNROLL-1)/N_UNROLL
     * - Uses N_UNROLL accumulator variables
     * - Instruction-level parallelism scales with N_UNROLL
     * - Code size increases linearly with N_UNROLL
     * - Handles remaining elements (K % N_UNROLL) with scalar loop
     */
    template<std::size_t N_UNROLL>
    static Matrix<T> mm_unrolled(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }
        Matrix<T> C(A.rows(), B.cols());
        const std::size_t N = A.cols();
        for (std::size_t i = 0; i < A.rows(); ++i) {
            for (std::size_t j = 0; j < B.cols(); ++j) {
                T acc[N_UNROLL] = {T{}};
                std::size_t k = 0;

                // Process n elements at a time
                for (; k + N_UNROLL - 1 < N; k += N_UNROLL) {
                    for (std::size_t u = 0; u < N_UNROLL; ++u) {
                        acc[u] += A(i, k + u) * B(k + u, j);
                    }
                }
                // Add the partial sums together
                T sum = T{};
                for (std::size_t u = 0; u < N_UNROLL; ++u) {
                    sum += acc[u];
                }
                // Handle leftover values
                for (; k < N; ++k) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }
        return C;
    }



    /**
     * @brief Cache-blocked (tiled) matrix multiplication
     * @tparam TILE_SIZE Dimension of square tiles (e.g., 16, 32, 64)
     * @param A Left-hand matrix (M x K)
     * @param B Right-hand matrix (K x N)
     * @return Result matrix C = A * B (M x N)
     * @throws std::invalid_argument if A.cols() != B.rows()
     * 
     * Algorithm: Divides matrices into TILE_SIZE x TILE_SIZE blocks to improve cache locality.
     * Uses ikj loop ordering with blocking to maximize cache reuse.
     * 
     * Performance characteristics:
     * - Time complexity: O(M * N * K) with improved cache efficiency
     * - Reduces cache misses dramatically (O(N³/B) vs O(N³/√B))
     * - Fits working set into L1/L2 cache
     * - Particularly effective for large matrices
     * - Automatically handles non-divisible dimensions
     */
    template<std::size_t TILE_SIZE>
    static Matrix<T> mm_tiled(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }

        Matrix<T> C(A.rows(), B.cols());
        for (std::size_t i = 0; i < A.rows(); i += TILE_SIZE) {
            size_t i_max = std::min(i + TILE_SIZE, A.rows());
            for (std::size_t k = 0; k < A.cols(); k += TILE_SIZE) {
                size_t k_max = std::min(k + TILE_SIZE, A.cols());
                for (std::size_t j = 0; j < B.cols(); j += TILE_SIZE) {
                    size_t j_max = std::min(j + TILE_SIZE, B.cols());

                    // Multiply the tiles
                    for (std::size_t ii = i; ii < i_max; ++ii) {
                        for (std::size_t kk = k; kk < k_max; ++kk) {
                            T a = A(ii, kk);
                            for (std::size_t jj = j; jj < j_max; ++jj) {
                                C(ii, jj) += a * B(kk, jj);
                            }
                        }
                    }
                }
            }
        }
        return C;
    }



    /**
     * @brief AVX2 SIMD-accelerated matrix multiplication (placeholder)
     * @param A Left-hand matrix (M x K)
     * @param B Right-hand matrix (K x N)
     * @return Result matrix C = A * B (M x N)
     * @throws std::invalid_argument if A.cols() != B.rows()
     * 
     * Status: NOT IMPLEMENTED - Placeholder for future AVX2 implementation.
     * 
     * Planned algorithm: Vectorized multiplication using 256-bit AVX2 instructions.
     * - 8 floats or 4 doubles per SIMD register
     * - Fused multiply-add (FMA) operations
     * - Horizontal reduction for accumulation
     * 
     * @note Currently returns empty matrix - implementation pending.
     */
    static Matrix<T> mm_avx2(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }
        Matrix<T> C(A.rows(), B.cols());

        // AVX2 implementation would go here

        return C;
    }


    /**
     * @brief AVX-512 SIMD-accelerated matrix multiplication
     * @param A Left-hand matrix (M x K)
     * @param B Right-hand matrix (K x N)
     * @return Result matrix C = A * B (M x N)
     * @throws std::invalid_argument if A.cols() != B.rows()
     * @throws std::invalid_argument if T is not float, double, or int
     * 
     * Algorithm: High-performance vectorized multiplication using 512-bit AVX-512 instructions.
     * - Transposes B for row-major access patterns
     * - Processes 16 floats, 8 doubles, or 16 ints per SIMD operation
     * - Uses FMA (fused multiply-add) for reduced latency
     * - Handles remainder elements with scalar code
     *
     * Implementation details:
     * - float: _mm512_fmadd_ps, 16 elements per vector
     * - double: _mm512_fmadd_pd, 8 elements per vector
     * - int: _mm512_mullo_epi32 + _mm512_add_epi32, 16 elements per vector
     * 
     * @see mm_avx512_transposed_ptr for low-level implementation
     */
    static Matrix<T> mm_avx512(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }
        Matrix<T> C(A.rows(), B.cols());

        // Transpose B for better memory access
        Matrix<T> B_t = B.transpose();

        // Run AVX512 implementation for matrix pointers
        mm_avx512_transposed_ptr(A.data(), B_t.data(), C.data(), A.rows(), B.cols(), A.cols());

        return C;
    }

    /**
     * @brief Dispatcher for type-specific AVX-512 implementations
     * @param A Pointer to left-hand matrix data (row-major, M x K)
     * @param B_t Pointer to transposed right-hand matrix data (row-major, N x K)
     * @param C Pointer to result matrix data (row-major, M x N)
     * @param M Number of rows in A
     * @param N Number of columns in B
     * @param K Number of columns in A (rows in B)
     * @throws std::invalid_argument if T is not float, double, or int
     * 
     * Uses compile-time type dispatch (if constexpr) to select optimal implementation:
     * - float → mm_avx512_transpose_float
     * - double → mm_avx512_transpose_double
     * - int → mm_avx512_transpose_int
     * 
     * @note Private helper method - not intended for direct use.
     */
    static void mm_avx512_transposed_ptr(const T* A, const T* B_t, T* C, std::size_t M, std::size_t N, std::size_t K) {
        
        if constexpr (std::is_same_v<T, float>) {
            mm_avx512_transpose_float(A, B_t, C, M, N, K);
        } else if constexpr (std::is_same_v<T, double>) {
            mm_avx512_transpose_double(A, B_t, C, M, N, K);
        } else if constexpr (std::is_same_v<T, int>) {
            mm_avx512_transpose_int(A, B_t, C, M, N, K);
        } else {
            throw std::invalid_argument("AVX-512 implementation only supports float and double types");
        }
    }


    /**
     * @brief AVX-512 matrix multiplication for 32-bit integers
     * @param A Pointer to left-hand matrix (M x K)
     * @param B_t Pointer to transposed right-hand matrix (N x K)
     * @param C Pointer to result matrix (M x N)
     * @param M Number of rows in A
     * @param N Number of columns in B
     * @param K Inner dimension
     * 
     * Implementation details:
     * - Uses _mm512_mullo_epi32 for 32-bit integer multiplication
     * - Processes 16 integers per SIMD operation (512 bits / 32 bits)
     * - Horizontal reduction by storing and summing vector elements
     * - Handles K % 16 remainder with scalar code
     * 
     * 
     * @note Private helper method for mm_avx512.
     */
    static void mm_avx512_transpose_int(const int* A, const int* B_t, int* C, std::size_t M, std::size_t N, std::size_t K) {
        constexpr std::size_t V = 16;
        for (std::size_t i = 0; i < M; ++i) {
            const int* arow = A + i * K;
            int* crow = C + i * N;
            for (std::size_t j = 0; j < N; ++j) {
                const int* brow = B_t + j * K;
                __m512i acc = _mm512_setzero_si512();
                std::size_t k = 0;
                for (; k + V <= K; k += V) {
                    __m512i va = _mm512_loadu_si512((__m512i const*)(arow + k));
                    __m512i vb = _mm512_loadu_si512((__m512i const*)(brow + k));
                    acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(va, vb));
                }
                int tmp[V];
                _mm512_storeu_si512((__m512i*)tmp, acc);
                int sum = 0;
                for (std::size_t t = 0; t < V; ++t) sum += tmp[t];
                for (; k < K; ++k) sum += arow[k] * brow[k];
                crow[j] = sum;
            }
        }
    }


    /**
     * @brief AVX-512 matrix multiplication for single-precision floats
     * @param A Pointer to left-hand matrix (M x K)
     * @param B_t Pointer to transposed right-hand matrix (N x K)
     * @param C Pointer to result matrix (M x N)
     * @param M Number of rows in A
     * @param N Number of columns in B
     * @param K Inner dimension
     * 
     * Implementation details:
     * - Uses _mm512_fmadd_ps (fused multiply-add) for optimal performance
     * - Processes 16 floats per SIMD operation (512 bits / 32 bits)
     * - Single-cycle latency FMA reduces computation time
     * - Horizontal reduction by storing and summing vector elements
     * - Handles K % 16 remainder with scalar code
     * 
     * @note Private helper method for mm_avx512.
     */
    static void mm_avx512_transpose_float(const float* A, const float* B_t, float* C, std::size_t M, std::size_t N, std::size_t K) {
        constexpr std::size_t V = 16;
        for (std::size_t i = 0; i < M; ++i) {
            const float* arow = A + i * K;
            float* crow = C + i * N;
            for (std::size_t j = 0; j < N; ++j) {
                const float* brow = B_t + j * K;
                __m512 acc = _mm512_setzero_ps();
                std::size_t k = 0;
                for (; k + V <= K; k += V) {
                    __m512 va = _mm512_loadu_ps(arow + k);
                    __m512 vb = _mm512_loadu_ps(brow + k);
                    acc = _mm512_fmadd_ps(va, vb, acc);
                }
                float tmp[V];
                _mm512_storeu_ps(tmp, acc);
                float sum = 0;
                for (std::size_t t = 0; t < V; ++t) sum += tmp[t];
                for (; k < K; ++k) sum += arow[k] * brow[k];
                crow[j] = sum;
            }
        }
    }


    /**
     * @brief AVX-512 matrix multiplication for double-precision floats
     * @param A Pointer to left-hand matrix (M x K)
     * @param B_t Pointer to transposed right-hand matrix (N x K)
     * @param C Pointer to result matrix (M x N)
     * @param M Number of rows in A
     * @param N Number of columns in B
     * @param K Inner dimension
     * 
     * Implementation details:
     * - Uses _mm512_fmadd_pd (fused multiply-add) for optimal performance
     * - Processes 8 doubles per SIMD operation (512 bits / 64 bits)
     * - Single-cycle latency FMA reduces computation time
     * - Horizontal reduction by storing and summing vector elements
     * - Handles K % 8 remainder with scalar code
     *  
     * @note Private helper method for mm_avx512.
     */
    static void mm_avx512_transpose_double(const double* A, const double* B_t, double* C, std::size_t M, std::size_t N, std::size_t K) {
        constexpr std::size_t V = 8;
        for (std::size_t i = 0; i < M; ++i) {
            const double* arow = A + i * K;
            double* crow = C + i * N;
            for (std::size_t j = 0; j < N; ++j) {
                const double* brow = B_t + j * K;
                __m512d acc = _mm512_setzero_pd();
                std::size_t k = 0;
                for (; k + V <= K; k += V) {
                    __m512d va = _mm512_loadu_pd(arow + k);
                    __m512d vb = _mm512_loadu_pd(brow + k);
                    acc = _mm512_fmadd_pd(va, vb, acc);
                }
                double tmp[V];
                _mm512_storeu_pd(tmp, acc);
                double sum = 0;
                for (std::size_t t = 0; t < V; ++t) sum += tmp[t];
                for (; k < K; ++k) sum += arow[k] * brow[k];
                crow[j] = sum;
            }
        }
    }


    /**
     * @brief Multi-threaded matrix multiplication using OpenMP
     * @param A Left-hand matrix (M x K)
     * @param B Right-hand matrix (K x N)
     * @param num_threads Number of threads (-1 for maximum available, default: -1)
     * @param block_k Block size for k-dimension to improve cache locality (default: 64)
     * @return Result matrix C = A * B (M x N)
     * @throws std::invalid_argument if A.cols() != B.rows()
     * 
     * Algorithm: Parallelizes outer i-loop using OpenMP with static scheduling.
     * Also applies k-blocking for better cache performance.
     * 
     * OpenMP configuration:
     * - num_threads = -1: Uses omp_get_max_threads() (all available cores)
     * - num_threads > 0: Uses specified number of threads
     * - Fallback to serial execution if OpenMP not enabled at compile time
     * 
     * Block size tuning:
     * - block_k = 64: Good default for most L1 cache sizes
     * - Smaller: Better cache locality, higher loop overhead
     * - Larger: Less overhead, possible cache thrashing
     * 
     * Use case: Large matrices on multi-core CPUs where parallelization overhead is justified.
     * 
     * @note Requires -fopenmp compilation flag. Falls back to serial if not available.
     * @see mm_openmp_ptr for low-level implementation
     */
    static Matrix<T> mm_openmp(const Matrix<T>& A, const Matrix<T>& B, int num_threads = -1, size_t block_k = 64) {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }
        Matrix<T> C(A.rows(), B.cols());

        // Run OpenMP implementation for matrix pointers
        mm_openmp_ptr(A, B, C, num_threads, block_k);

        return C;
    }

    /**
     * @brief Low-level OpenMP implementation using raw pointer access
     * @param A Left-hand matrix reference
     * @param B Right-hand matrix reference
     * @param C Result matrix reference (modified in-place)
     * @param num_threads Number of threads (-1 for maximum)
     * @param block_k Block size for k-dimension
     * 
     * Implementation details:
     * - Uses raw pointers for cache-friendly access
     * - #pragma omp parallel for schedule(static) on outer loop
     * - Static scheduling for minimal overhead
     * - Each thread processes contiguous rows (better cache locality)
     * - k-blocking reduces cache misses in inner loop
     * 
     * Thread safety:
     * - No race conditions: each thread writes to distinct C elements
     * - Read-only access to A and B matrices
     * 
     * @note Private helper method - not intended for direct use.
     */
    static void mm_openmp_ptr(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, int num_threads = -1, size_t block_k = 64) {
        const std::size_t M = A.rows();
        const std::size_t N = B.cols();
        const std::size_t K = A.cols();
        const T* A_data = A.data();
        const T* B_data = B.data();
        T* C_data = C.data();

        
#if !defined(_OPENMP)
            // Warning if openmp is not enabled
            #warning "OpenMP is not enabled. Compiling without OpenMP support."
#endif

        // Set number of threads
#if defined(_OPENMP)
            if (num_threads <= 0) {
                num_threads = omp_get_max_threads();
            }
            omp_set_num_threads(num_threads);

        // Parallelize outer loop with OpenMP
            #pragma omp parallel for schedule(static)
#endif
            for (std::size_t i = 0; i < M; ++i) {
                for (std::size_t j = 0; j < N; ++j) {
                    T sum = T{};
                    for (std::size_t k = 0; k < K; k += block_k) {
                        std::size_t k_max = std::min(k + block_k, K);
                        for (std::size_t kk = k; kk < k_max; ++kk) {
                            sum += A_data[i * K + kk] * B_data[kk * N + j];
                        }
                    }
                    C_data[i * N + j] = sum;
                }
            }
    }



    /**
     * @brief Highly optimized matrix multiplication combining multiple techniques
     * @param A Left-hand matrix (M x K)
     * @param B Right-hand matrix (K x N)
     * @param num_threads Number of threads (-1 for maximum available, default: -1)
     * @param tile_i Tile size for i (row) dimension (default: 16)
     * @param tile_j Tile size for j (column) dimension (default: 16)
     * @param tile_k Tile size for k (inner) dimension (default: 64)
     * @return Result matrix C = A * B (M x N)
     * @throws std::invalid_argument if A.cols() != B.rows()
     * 
     * Algorithm: Combines three optimization techniques for maximum performance:
     * 1. **Cache blocking (tiling)**: Improves data locality
     * 2. **SIMD vectorization (AVX-512)**: Parallel computation within cores
     * 3. **Multi-threading (OpenMP)**: Parallel computation across cores
     * 
     * Implementation strategy:
     * - Transposes B once for row-major access
     * - Tiles computation into cache-friendly blocks
     * - Parallelizes tile processing with OpenMP
     * - Uses AVX-512 intrinsics within each tile (float/double only)
     * - Falls back to scalar for unsupported types
     * 
     * Parameter tuning guide:
     * - tile_i, tile_j: Should fit in L1 cache (16-32 typical)
     * - tile_k: Should fit in L2 cache (64-128 typical)
     * - num_threads: Use -1 for automatic (all cores)
     *  
     * @see mm_simd_openmp_tile_ptr for low-level implementation details
     */
    static Matrix<T> mm_simd_openmp_tile(const Matrix<T>& A, const Matrix<T>& B, int num_threads = -1, size_t tile_i = 16, size_t tile_j = 16, size_t tile_k = 64) {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }

        const size_t M = A.rows();
        const size_t N = B.cols();
        const size_t K = A.cols();

        Matrix<T> C(M, N);

        // Transpose B for better memory access
        Matrix<T> B_t = B.transpose();

        // SIMD + OpenMP + Tiled implementation
        mm_simd_openmp_tile_ptr(A.data(), B_t.data(), C.data(), M, N, K, num_threads, tile_i, tile_j, tile_k);

        return C;
    }


    /**
     * @brief Low-level optimized implementation using raw pointers
     * @param A Pointer to left-hand matrix (M x K, row-major)
     * @param B_t Pointer to transposed right-hand matrix (N x K, row-major)
     * @param C Pointer to result matrix (M x N, row-major, modified in-place)
     * @param M Number of rows in A
     * @param N Number of columns in B
     * @param K Inner dimension
     * @param num_threads Number of OpenMP threads
     * @param tile_i, tile_j, tile_k Tile dimensions for blocking
     * 
     * Implementation details:
     * - Three-level nested tiling (ii, jj, kk)
     * - OpenMP parallelization at tile level (#pragma omp parallel for)
     * - SIMD vectorization within tiles (compile-time dispatch)
     * - Row-major access patterns after B transpose
     * 
     * SIMD implementation (float/double):
     * - float: __m512 (16 elements), _mm512_fmadd_ps
     * - double: __m512d (8 elements), _mm512_fmadd_pd
     * - Accumulates within SIMD registers
     * - Horizontal reduction after vectorized loop
     * - Scalar cleanup for remainder elements
     * 
     * Fallback for other types:
     * - Scalar accumulation for int and custom types
     * - Still benefits from tiling and OpenMP
     * 
     * Memory access pattern:
     * - A: row i, columns [k, k+tile_k)
     * - B_t: row j, columns [k, k+tile_k) (originally B column j)
     * - C: Accumulates to element (i, j)
     * 
     * @note Private helper method with maximum optimization for float/double.
     */
    static void mm_simd_openmp_tile_ptr(const T* A, const T* B_t, T* C, size_t M, size_t N, size_t K, int num_threads = -1, size_t tile_i = 16, size_t tile_j = 16, size_t tile_k = 64) {
        
        
# if defined(_OPENMP)
            if (num_threads <= 0) {
                num_threads = omp_get_max_threads();
            }
            omp_set_num_threads(num_threads);

            #pragma omp parallel for schedule(static)
#endif

        for (size_t ii = 0; ii < M; ii += tile_i) {
            size_t i_max = std::min(ii + tile_i, M);
            
            for (size_t jj = 0; jj < N; jj += tile_j) {
                size_t j_max = std::min(jj + tile_j, N);
                
                for (size_t kk = 0; kk < K; kk += tile_k) {
                    size_t k_max = std::min(kk + tile_k, K);
                
                    // Multiply the tiles
                    for (size_t i = ii; i < i_max; ++i) {   
                        const T* arow = A + i * K;

                        for (size_t j = jj; j < j_max; ++j) {
                            const T* brow = B_t + j * K;

                            // SIMD blocks
                            if constexpr (std::is_same_v<T, float>) {
                                __m512 acc = _mm512_setzero_ps();
                                size_t k = kk;
                                constexpr size_t V = 16;
                                for (; k + V <= k_max; k += V) {
                                    __m512 va = _mm512_loadu_ps(arow + k);
                                    __m512 vb = _mm512_loadu_ps(brow + k);
                                    acc = _mm512_fmadd_ps(va, vb, acc);
                                }
                                float tmp[16];
                                _mm512_storeu_ps(tmp, acc);
                                float sum = 0;
                                for (size_t t = 0; t < V; ++t) sum += tmp[t];
                                for (; k < k_max; ++k) sum += arow[k] * brow[k];
                                C[i * N + j] += sum;
                            } else if constexpr (std::is_same_v<T, double>) {
                                __m512d acc = _mm512_setzero_pd();
                                size_t k = kk;
                                constexpr size_t V = 8;
                                for (; k + V <= k_max; k += V) {
                                    __m512d va = _mm512_loadu_pd(arow + k);
                                    __m512d vb = _mm512_loadu_pd(brow + k);
                                    acc = _mm512_fmadd_pd(va, vb, acc);
                                }
                                double tmp[8];
                                _mm512_storeu_pd(tmp, acc);
                                double sum = 0;
                                for (size_t t = 0; t < V; ++t) sum += tmp[t];
                                for (; k < k_max; ++k) sum += arow[k] * brow[k];
                                C[i * N + j] += sum;
                            } else {
                                // Fallback to scalar multiplication for unsupported types
                                T sum = T{};
                                for (size_t k = kk; k < k_max; ++k) {
                                    sum += arow[k] * brow[k];
                                }
                                C[i * N + j] += sum;
                            }
                        }
                    }
                }
            }
            
        }

    }
    





    /**
     * @brief GPU-accelerated matrix multiplication using CUDA
     * @param A Left-hand matrix (M x K)
     * @param B Right-hand matrix (K x N)
     * @return Result matrix C = A * B (M x N)
     * @throws std::invalid_argument if A.cols() != B.rows()
     * @throws std::runtime_error if CUDA is not available or on CUDA errors
     * 
     * Algorithm: GPU-accelerated GEMM using tiled CUDA kernels with shared memory.
     * 
     * Implementation details:
     * - Uses 16x16 thread blocks (256 threads per block)
     * - Shared memory tiling for efficient data reuse
     * - Coalesced global memory access patterns
     * - Handles arbitrary matrix dimensions with boundary checking
     * 
     * Workflow:
     * 1. Allocate device memory for matrices A, B, C
     * 2. Copy A and B from host to device (cudaMemcpyHostToDevice)
     * 3. Launch tiled kernel with appropriate grid/block dimensions
     * 4. Copy result C from device to host (cudaMemcpyDeviceToHost)
     * 5. Free device memory
     * 
     * Performance characteristics:
     * - Massive parallelism: Each thread computes one element of C
     * - Reduced memory bandwidth: Tiles are reused TILE_SIZE times from shared memory
     * - Overhead: Memory transfers can dominate for small matrices
     * - Best for: Large matrices (N > 256) where GPU parallelism outweighs transfer cost
     * 
     * Supported types:
     * - float: Single precision (best GPU performance)
     * - double: Double precision (requires compute capability >= 1.3)
     * 
     * @note Requires CUDA toolkit and NVIDIA GPU
     * @note Compile with nvcc: nvcc -x cu file.cpp -o output
     * @note Falls back to error if compiled without CUDA support
     * 
     * @see matmul_cuda.cuh for kernel implementation details
     */
    static Matrix<T> mm_cuda(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }
        
        Matrix<T> C(A.rows(), B.cols());

#ifdef CUDA_AVAILABLE
        // Check if CUDA is available at runtime
        if (!cuda_matmul::isCudaAvailable()) {
            throw std::runtime_error("No CUDA-capable device found");
        }
        
        // Only float and double are supported for CUDA
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            cuda_matmul::matmul_cuda_wrapper<T>(
                A.data(), 
                B.data(), 
                C.data(),
                A.rows(), 
                B.cols(), 
                A.cols()
            );
        } else {
            throw std::invalid_argument("CUDA matrix multiplication only supports float and double types");
        }
#else
        // CUDA not available at compile time
        throw std::runtime_error("CUDA support not compiled. Recompile with CMake and ensure CUDA toolkit is installed.");
#endif

        return C;
    }


    /**
     * @brief Hybrid CUDA + OpenMP matrix multiplication
     * @param A Left-hand matrix (M x K)
     * @param B Right-hand matrix (K x N)
     * @return Result matrix C = A * B (M x N)
     * @throws std::invalid_argument if A.cols() != B.rows()
     * @throws std::runtime_error if CUDA is not available
     * 
     * Algorithm: Heterogeneous computing approach combining GPU and CPU parallelism.
     * Divides the workload into horizontal strips (row blocks) processed by multiple CUDA streams.
     * 
     * **OpenMP's Role** (High-Level Task Parallelism):
     * - Divides the matrix into horizontal strips (row blocks)
     * - Manages multiple CUDA streams for concurrent execution
     * - Orchestrates asynchronous data transfers
     * - Handles work distribution across streams
     * 
     * **CUDA's Role** (Fine-Grained Parallelism):
     * - Performs tiled matrix multiplication on each block
     * - Uses shared memory to reduce global memory bandwidth
     * - Thousands of threads work in parallel on each tile
     * - Coalesced memory access patterns for efficiency
     * 
     * **Memory Management Strategy**:
     * - Small matrices: Unified Memory (cudaMallocManaged) for simplicity
     * - Large matrices: Explicit memory management with stream-based transfers
     * 
     * **Pipeline Execution**:
     * While GPU computes block i, the next block i+1 is transferred,
     * achieving near-optimal overlap of computation and data movement.
     * 
     * 
     * @see matmul_cuda_openmp.cuh for detailed implementation
     */
    static Matrix<T> mm_cuda_openmp(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }
        
        Matrix<T> C(A.rows(), B.cols());

#ifdef CUDA_AVAILABLE
        // Check if CUDA is available at runtime
        if (!cuda_matmul::isCudaAvailable()) {
            throw std::runtime_error("No CUDA-capable device found");
        }
        
        // Only float and double are supported
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            cuda_matmul::matmul_cuda_openmp_wrapper<T>(
                A.data(), 
                B.data(), 
                C.data(),
                A.rows(), 
                B.cols(), 
                A.cols()
            );
        } else {
            throw std::invalid_argument("CUDA+OpenMP matrix multiplication only supports float and double types");
        }
#else
        throw std::runtime_error("CUDA support not compiled. Recompile with CMake and ensure CUDA toolkit is installed.");
#endif

        return C;
    }


};