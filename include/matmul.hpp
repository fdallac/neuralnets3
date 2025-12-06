#pragma once

#include "matrix.hpp"
#if defined(_OPENMP)
    #include <omp.h>
#endif
#include <immintrin.h>


enum class MatMulMethod {
    Vanilla,
    LoopUnrolled4,
    LoopUnrolled8,
    LoopUnrolled,
    Tiled,
    SIMD_AVX2,
    SIMD_AVX512,
    OpenMP,
    CUDA
};

template<typename T>
class MatMul {
public:
    /// Main interface for matrix multiplication


    /// @brief  Single interface for matrix multiplication
    /// @param A Left-hand side matrix
    /// @param B Right-hand side matrix
    /// @param method Method to use for multiplication (default: Vanilla)
    /// @return C = A * B
    static Matrix<T> mm(const Matrix<T>& A, const Matrix<T>& B,
                 MatMulMethod method = MatMulMethod::Vanilla) { 
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
            
            // Future methods can be added here
            default:
                throw std::invalid_argument("Unknown matrix multiplication method");
        }
    }


    
    /// Specific implementations


    /// @brief  Vanilla matrix multiplication
    /// @param A 
    /// @param B
    /// @return C = A * B
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


    /// @brief  Loop unrolled by N matrix multiplication
    /// @tparam N_UNROLL
    /// @param A 
    /// @param B
    /// @return C = A * B
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



    /// @brief  Tiled matrix multiplication
    /// @tparam TILE_SIZE 
    /// @param A 
    /// @param B 
    /// @return C = A * B
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



    /// TODO: Placeholder for SIMD (AVX2) implementation
    static Matrix<T> mm_avx2(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }
        Matrix<T> C(A.rows(), B.cols());

        // AVX2 implementation would go here

        return C;
    }


    /// TODO: Placeholder for SIMD (AVX-512) implementation
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


    /// @brief  OpenMP parallelized matrix multiplication
    /// @param A
    /// @param B
    /// @param num_threads Number of threads to use (-1 for max available)
    /// @param block_k Block size for the K dimension
    /// @return C = A * B
    static Matrix<T> mm_openmp(const Matrix<T>& A, const Matrix<T>& B, int num_threads = -1, size_t block_k = 64) {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }
        Matrix<T> C(A.rows(), B.cols());

        // Run OpenMP implementation for matrix pointers
        mm_openmp_ptr(A, B, C, num_threads, block_k);

        return C;
    }

    /// Private helper for OpenMP implementation using raw pointers
    static void mm_openmp_ptr(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, int num_threads = -1, size_t block_k = 64) {
        const std::size_t M = A.rows();
        const std::size_t N = B.cols();
        const std::size_t K = A.cols();
        const T* A_data = A.data();
        const T* B_data = B.data();
        T* C_data = C.data();

        // Warning if openmp is not enabled
        #if !defined(_OPENMP)
            #warning "OpenMP is not enabled. Compiling without OpenMP support."
        #endif

        // Set number of threads
        #if defined(_OPENMP)
            if (num_threads <= 0) {
                num_threads = omp_get_max_threads();
            }
            omp_set_num_threads(num_threads);
        #endif

        // Parallelize outer loop with OpenMP
        #if defined(_OPENMP)
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


    /// TODO: Placeholder for CUDA implementation
    static Matrix<T> mm_cuda(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }
        Matrix<T> C(A.rows(), B.cols());

        // CUDA implementation would go here

        return C;
    }

};