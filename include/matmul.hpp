#pragma once

#include "matrix.hpp"
#if defined(_OPENMP)
    #include <omp.h>
#endif


enum class MatMulMethod {
    Vanilla,
    LoopUnrolled4,
    LoopUnrolled8,
    LoopUnrolled,
    Tiled,
    AVX2,
    AVX512,
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

        // AVX512 implementation would go here

        return C;
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