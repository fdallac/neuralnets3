#pragma once

#include "matrix.hpp"

enum class MatMulMethod {
    Vanilla,
    LoopUnrolled4,
    LoopUnrolled8,
    LoopUnrolledN,
    SIMD,
    OpenMP,
    CUDA
};

template<typename T>
class MatMul {
public:
    // Single interface for matrix multiplication
    static Matrix<T> mm(
        const Matrix<T>& A, 
        const Matrix<T>& B, 
        MatMulMethod method = MatMulMethod::Vanilla, 
        std::size_t n_unroll = 4
    );

    // Specific implementations
    static Matrix<T> mm_vanilla(const Matrix<T>& A, const Matrix<T>& B);
    static Matrix<T> mm_unrolled4(const Matrix<T>& A, const Matrix<T>& B);
    static Matrix<T> mm_unrolled8(const Matrix<T>& A, const Matrix<T>& B);
    static Matrix<T> mm_unrolledN(const Matrix<T>& A, const Matrix<T>& B, std::size_t n_unroll);
    // static Matrix<T> mm_simd(const Matrix<T>& A, const Matrix<T>& B);
    // static Matrix<T> mm_openmp(const Matrix<T>& A, const Matrix<T>& B);
    // static Matrix<T> mm_cuda(const Matrix<T>& A, const Matrix<T>& B);
};