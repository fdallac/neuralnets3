#pragma once

#include "matrix.hpp"

enum class MatMulMethod {
    Vanilla,
    LoopUnrolling,
    SIMD,
    OpenMP,
    CUDA
};

template<typename T>
class MatMul {
public:
    // Single interface for matrix multiplication
    static Matrix<T> mm(const Matrix<T>& A, const Matrix<T>& B, MatMulMethod method = MatMulMethod::Vanilla);

    // Specific implementations
    static Matrix<T> mm_vanilla(const Matrix<T>& A, const Matrix<T>& B);
    // static Matrix<T> mm_loop_unrolling(const Matrix<T>& A, const Matrix<T>& B);
    // static Matrix<T> mm_simd(const Matrix<T>& A, const Matrix<T>& B);
    // static Matrix<T> mm_openmp(const Matrix<T>& A, const Matrix<T>& B);
    // static Matrix<T> mm_cuda(const Matrix<T>& A, const Matrix<T>& B);
};