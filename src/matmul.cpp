#include "matmul.hpp"

template<typename T>
Matrix<T> MatMul<T>::mm(const Matrix<T>& A, const Matrix<T>& B, MatMulMethod method) {
    switch (method) {
        case MatMulMethod::Vanilla:
            return mm_vanilla(A, B);
        case MatMulMethod::LoopUnrolled4:
            return mm_unrolled4(A, B);
        
        // Future methods can be added here
        default:
            throw std::invalid_argument("Unknown matrix multiplication method");
    }
}


template<typename T>
Matrix<T> MatMul<T>::mm_vanilla(const Matrix<T>& A, const Matrix<T>& B) {
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


template<typename T>
Matrix<T> MatMul<T>::mm_unrolled4(const Matrix<T>& A, const Matrix<T>& B) {
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




// Explicit template instantiation for common types
template class MatMul<int>;
template class MatMul<double>;
template class MatMul<float>;