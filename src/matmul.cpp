#include "matmul.hpp"

template<typename T>
Matrix<T> MatMul<T>::mm(const Matrix<T>& A, const Matrix<T>& B, MatMulMethod method) {
    switch (method) {
        case MatMulMethod::Vanilla:
            return mm_vanilla(A, B);
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






// Explicit template instantiation for common types
template class MatMul<int>;
template class MatMul<double>;
template class MatMul<float>;