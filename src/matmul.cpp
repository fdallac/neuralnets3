#include "matmul.hpp"

template<typename T>
Matrix<T> MatMul<T>::mm(const Matrix<T>& A, const Matrix<T>& B, MatMulMethod method, std::size_t n_unroll) {
    switch (method) {
        case MatMulMethod::Vanilla:
            return mm_vanilla(A, B);
        case MatMulMethod::LoopUnrolled4:
            return mm_unrolled4(A, B);
        case MatMulMethod::LoopUnrolled8:
            return mm_unrolled8(A, B);
        case MatMulMethod::LoopUnrolledN:
            // Example with unroll factor of 16
            return mm_unrolledN(A, B, n_unroll);
        
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



template<typename T>
Matrix<T> MatMul<T>::mm_unrolled8(const Matrix<T>& A, const Matrix<T>& B) {
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


template<typename T>
Matrix<T> MatMul<T>::mm_unrolledN(const Matrix<T>& A, const Matrix<T>& B, std::size_t n_unroll) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
    }
    Matrix<T> C(A.rows(), B.cols());
    const std::size_t N = A.cols();
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < B.cols(); ++j) {
            T acc[n_unroll] = {T{}};
            std::size_t k = 0;
            // Process n elements at a time
            for (; k + n_unroll - 1 < N; k += n_unroll) {
                for (std::size_t u = 0; u < n_unroll; ++u) {
                    acc[u] += A(i, k + u) * B(k + u, j);
                }
            }
            // Add the partial sums together
            T sum = T{};
            for (std::size_t u = 0; u < n_unroll; ++u) {
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




// Explicit template instantiation for common types
template class MatMul<int>;
template class MatMul<double>;
template class MatMul<float>;