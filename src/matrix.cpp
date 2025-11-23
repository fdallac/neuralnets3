#include "matrix.hpp"
#include "matmul.hpp"

template<typename T>
Matrix<T>::Matrix()
    : rows_(0), cols_(0), data_() {}

template<typename T>
Matrix<T>::Matrix(std::size_t i, std::size_t j) 
    : rows_(i), cols_(j), data_(i * j, T{}) {}

template<typename T>
Matrix<T>::Matrix(std::size_t i, std::size_t j, const std::vector<T>& data)
    : rows_(i), cols_(j), data_(data) {
    if (data.size() != i * j) {
        throw std::invalid_argument("Data size does not match matrix dimensions");
    }
}

template<typename T>
T& Matrix<T>::operator()(std::size_t i, std::size_t j) {
    return data_[i * cols_ + j];
}

template<typename T>
const T& Matrix<T>::operator()(std::size_t i, std::size_t j) const {
    return data_[i * cols_ + j];
} 


template<typename T>
T& Matrix<T>::at(std::size_t i, std::size_t j) {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data_[i * cols_ + j];
}

template<typename T>
const T& Matrix<T>::at(std::size_t i, std::size_t j) const {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("Matrix indices out of range");
    }
    return data_[i * cols_ + j];
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    return MatMul<T>::mm(*this, other);
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (rows_ != other.rows() || cols_ != other.cols()) {
        throw std::invalid_argument("Matrix dimensions does not match for addition");
    }
    Matrix<T> result(rows_, cols_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result(i, j) = (*this)(i, j) + other(i, j);
        }
    }
    return result;
}

template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other) {
    if (rows_ != other.rows() || cols_ != other.cols()) {
        throw std::invalid_argument("Matrix dimensions does not match for addition");
    }
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            (*this)(i, j) += other(i, j);
        }
    }
    return *this;
}

template<typename T>
bool Matrix<T>::operator==(const Matrix<T>& other) const {
    if (rows_ != other.rows() || cols_ != other.cols()) {
        return false;
    }
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            if ((*this)(i, j) != other(i, j)) {
                return false;
            }
        }
    }
    return true;
}

template<typename T>
bool Matrix<T>::operator!=(const Matrix<T>& other) const {
    return !(*this == other);
}

// Explicit template instantiation for common types
template class Matrix<int>;
template class Matrix<double>;
template class Matrix<float>;