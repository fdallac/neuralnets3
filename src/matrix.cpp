#include "matrix.hpp"

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


// Explicit template instantiation for common types
template class Matrix<int>;
template class Matrix<double>;
template class Matrix<float>;