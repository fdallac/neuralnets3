#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>



template<typename T>
class Matrix {
public:
    // Default constructor
    Matrix()
        : rows_(0), cols_(0), data_() {}

    // Constructor: rows x cols zero-initialized
    Matrix(std::size_t i, std::size_t j)
        : rows_(i), cols_(j), data_(i * j, T{}) {}

    // Constructor with data
    Matrix(std::size_t i, std::size_t j, const std::vector<T>& data)
        : rows_(i), cols_(j), data_(data) {
        if (data.size() != i * j) {
            throw std::invalid_argument("Data size does not match matrix dimensions");
        }
    }

    // Element access (unchecked)
    T& operator()(std::size_t i, std::size_t j) {
        return data_[i * cols_ + j];
    }

    const T& operator()(std::size_t i, std::size_t j) const {
        return data_[i * cols_ + j];
    }

    // Element access (checked)
    T& at(std::size_t i, std::size_t j) {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data_[i * cols_ + j];
    }

    const T& at(std::size_t i, std::size_t j) const {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data_[i * cols_ + j];
    }

    // Getters
    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    const T* data() const noexcept { return data_.data(); }
    T* data() noexcept { return data_.data(); }

    // Operators
    Matrix<T> operator*(const Matrix<T>& other) const;
    //  {
    //     return MatMul<T>::mm(*this, other);
    // }

    Matrix<T> operator+(const Matrix<T>& other) const {
        if (rows_ != other.rows() || cols_ != other.cols()) {
            throw std::invalid_argument("Matrix dimensions do not match for addition");
        }
        Matrix<T> result(rows_, cols_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result(i, j) = (*this)(i, j) + other(i, j);
            }
        }
        return result;
    }

    Matrix<T>& operator+=(const Matrix<T>& other) {
        if (rows_ != other.rows() || cols_ != other.cols()) {
            throw std::invalid_argument("Matrix dimensions do not match for addition");
        }
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                (*this)(i, j) += other(i, j);
            }
        }
        return *this;
    }

    bool operator==(const Matrix<T>& other) const {
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

    bool operator!=(const Matrix<T>& other) const {
        return !(*this == other);
    }

    // Transpose
    Matrix<T> transpose() const {
        Matrix<T> result(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    // Apply function element-wise
    Matrix<T> apply(T (*func)(T)) const {
        Matrix<T> result(rows_, cols_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result(i, j) = func((*this)(i, j));
            }
        }
        return result;
    }

    // Row and column sums
    Matrix<T> vertical_sum() const {
        Matrix<T> result(rows_, 1);
        for (std::size_t i = 0; i < rows_; ++i) {
            T sum = T{};
            for (std::size_t j = 0; j < cols_; ++j) {
                sum += (*this)(i, j);
            }
            result(i, 0) = sum;
        }
        return result;
    }

    Matrix<T> horizontal_sum() const {
        Matrix<T> result(1, cols_);
        for (std::size_t j = 0; j < cols_; ++j) {
            T sum = T{};
            for (std::size_t i = 0; i < rows_; ++i) {
                sum += (*this)(i, j);
            }
            result(0, j) = sum;
        }
        return result;
    }

    // Broadcasting
    Matrix<T> broadcast_vertical_sum(const Matrix<T>& other_col_vector) const {
        if (other_col_vector.cols() != 1 || other_col_vector.rows() != rows_) {
            throw std::invalid_argument("Column vector dimensions do not match for broadcasting addition");
        }
        Matrix<T> result(rows_, cols_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result(i, j) = (*this)(i, j) + other_col_vector(i, 0);
            }
        }
        return result;
    }

    Matrix<T>& broadcast_vertical_sum_inplace(const Matrix<T>& other_col_vector) {
        if (other_col_vector.cols() != 1 || other_col_vector.rows() != rows_) {
            throw std::invalid_argument("Column vector dimensions do not match for broadcasting addition");
        }
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                (*this)(i, j) += other_col_vector(i, 0);
            }
        }
        return *this;
    }

    Matrix<T> broadcast_horizontal_sum(const Matrix<T>& other_row_vector) const {
        if (other_row_vector.rows() != 1 || other_row_vector.cols() != cols_) {
            throw std::invalid_argument("Row vector dimensions do not match for broadcasting addition");
        }
        Matrix<T> result(rows_, cols_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result(i, j) = (*this)(i, j) + other_row_vector(0, j);
            }
        }
        return result;
    }

    Matrix<T>& broadcast_horizontal_sum_inplace(const Matrix<T>& other_row_vector) {
        if (other_row_vector.rows() != 1 || other_row_vector.cols() != cols_) {
            throw std::invalid_argument("Row vector dimensions do not match for broadcasting addition");
        }
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                (*this)(i, j) += other_row_vector(0, j);
            }
        }
        return *this;
    }

    // Element-wise multiplication
    Matrix<T> elementwise_multiply(const Matrix<T>& other) const {
        if (rows_ != other.rows() || cols_ != other.cols()) {
            throw std::invalid_argument("Matrix dimensions do not match for element-wise multiplication");
        }
        Matrix<T> result(rows_, cols_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result(i, j) = (*this)(i, j) * other(i, j);
            }
        }
        return result;
    }

    Matrix<T>& elementwise_multiply_inplace(const Matrix<T>& other) {
        if (rows_ != other.rows() || cols_ != other.cols()) {
            throw std::invalid_argument("Matrix dimensions do not match for element-wise multiplication");
        }
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                (*this)(i, j) *= other(i, j);
            }
        }
        return *this;
    }

    // Display (for debugging)
    Matrix<T> display() const {
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
        return *this;
    }

private:
    std::size_t rows_, cols_;
    std::vector<T> data_;
};

// Include MatMul after Matrix is defined to avoid circular dependency
#include "matmul.hpp"

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    return MatMul<T>::mm(*this, other);
}