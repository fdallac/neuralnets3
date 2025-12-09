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
Matrix<T> Matrix<T>::broadcast_vertical_sum(const Matrix<T>& other_col_vector) const {
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


template<typename T>
Matrix<T>& Matrix<T>::broadcast_vertical_sum_inplace(const Matrix<T>& other_col_vector) {
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


template<typename T>
Matrix<T> Matrix<T>::broadcast_horizontal_sum(const Matrix<T>& other_row_vector) const {
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



template<typename T>
Matrix<T>& Matrix<T>::broadcast_horizontal_sum_inplace(const Matrix<T>& other_row_vector) {
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



template<typename T>
Matrix<T> Matrix<T>::elementwise_multiply(const Matrix<T>& other) const {
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


template<typename T>
Matrix<T>& Matrix<T>::elementwise_multiply_inplace(const Matrix<T>& other) {
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



template<typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result(cols_, rows_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}


template<typename T>
Matrix<T> Matrix<T>::apply(T (*func)(T)) const {
    Matrix<T> result(rows_, cols_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result(i, j) = func((*this)(i, j));
        }
    }
    return result;
}


template<typename T>
Matrix<T> Matrix<T>::vertical_sum() const {
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


template<typename T>
Matrix<T> Matrix<T>::horizontal_sum() const { 
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



// Explicit template instantiation for common types
template class Matrix<int>;
template class Matrix<double>;
template class Matrix<float>;