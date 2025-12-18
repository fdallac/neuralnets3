/**
 * @file matrix.hpp
 * @brief Dense matrix class with linear algebra operations
 * 
 * This file contains the core Matrix class implementing:
 * - Basic matrix construction and element access
 * - Arithmetic operations (addition, multiplication, element-wise operations)
 * - Matrix operations (transpose, broadcasting, reduction)
 * - Multiple matrix multiplication backends
 */

#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>



/**
 * @brief Dense matrix class with row-major storage
 * @tparam T Numeric type (float, double, int, etc.)
 * 
 * Stores matrix data in row-major order (rows stored contiguously).
 * Provides standard linear algebra operations and integrates with
 * multiple matrix multiplication backends (see MatMul class).
 * 
 * @note Data is stored in a single contiguous std::vector<T>
 * @note Indexing uses operator()(i, j) for 0-based access
 */
template<typename T>
class Matrix {
public:
    // ================================================================================
    // Constructors
    // ================================================================================

    /**
     * @brief Default constructor - creates empty 0x0 matrix
     */
    Matrix()
        : rows_(0), cols_(0), data_() {}

    /**
     * @brief Construct zero-initialized matrix
     * @param i Number of rows
     * @param j Number of columns
     * 
     * All elements initialized to T{} (zero for numeric types).
     */
    Matrix(std::size_t i, std::size_t j)
        : rows_(i), cols_(j), data_(i * j, T{}) {}


    /**
     * @brief Construct matrix from existing data
     * @param i Number of rows
     * @param j Number of columns
     * @param data Vector containing matrix elements in row-major order
     * @throws std::invalid_argument if data.size() != i * j
     * 
     * Data should be laid out as: [row0_col0, row0_col1, ..., row1_col0, ...]
     */
    Matrix(std::size_t i, std::size_t j, const std::vector<T>& data)
        : rows_(i), cols_(j), data_(data) {
        if (data.size() != i * j) {
            throw std::invalid_argument("Data size does not match matrix dimensions");
        }
    }


    // =================================================================================
    // Getters
    // =================================================================================
    
    /** @brief Get number of rows */
    std::size_t rows() const noexcept { return rows_; }
    
    /** @brief Get number of columns */
    std::size_t cols() const noexcept { return cols_; }
    
    /** @brief Get const pointer to underlying data array */
    const T* data() const noexcept { return data_.data(); }
    
    /** @brief Get mutable pointer to underlying data array */
    T* data() noexcept { return data_.data(); }


    // =================================================================================
    // Operators and element access
    // =================================================================================

    /**
     * @brief Access element (unchecked)
     * @param i Row index
     * @param j Column index
     * @return Reference to element at (i, j)
     * @note No bounds checking - faster but unsafe
     */
    T& operator()(std::size_t i, std::size_t j) {
        return data_[i * cols_ + j];
    }

    /**
     * @brief Access element (unchecked, const)
     * @param i Row index
     * @param j Column index
     * @return Const reference to element at (i, j)
     */
    const T& operator()(std::size_t i, std::size_t j) const {
        return data_[i * cols_ + j];
    }

    /**
     * @brief Access element with bounds checking
     * @param i Row index
     * @param j Column index
     * @return Reference to element at (i, j)
     * @throws std::out_of_range if indices are out of bounds
     */
    T& at(std::size_t i, std::size_t j) {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data_[i * cols_ + j];
    }

    /**
     * @brief Access element with bounds checking (const)
     * @param i Row index
     * @param j Column index
     * @return Const reference to element at (i, j)
     * @throws std::out_of_range if indices are out of bounds
     */
    const T& at(std::size_t i, std::size_t j) const {
        if (i >= rows_ || j >= cols_) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data_[i * cols_ + j];
    }


    /**
     * @brief Matrix multiplication
     * @param other Right-hand matrix
     * @return Result matrix
     * @throws std::invalid_argument if dimensions incompatible (cols != other.rows)
     * 
     * Delegates to MatMul<T>::mm() which selects optimal implementation.
     * Defined below class definition.
     */
    Matrix<T> operator*(const Matrix<T>& other) const; // Defined below

    /**
     * @brief Element-wise matrix addition
     * @param other Matrix to add
     * @return New matrix with element-wise sum
     * @throws std::invalid_argument if dimensions don't match
     */
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

    /**
     * @brief In-place element-wise addition
     * @param other Matrix to add
     * @return Reference to this matrix
     * @throws std::invalid_argument if dimensions don't match
     */
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

    /**
     * @brief Test matrix equality
     * @param other Matrix to compare with
     * @return true if dimensions and all elements match
     */
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

    /** @brief Test matrix inequality */
    bool operator!=(const Matrix<T>& other) const {
        return !(*this == other);
    }

    // =================================================================================
    // Additional matrix arithmetic operations
    // =================================================================================

    /**
     * @brief Compute matrix transpose
     * @return Transposed matrix (cols x rows)
     * 
     * Swaps rows and columns: result(j, i) = this(i, j)
     */
    Matrix<T> transpose() const {
        Matrix<T> result(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }


    /**
     * @brief Sum across columns (row-wise sum)
     * @return Column vector (rows x 1) with sum of each row
     */
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

    /**
     * @brief Sum across rows (column-wise sum)
     * @return Row vector (1 x cols) with sum of each column
     * 
     * Used in backpropagation to aggregate gradients across batch.
     */
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

    /**
     * @brief Add column vector to each column
     * @param other_col_vector Column vector (rows x 1)
     * @return Matrix with column vector added to each column
     * @throws std::invalid_argument if dimensions don't match
     */
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

    // In-place broadcasting vertical sum
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

    /**
     * @brief Add row vector to each row (bias addition)
     * @param other_row_vector Row vector (1 x cols)
     * @return Matrix with row vector added to each row
     * @throws std::invalid_argument if dimensions don't match
     * 
     * Commonly used for adding bias: Z = X*W + b
     */
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

    /**
     * @brief Add row vector to each row in-place
     * @param other_row_vector Row vector (1 x cols)
     * @return Reference to this matrix
     * @throws std::invalid_argument if dimensions don't match
     */
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

    /**
     * @brief Element-wise (Hadamard) multiplication
     * @param other Matrix to multiply with
     * @return New matrix with element-wise product
     * @throws std::invalid_argument if dimensions don't match
     * 
     * Used in backpropagation: gradient ⊙ activation_derivative
     */
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

    // In-place element-wise multiplication
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


    // =================================================================================
    // Utility functions
    // =================================================================================

    /**
     * @brief Create deep copy of matrix
     * @return New matrix with same dimensions and values
     */
    Matrix<T> copy() const {
        Matrix<T> result(rows_, cols_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result(i, j) = (*this)(i, j);
            }
        }
        return result;
    }

    /**
     * @brief Apply function element-wise
     * @param func Function pointer taking T and returning T
     * @return New matrix with function applied to each element
     * 
     * Can also accept lambdas: matrix.apply([](T x) { return x*x; })
     */
    Matrix<T> apply(T (*func)(T)) const {
        Matrix<T> result(rows_, cols_); 
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result(i, j) = func((*this)(i, j));
            }
        }
        return result;
    }


    /**
     * @brief Fill matrix with ones
     */
    void fill_ones() {
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                (*this)(i, j) = static_cast<T>(1.0);
            }
        }
    }

    /**
     * @brief Fill matrix with uniform random values
     * @param min_val Minimum value (inclusive)
     * @param max_val Maximum value (inclusive)
     */
    void fill_uniform_noise(T min_val, T max_val) {
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                T random_val = min_val + static_cast<T>(rand()) / (static_cast<T>(RAND_MAX / (max_val - min_val)));
                (*this)(i, j) = random_val;
            }
        }
    }

    /**
     * @brief Fill matrix with Gaussian-like noise
     * @param mean Mean of distribution
     * @param stddev Standard deviation
     * @note Uses uniform approximation, not true Gaussian
     */
    void fill_gaussian_noise(T mean, T stddev) {
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                T random_val = static_cast<T>(2.0) * static_cast<T>(rand()) / static_cast<T>(RAND_MAX) - static_cast<T>(1.0);
                (*this)(i, j) = mean + random_val * stddev;
            }
        }
    }

    /**
     * @brief Print matrix to stdout (for debugging)
     * @return Reference to this matrix (for chaining)
     */
    Matrix<T> display() const {
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
        return *this;
    }


    /**
     * @brief Save matrix to CSV file
     * @param filename Path to output file
     */
    void save(const std::string& filename);

    /**
     * @brief Load matrix from CSV file
     * @param filename Path to input file
     */
    void load(const std::string& filename);


private:
    std::size_t rows_; ///< Number of rows
    std::size_t cols_; ///< Number of columns
    std::vector<T> data_; ///< Row-major data storage
};

// Include MatMul after Matrix is defined to avoid circular dependency
#include "matrix/matmul.hpp"

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    return MatMul<T>::mm(*this, other, MatMulMethod::Optimized);
}


#include "matrix/iohelper.hpp"

template<typename T>
void Matrix<T>::save(const std::string& filename) {
    IOHelper<T>::write_csv(*this, filename);
}

template<typename T>
void Matrix<T>::load(const std::string& filename) {
    Matrix<T> loaded = IOHelper<T>::read_csv(filename);
    *this = loaded;
}