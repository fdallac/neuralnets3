# pragma once

#include <vector>
#include <stdexcept>

template<typename T>
class Matrix {
    public:
        // Default constructor: empty matrix
        Matrix();

        // Constructor: i (rows) x j (cols) matrix initialized to zero
        Matrix(std::size_t i, std::size_t j);

        // Constructor: i (rows) x j (cols) matrix initialized with data
        Matrix(std::size_t i, std::size_t j, const std::vector<T>& data);

        // Element access (unchecked)
        T& operator()(std::size_t i, std::size_t j);
        const T& operator()(std::size_t i, std::size_t j) const;
        // Element access (checked)
        T& at(std::size_t i, std::size_t j);
        const T& at(std::size_t i, std::size_t j) const;

        // Getters
        std::size_t rows() const noexcept { return rows_; }
        std::size_t cols() const noexcept { return cols_; }
        const T* data() const noexcept { return data_.data(); }
        T* data() noexcept { return data_.data(); }

        // Operators
        Matrix<T> operator*(const Matrix<T>& other) const; // Matrix multiplication
        Matrix<T> operator+(const Matrix<T>& other) const; // Matrix addition
        Matrix<T>& operator+=(const Matrix<T>& other); // In-place addition
        // Matrix<T> operator-(const Matrix<T>& other) const; // Matrix subtraction
        // Matrix<T>& operator-=(const Matrix<T>& other); // In-place subtraction
        // Matrix<T> operator*(T scalar) const; // Scalar multiplication
        // Matrix<T>& operator*=(T scalar); // In-place scalar multiplication
        // Matrix<T> operator/(T scalar) const; // Scalar division
        // Matrix<T>& operator/=(T scalar); // In-place scalar division
        Matrix<T> operator+|(const Matrix<T>& other_col_vector) const; // Matrix addition with a column vector broadcast
        Matrix<T>& operator+|=(const Matrix<T>& other_col_vector); // In-place matrix addition with column vector broadcast
        Matrix<T> operator+_(const Matrix<T>& other_row_vector) const; // Matrix addition with a row vector broadcast
        Matrix<T>& operator+_=(const Matrix<T>& other_row_vector); // In-place matrix addition with row vector broadcast
        Matrix<T> operator*.(const Matrix<T>& other) const; // Element-wise multiplication
        Matrix<T>& operator*.=(const Matrix<T>& other); // In-place element-wise multiplication
        
        bool operator==(const Matrix<T>& other) const; // Equality check
        bool operator!=(const Matrix<T>& other) const; // Inequality check

        // Transpose
        Matrix<T> transpose() const;

        // Apply function element-wise
        Matrix<T> apply(T (*func)(T)) const;


    private:
        std::size_t rows_, cols_;
        std::vector<T> data_;
};