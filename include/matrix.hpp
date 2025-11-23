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

        // Operators
        Matrix<T> operator*(const Matrix<T>& other) const;
        Matrix<T> operator+(const Matrix<T>& other) const;
        Matrix<T>& operator+=(const Matrix<T>& other);
        bool operator==(const Matrix<T>& other) const;
        bool operator!=(const Matrix<T>& other) const;


    private:
        std::size_t rows_, cols_;
        std::vector<T> data_;
};