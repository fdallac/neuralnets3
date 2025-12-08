#pragma once
#include "matrix.hpp"
#include <cmath>

template<typename T>
class Activation {
    public:
        virtual T forward(const Matrix<T>& M) = 0;
        virtual Matrix<T> backward(const Matrix<T>& M) = 0;
        virtual ~Activation() = default;
};


template<typename T>
class ReLU : public Activation<T> {
    public:
        static Matrix<T> forward(const Matrix<T>& M) {
            Matrix<T> result(M.rows(), M.cols());
            for (std::size_t i = 0; i < M.rows(); ++i) {
                for (std::size_t j = 0; j < M.cols(); ++j) {
                    result(i, j) = M(i, j) > T{} ? M(i, j) : T{};
                }
            }
            return result;
        }

        static Matrix<T> backward(const Matrix<T>& M) {
            Matrix<T> result(M.rows(), M.cols());
            for (std::size_t i = 0; i < M.rows(); ++i) {
                for (std::size_t j = 0; j < M.cols(); ++j) {
                    result(i, j) = M(i, j) > T{} ? T{1} : T{};
                }
            }
            return result;
        }
};


template<typename T>
class Sigmoid : public Activation<T> {
    public:
        static Matrix<T> forward(const Matrix<T>& M) {
            Matrix<T> result(M.rows(), M.cols());
            for (std::size_t i = 0; i < M.rows(); ++i) {
                for (std::size_t j = 0; j < M.cols(); ++j) {
                    result(i, j) = T{1} / (T{1} + std::exp(-M(i, j)));
                }
            }
            return result;
        }

        static Matrix<T> backward(const Matrix<T>& M) {
            Matrix<T> result(M.rows(), M.cols());
            for (std::size_t i = 0; i < M.rows(); ++i) {
                for (std::size_t j = 0; j < M.cols(); ++j) {
                    T sig = T{1} / (T{1} + std::exp(-M(i, j)));
                    result(i, j) = sig * (T{1} - sig);
                }
            }
            return result;
        }
};



template<typename T>
class Tanh : public Activation<T> {
    public:
        static Matrix<T> forward(const Matrix<T>& M) {
            Matrix<T> result(M.rows(), M.cols());
            for (std::size_t i = 0; i < M.rows(); ++i) {
                for (std::size_t j = 0; j < M.cols(); ++j) {
                    result(i, j) = std::tanh(M(i, j));
                }
            }
            return result;
        }

        static Matrix<T> backward(const Matrix<T>& M) {
            Matrix<T> result(M.rows(), M.cols());
            for (std::size_t i = 0; i < M.rows(); ++i) {
                for (std::size_t j = 0; j < M.cols(); ++j) {
                    T t = std::tanh(M(i, j));
                    result(i, j) = T{1} - t * t;
                }
            }
            return result;
        }
};
