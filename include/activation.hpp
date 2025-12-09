#pragma once
#include "matrix.hpp"
#include <cmath>

template<typename T>
class Activation {
    public:
        virtual Matrix<T> forward(const Matrix<T>& M) = 0;
        virtual Matrix<T> backward(const Matrix<T>& M) = 0;
        virtual ~Activation() = default;
};


template<typename T>
class ReLU : public Activation<T> {
    public:
        Matrix<T> forward(const Matrix<T>& M) {
            return M.apply([](T x) { return x > T{} ? x : T{}; });
        }

        Matrix<T> backward(const Matrix<T>& M) {
            return M.apply([](T x) { return x > T{} ? T{1} : T{}; });
        }
};


template<typename T>
class Sigmoid : public Activation<T> {
    public:
        Matrix<T> forward(const Matrix<T>& M) {
            return M.apply([](T x) { return T{1} / (T{1} + std::exp(-x)); });
        }

        Matrix<T> backward(const Matrix<T>& M) {
            return M.apply([](T x) { 
                T sig = T{1} / (T{1} + std::exp(-x));
                return sig * (T{1} - sig);
            });
        }
};



template<typename T>
class Tanh : public Activation<T> {
    public:
        Matrix<T> forward(const Matrix<T>& M) {
            return M.apply([](T x) { return std::tanh(x); });
        }

        Matrix<T> backward(const Matrix<T>& M) {
            return M.apply([](T x) { 
                T t = std::tanh(x);
                return T{1} - t * t;
            });
        }
};  
