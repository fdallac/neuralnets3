#pragma once
#include "matrix.hpp"
#include <cmath>

template<typename T>
class Activation {
    public:
        virtual Matrix<T> forward(const Matrix<T>& M) = 0;
        virtual Matrix<T> backward(const Matrix<T>& M) = 0; // FIXME: Generalize to accept non-diagonal Jacobian activations
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
class LeakyReLU : public Activation<T> {
    // private:
    //     // T alpha;
    public:
        // LeakyReLU(T alpha = static_cast<T>(0.01)) : alpha(alpha) {}
                
        Matrix<T> forward(const Matrix<T>& M) {
            return M.apply([](T x) { return x > T{} ? x : static_cast<T>(0.01) * x; }); // FIXME: Use custom alpha
        }

        Matrix<T> backward(const Matrix<T>& M) {
            return M.apply([](T x) { return x > T{} ? T{1} : static_cast<T>(0.01); }); // FIXME: Use custom alpha
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


template<typename T>
class Softmax : public Activation<T> {
    public:
        Matrix<T> forward(const Matrix<T>& M) {
            Matrix<T> result(M.rows(), M.cols());
            for (std::size_t i = 0; i < M.rows(); ++i) {
                T max_val = M(i, 0);
                for (std::size_t j = 1; j < M.cols(); ++j) {
                    if (M(i, j) > max_val) {
                        max_val = M(i, j);
                    }
                }
                T sum_exp = T{};
                for (std::size_t j = 0; j < M.cols(); ++j) {
                    result(i, j) = std::exp(M(i, j) - max_val);
                    sum_exp += result(i, j);
                }
                for (std::size_t j = 0; j < M.cols(); ++j) {
                    result(i, j) /= sum_exp;
                }
            }
            return result;
        }

        Matrix<T> backward(const Matrix<T>& M) {
            // TODO: To be implemented after generalizing the activation backward pass
            throw std::logic_error("Softmax backward pass not implemented");
        }
};


