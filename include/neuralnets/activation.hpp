/**
 * @file activation.hpp
 * @brief Activation functions for neural networks
 * 
 * This file contains implementations of various activation functions used in
 * neural network layers, including ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, and Linear.
 */

#pragma once

#include "matrix/matrix.hpp"
#include <cmath>

/**
 * @brief Abstract base class for activation functions
 * @tparam T Numeric type (float, double, etc.)
 * 
 * This class defines the interface that all activation functions must implement.
 * Activation functions compute both the forward activation and backward derivative.
 */
template<typename T>
class Activation {
    public:
        /**
         * @brief Apply activation function element-wise
         * @param M Input matrix
         * @return Matrix with activation applied
         */
        virtual Matrix<T> forward(const Matrix<T>& M) = 0;
        
        /**
         * @brief Compute derivative of activation function
         * @param M Input matrix
         * @return Matrix of derivatives (for diagonal Jacobian activations)
         * @note Currently assumes diagonal Jacobian (element-wise activations)
         */
        virtual Matrix<T> backward(const Matrix<T>& M) = 0; // FIXME: Generalize to accept non-diagonal Jacobian activations
        
        /**
         * @brief Virtual destructor
         */
        virtual ~Activation() = default;
};


/**
 * @brief Rectified Linear Unit (ReLU) activation function
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Applies the ReLU activation: f(x) = max(0, x)
 * Derivative: f'(x) = 1 if x > 0, else 0
 * 
 * @warning Can suffer from "dying ReLU" problem where neurons output zero for all inputs
 */
template<typename T>
class ReLU : public Activation<T> {
    public:
        /**
         * @brief Apply ReLU activation: max(0, x)
         * @param M Input matrix
         * @return Matrix with ReLU applied element-wise
         */
        Matrix<T> forward(const Matrix<T>& M) {
            return M.apply([](T x) { return x > T{} ? x : T{}; });
        }

        /**
         * @brief Compute ReLU derivative
         * @param M Input matrix
         * @return Matrix of derivatives (1 if x > 0, else 0)
         */
        Matrix<T> backward(const Matrix<T>& M) {
            return M.apply([](T x) { return x > T{} ? T{1} : T{}; });
        }
};



/**
 * @brief Leaky Rectified Linear Unit (LeakyReLU) activation function
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Applies the Leaky ReLU activation: f(x) = x if x > 0, else alpha * x
 * Derivative: f'(x) = 1 if x > 0, else alpha
 * 
 * Uses alpha = 0.01 to prevent dead neurons by allowing small negative gradients.
 * This solves the "dying ReLU" problem.
 */
template<typename T>
class LeakyReLU : public Activation<T> {
    // private:
    //     // T alpha;
    public:
        // LeakyReLU(T alpha = static_cast<T>(0.01)) : alpha(alpha) {}
        
        /**
         * @brief Apply Leaky ReLU activation
         * @param M Input matrix
         * @return Matrix with Leaky ReLU applied (alpha = 0.01)
         * @note Currently uses fixed alpha = 0.01
         */
        Matrix<T> forward(const Matrix<T>& M) {
            return M.apply([](T x) { return x > T{} ? x : static_cast<T>(0.01) * x; }); // FIXME: Use custom alpha
        }

        /**
         * @brief Compute Leaky ReLU derivative
         * @param M Input matrix
         * @return Matrix of derivatives (1 if x > 0, else 0.01)
         */
        Matrix<T> backward(const Matrix<T>& M) {
            return M.apply([](T x) { return x > T{} ? T{1} : static_cast<T>(0.01); }); // FIXME: Use custom alpha
        }
};


/**
 * @brief Sigmoid activation function
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Applies the sigmoid activation: f(x) = 1 / (1 + exp(-x))
 * Derivative: f'(x) = f(x) * (1 - f(x))
 * 
 * Output range: (0, 1). Commonly used for binary classification.
 * 
 * @warning Can suffer from vanishing gradients for large |x|
 */
template<typename T>
class Sigmoid : public Activation<T> {
    public:
        /**
         * @brief Apply sigmoid activation
         * @param M Input matrix
         * @return Matrix with sigmoid applied (values in range (0, 1))
         */
        Matrix<T> forward(const Matrix<T>& M) {
            return M.apply([](T x) { return T{1} / (T{1} + std::exp(-x)); });
        }

        /**
         * @brief Compute sigmoid derivative
         * @param M Input matrix
         * @return Matrix of derivatives (σ(x) * (1 - σ(x)))
         */
        Matrix<T> backward(const Matrix<T>& M) {
            return M.apply([](T x) { 
                T sig = T{1} / (T{1} + std::exp(-x));
                return sig * (T{1} - sig);
            });
        }
};



/**
 * @brief Hyperbolic Tangent (Tanh) activation function
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Applies the tanh activation: f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * Derivative: f'(x) = 1 - tanh²(x)
 * 
 * Output range: (-1, 1). Zero-centered unlike sigmoid.
 * 
 * @warning Can suffer from vanishing gradients for large |x|
 */
template<typename T>
class Tanh : public Activation<T> {
    public:
        /**
         * @brief Apply tanh activation
         * @param M Input matrix
         * @return Matrix with tanh applied (values in range (-1, 1))
         */
        Matrix<T> forward(const Matrix<T>& M) {
            return M.apply([](T x) { return std::tanh(x); });
        }

        /**
         * @brief Compute tanh derivative
         * @param M Input matrix
         * @return Matrix of derivatives (1 - tanh²(x))
         */
        Matrix<T> backward(const Matrix<T>& M) {
            return M.apply([](T x) { 
                T t = std::tanh(x);
                return T{1} - t * t;
            });
        }
};




/**
 * @brief Linear (Identity) activation function
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Applies the linear activation: f(x) = x
 * Derivative: f'(x) = 1
 * 
 * Used for regression tasks where unbounded output is needed.
 */
template<typename T>
class Linear : public Activation<T> {
    public:
        /**
         * @brief Apply linear activation (identity function)
         * @param M Input matrix
         * @return Copy of input matrix
         */
        Matrix<T> forward(const Matrix<T>& M) {
            return M.copy();
        }

        /**
         * @brief Compute linear activation derivative (always 1)
         * @param M Input matrix
         * @return Matrix of ones with same dimensions as input
         */
        Matrix<T> backward(const Matrix<T>& M) {
            Matrix<T> result(M.rows(), M.cols());
            result.fill_ones();
            return result;
        }
};


/**
 * @brief Softmax activation function
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Applies the softmax activation: f(x_i) = exp(x_i) / Σ exp(x_j)
 * Normalizes outputs to a probability distribution (sum to 1).
 * 
 * Applied row-wise (per sample). Uses numerical stability trick by
 * subtracting max value before exponentiation to avoid overflow.
 * 
 * Commonly used for multi-class classification output layers.
 * 
 * @warning Backward pass not yet implemented - requires full Jacobian
 */
template<typename T>
class Softmax : public Activation<T> {
    public:
        /**
         * @brief Apply softmax activation row-wise
         * @param M Input matrix (batch_size x num_classes)
         * @return Matrix of probability distributions (each row sums to 1)
         * @note Uses exp(x - max(x)) for numerical stability
         */
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

        /**
         * @brief Compute softmax derivative (not implemented)
         * @param M Input matrix
         * @return Not implemented - throws exception
         * @throws std::logic_error Always thrown - requires full Jacobian implementation
         * @note Softmax has non-diagonal Jacobian requiring generalized backward pass
         */
        Matrix<T> backward(const Matrix<T>& M) {
            // TODO: To be implemented after generalizing the activation backward pass
            throw std::logic_error("Softmax backward pass not implemented");
        }
};



