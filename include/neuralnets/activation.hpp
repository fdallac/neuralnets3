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
         * @param Z Input matrix
         * @return Matrix with activation applied (A)
         */
        virtual Matrix<T> forward(const Matrix<T>& Z) = 0;
        
        /**
         * @brief Compute derivative of activation function
         * @param Z Input matrix
         * @param d_A Upstream gradient matrix
         * @return Gradient matrix to pass to previous layer (d_Z)
         */
        virtual Matrix<T> backward(const Matrix<T>& Z, const Matrix<T>& d_A) = 0; // FIXME: Generalize to accept non-diagonal Jacobian activations
        
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
         * @param Z Input matrix
         * @return Matrix with ReLU applied element-wise
         */
        Matrix<T> forward(const Matrix<T>& Z) {
            return Z.apply([](T x) { return x > T{} ? x : T{}; });
        }

        /**
         * @brief Compute ReLU derivative
         * @param Z Input matrix
         * @param d_A Upstream gradient matrix (not used here)
         * @return Matrix of derivatives (1 if x > 0, else 0)
         */
        Matrix<T> backward(const Matrix<T>& Z, const Matrix<T>& d_A) {
            Matrix<T> d_Z(Z.rows(), Z.cols());
            for (std::size_t i = 0; i < Z.rows(); ++i) {
                for (std::size_t j = 0; j < Z.cols(); ++j) {
                    d_Z(i, j) = Z(i, j) > T{} ? d_A(i, j) : T{};
                }
            }
            return d_Z;
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
         * @param Z Input matrix
         * @return Matrix with Leaky ReLU applied (alpha = 0.01)
         * @note Currently uses fixed alpha = 0.01
         */
        Matrix<T> forward(const Matrix<T>& Z) {
            return Z.apply([](T x) { return x > T{} ? x : static_cast<T>(0.01) * x; }); // FIXME: Use custom alpha
        }

        /**
         * @brief Compute Leaky ReLU derivative
         * @param Z Input matrix
         * @param d_A Upstream gradient matrix (not used here)
         * @return Matrix of derivatives (1 if x > 0, else 0.01)
         */
        Matrix<T> backward(const Matrix<T>& Z, const Matrix<T>& d_A) {
            Matrix<T> d_Z(Z.rows(), Z.cols());
            for (std::size_t i = 0; i < Z.rows(); ++i) {
                for (std::size_t j = 0; j < Z.cols(); ++j) {
                    d_Z(i, j) = Z(i, j) > T{} ? d_A(i, j) : static_cast<T>(0.01) * d_A(i, j);
                }
            }
            return d_Z;
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
         * @param Z Input matrix
         * @return Matrix with sigmoid applied (values in range (0, 1))
         */
        Matrix<T> forward(const Matrix<T>& Z) {
            return Z.apply([](T x) { return T{1} / (T{1} + std::exp(-x)); });
        }

        /**
         * @brief Compute sigmoid derivative
         * @param Z Input matrix
         * @param d_A Upstream gradient matrix (not used here)
         * @return Matrix of derivatives (σ(x) * (1 - σ(x)))
         */
        Matrix<T> backward(const Matrix<T>& Z, const Matrix<T>& d_A) {
            Matrix<T> S = forward(Z); // Sigmoid output
            Matrix<T> d_Z(Z.rows(), Z.cols());

            for (std::size_t i = 0; i < Z.rows(); ++i) {
                for (std::size_t j = 0; j < Z.cols(); ++j) {
                    d_Z(i, j) = d_A(i, j) * S(i, j) * (T{1} - S(i, j));
                }
            }
            return d_Z;
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
         * @param Z Input matrix
         * @return Matrix with tanh applied (values in range (-1, 1))
         */
        Matrix<T> forward(const Matrix<T>& Z) {
            return Z.apply([](T x) { return std::tanh(x); });
        }

        /**
         * @brief Compute tanh derivative
         * @param Z Input matrix
         * @param d_A Upstream gradient matrix
         * @return Matrix of derivatives (1 - tanh²(x))
         */
        Matrix<T> backward(const Matrix<T>& Z, const Matrix<T>& d_A) {
            Matrix<T> d_Z(Z.rows(), Z.cols());

            for (std::size_t i = 0; i < Z.rows(); ++i) {
                for (std::size_t j = 0; j < Z.cols(); ++j) {
                    T t = std::tanh(Z(i, j));
                    d_Z(i, j) = d_A(i, j) * (T{1} - t * t);
                }
            }
            return d_Z;
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
         * @param Z Input matrix
         * @return Copy of input matrix
         */
        Matrix<T> forward(const Matrix<T>& Z) {
            return Z.copy();
        }

        /**
         * @brief Compute linear activation derivative (always 1)
         * @param Z Input matrix
         * @param d_A Upstream gradient matrix
         * @return Copy of upstream gradient (unchanged)
         */
        Matrix<T> backward(const Matrix<T>& Z, const Matrix<T>& d_A) {
            return d_A.copy();
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
         * @param Z Input matrix (batch_size x num_classes)
         * @return Matrix of probability distributions (each row sums to 1)
         * @note Uses exp(x - max(x)) for numerical stability
         */
        Matrix<T> forward(const Matrix<T>& Z) {
            Matrix<T> result(Z.rows(), Z.cols());
            for (std::size_t i = 0; i < Z.rows(); ++i) {
                T max_val = Z(i, 0);
                for (std::size_t j = 1; j < Z.cols(); ++j) {
                    if (Z(i, j) > max_val) {
                        max_val = Z(i, j);
                    }
                }
                T sum_exp = T{};
                for (std::size_t j = 0; j < Z.cols(); ++j) {
                    result(i, j) = std::exp(Z(i, j) - max_val);
                    sum_exp += result(i, j);
                }
                for (std::size_t j = 0; j < Z.cols(); ++j) {
                    result(i, j) /= sum_exp;
                }
            }
            return result;
        }

        /**
         * @brief Compute softmax derivative (not implemented)
         * @param Z Input matrix
         * @param d_A Upstream gradient matrix
         * @return Not implemented - throws exception
         * @throws std::logic_error Always thrown - requires full Jacobian implementation
         * @note Softmax has non-diagonal Jacobian requiring generalized backward pass
         */
        Matrix<T> backward(const Matrix<T>& Z, const Matrix<T>& d_A) {
            // Recompute Softmax(Z) -> S
            Matrix<T> S = forward(Z);
            Matrix<T> d_Z(Z.rows(), Z.cols());

            // Compute Gradient: dZ_i = S_i * (dA_i - Sum(dA_k * S_k))
            for (size_t i = 0; i < Z.rows(); ++i) {
                
                // Calculate dot product (dA . S) for this row
                T dot = static_cast<T>(0);
                for (size_t k = 0; k < Z.cols(); ++k) {
                    dot += d_A(i, k) * S(i, k);
                }

                // Apply formula
                for (size_t j = 0; j < Z.cols(); ++j) {
                    d_Z(i, j) = S(i, j) * (d_A(i, j) - dot);
                }
            }
            return d_Z;
        }
};



