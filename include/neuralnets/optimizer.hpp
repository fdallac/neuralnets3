/**
 * @file optimizer.hpp
 * @brief Optimization algorithms for neural network training
 * 
 * This file contains implementations of various optimization algorithms
 * used to update neural network weights and biases during training.
 */

#pragma once

#include "matrix/matrix.hpp"

/**
 * @brief Abstract base class for optimization algorithms
 * @tparam T Numeric type (float, double, etc.)
 * 
 * This class defines the interface that all optimizers must implement.
 * Optimizers are responsible for updating model parameters based on gradients.
 */
template<typename T>
class Optimizer {
    public:
        /**
         * @brief Update weight matrix using computed gradients
         * @param weights Reference to weight matrix to update (modified in-place)
         * @param weight_gradient Gradient of loss with respect to weights
         */
        virtual void update_weights(Matrix<T>& weights, const Matrix<T>& weight_gradient) = 0;
        
        /**
         * @brief Update bias vector using computed gradients
         * @param bias Reference to bias vector to update (modified in-place)
         * @param bias_gradient Gradient of loss with respect to biases
         */
        virtual void update_bias(Matrix<T>& bias, const Matrix<T>& bias_gradient) = 0;
        
        /**
         * @brief Virtual destructor
         */
        virtual ~Optimizer() = default;
};


/**
 * @brief Stochastic Gradient Descent (SGD) optimizer
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Implements basic SGD with fixed learning rate:
 * θ = θ - η * ∇L(θ)
 * 
 * where θ are the parameters, η is the learning rate, and ∇L(θ) is the gradient.
 */
template<typename T>
class SGD : public Optimizer<T> {
    public:
        /**
         * @brief Construct SGD optimizer with specified learning rate
         * @param learning_rate Step size for gradient descent (typically 0.001 - 0.1)
         */
        SGD(T learning_rate) : learning_rate(learning_rate) {}
        
        /**
         * @brief Get the current learning rate
         * @return Learning rate value
         */
        T get_learning_rate() const { return learning_rate; }

        /**
         * @brief Update weights using SGD rule
         * 
         * Applies: W = W - η * ∇W
         * 
         * @param weights Reference to weight matrix (modified in-place)
         * @param weight_gradient Gradient matrix with same dimensions as weights
         */
        void update_weights(Matrix<T>& weights, const Matrix<T>& weight_gradient) override {
            for (std::size_t i = 0; i < weights.rows(); ++i) {
                for (std::size_t j = 0; j < weights.cols(); ++j) {
                    weights(i, j) -= learning_rate * weight_gradient(i, j);
                }
            }
        }

        /**
         * @brief Update bias using SGD rule
         * 
         * Applies: b = b - η * ∇b
         * 
         * @param bias Reference to bias vector (modified in-place)
         * @param bias_gradient Gradient vector with same dimensions as bias
         */
        void update_bias(Matrix<T>& bias, const Matrix<T>& bias_gradient) override {
            for (std::size_t j = 0; j < bias.cols(); ++j) {
                bias(0, j) -= learning_rate * bias_gradient(0, j);
            }
        }

    private:
        T learning_rate; ///< Step size for gradient descent
};