/**
 * @file optimizer.hpp
 * @brief Optimization algorithms for neural network training
 * 
 * This file contains implementations of various optimization algorithms
 * used to update neural network weights and biases during training.
 */

#pragma once

#include "matrix/matrix.hpp"
#include <cmath>
#include <unordered_map>

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
         * @brief Advance the optimizer to the next step
         * @note Some optimizers (like ADAM) use timestep for bias correction
         */
        virtual void next_step() {}

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
         * @param weight_momentum Momentum matrix for weights (unused in SGD)
         * @param weight_velocity Velocity matrix for weights (unused in SGD)
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
         * @param bias_momentum Momentum vector for biases (unused in SGD)
         * @param bias_velocity Velocity vector for biases (unused in SGD)
         */
        void update_bias(Matrix<T>& bias, const Matrix<T>& bias_gradient) override {
            for (std::size_t j = 0; j < bias.cols(); ++j) {
                bias(0, j) -= learning_rate * bias_gradient(0, j);
            }
        }

    private:
        T learning_rate; ///< Step size for gradient descent
};



/**
 * @brief Adam optimizer
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Implements the Adam optimization algorithm which combines
 * momentum and adaptive learning rates for each parameter.
 */
template<typename T>
class Adam : public Optimizer<T> {
    public:
        /**
         * @brief Construct Adam optimizer with specified parameters
         * @param learning_rate Initial learning rate (typically 0.001)
         * @param beta1 Exponential decay rate for first moment estimates (default 0.9)
         * @param beta2 Exponential decay rate for second moment estimates (default 0.999)
         * @param epsilon Small constant to prevent division by zero (default 1e-8)
         */
        Adam(T learning_rate, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
            : learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), ts(1) {}

        /**
         * @brief Get the current learning rate
         * @return Learning rate value
         */
        T get_learning_rate() const { return learning_rate; }

        /**
         * @brief Update weights using Adam optimization algorithm
         * @param weights Reference to weight matrix (modified in-place)
         * @param weight_gradient Gradient matrix with same dimensions as weights
         */
        void update_weights(Matrix<T>& weights, const Matrix<T>& weight_gradient) override {
            auto& weight_momentum = this->mw[&weights];
            auto& weight_velocity = this->vw[&weights];

            if (weight_momentum.rows() == 0) {
                weight_momentum = Matrix<T>(weights.rows(), weights.cols());
                weight_velocity = Matrix<T>(weights.rows(), weights.cols());
            }

            for (std::size_t i = 0; i < weights.rows(); ++i) {
                for (std::size_t j = 0; j < weights.cols(); ++j) {
                    // Update biased first moment estimate
                    weight_momentum(i, j) = beta1 * weight_momentum(i, j) + (1 - beta1) * weight_gradient(i, j);
                    // Update biased second moment estimate
                    weight_velocity(i, j) = beta2 * weight_velocity(i, j) + (1 - beta2) * weight_gradient(i, j) * weight_gradient(i, j);
                    
                    // Compute bias-corrected first and second moment estimates
                    T m_hat = weight_momentum(i, j) / (1 - std::pow(beta1, ts));
                    T v_hat = weight_velocity(i, j) / (1 - std::pow(beta2, ts));
                    
                    // Update parameters
                    weights(i, j) -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
                }
            }
        }

        /**
         * @brief Update bias using Adam optimization algorithm
         * @param bias Reference to bias vector (modified in-place)
         * @param bias_gradient Gradient vector with same dimensions as bias
         */
        void update_bias(Matrix<T>& bias, const Matrix<T>& bias_gradient) override {
            auto& bias_momentum = this->mb[&bias];
            auto& bias_velocity = this->vb[&bias];

            if (bias_momentum.rows() == 0) {
                bias_momentum = Matrix<T>(1, bias.cols());
                bias_velocity = Matrix<T>(1, bias.cols());
            }

            for (std::size_t j = 0; j < bias.cols(); ++j) {
                // Update biased first moment estimate
                bias_momentum(0, j) = beta1 * bias_momentum(0, j) + (1 - beta1) * bias_gradient(0, j);
                // Update biased second moment estimate
                bias_velocity(0, j) = beta2 * bias_velocity(0, j) + (1 - beta2) * bias_gradient(0, j) * bias_gradient(0, j);

                // Compute bias-corrected first and second moment estimates
                T m_hat = bias_momentum(0, j) / (1 - std::pow(beta1, ts));
                T v_hat = bias_velocity(0, j) / (1 - std::pow(beta2, ts));
                // Update parameters
                bias(0, j) -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }

        void next_step() override {
            // Increment timestep
            ts++;
        }

    private:
        T learning_rate;
        T beta1;
        T beta2;
        T epsilon;
        int ts;  ///< Timestep counter for parameter updates

        std::unordered_map<const void*, Matrix<T>> mw, mb, vw, vb; ///< Map to track moments per layer for weights and biases
};