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
         * @param layer_id Identifier for the layer (used for stateful optimizers)
         */
        virtual void update_weights(Matrix<T>& weights, const Matrix<T>& weight_gradient, const std::string& layer_id = "") = 0;
        
        /**
         * @brief Update bias vector using computed gradients
         * @param bias Reference to bias vector to update (modified in-place)
         * @param bias_gradient Gradient of loss with respect to biases
         * @param layer_id Identifier for the layer (used for stateful optimizers)
         */
        virtual void update_bias(Matrix<T>& bias, const Matrix<T>& bias_gradient, const std::string& layer_id = "") = 0;
        
        /**
         * @brief Update normalization parameters if applicable
         * @param norm_params Reference to normalization parameters matrix
         * @param norm_gradient Gradient of loss with respect to normalization parameters
         * @param layer_id Identifier for the layer (used for stateful optimizers)
         * @note Some optimizers (like Adam) maintain internal state that
         */
        virtual void update_normalization_params(Matrix<T>& norm_params, const Matrix<T>& norm_gradient, const std::string& layer_id = "") = 0;

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
         * @brief Apply SGD update rule to a parameter matrix
         * @param parameter Reference to parameter matrix (modified in-place)
         * @param gradient Gradient matrix with same dimensions as parameter
         */
        void update(Matrix<T>& parameter, const Matrix<T>& gradient, const std::string& layer_id = "") {
            for (std::size_t i = 0; i < parameter.rows(); ++i) {
                for (std::size_t j = 0; j < parameter.cols(); ++j) {
                    parameter(i, j) -= learning_rate * gradient(i, j);
                }
            }
        }

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
        void update_weights(Matrix<T>& weights, const Matrix<T>& weight_gradient, const std::string& layer_id = "") override {
            update(weights, weight_gradient, layer_id);
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
        void update_bias(Matrix<T>& bias, const Matrix<T>& bias_gradient, const std::string& layer_id = "") override {
            update(bias, bias_gradient, layer_id);
        }

        /**
         * @brief Update normalization parameters using SGD rule
         * 
         * Applies: x = x - η * ∇x
         * 
         * @param norm_params Reference to normalization parameters matrix
         * @param norm_gradient Gradient matrix with same dimensions as norm_params
         * @param layer_id Identifier for the layer (used for stateful optimizers)
         */
        void update_normalization_params(Matrix<T>& norm_params, const Matrix<T>& norm_gradient, const std::string& layer_id = "") override {
            update(norm_params, norm_gradient, layer_id);
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
         * @brief Apply Adam update rule to a parameter matrix
         * @param parameter Reference to parameter matrix (modified in-place)
         * @param gradient Gradient matrix with same dimensions as parameter
         * @param momentum First moment estimate matrix
         * @param velocity Second moment estimate matrix
         * @param layer_id Identifier for the layer (used for stateful optimizers)
         */
        void update(Matrix<T>& parameter, const Matrix<T>& gradient, Matrix<T>& momentum, Matrix<T>& velocity, const std::string& layer_id = "") {
            for (std::size_t i = 0; i < parameter.rows(); ++i) {
                for (std::size_t j = 0; j < parameter.cols(); ++j) {
                    // Update biased first moment estimate
                    momentum(i, j) = beta1 * momentum(i, j) + (1 - beta1) * gradient(i, j);
                    // Update biased second moment estimate
                    velocity(i, j) = beta2 * velocity(i, j) + (1 - beta2) * gradient(i, j) * gradient(i, j);
                    
                    // Compute bias-corrected first and second moment estimates
                    T m_hat = momentum(i, j) / (1 - std::pow(beta1, ts));
                    T v_hat = velocity(i, j) / (1 - std::pow(beta2, ts));
                    
                    // Update parameters
                    parameter(i, j) -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
                }
            }
        }

        /**
         * @brief Update weights using Adam optimization algorithm
         * @param weights Reference to weight matrix (modified in-place)
         * @param weight_gradient Gradient matrix with same dimensions as weights
         * @param layer_id Identifier for the layer (used for stateful optimizers)
         */
        void update_weights(Matrix<T>& weights, const Matrix<T>& weight_gradient, const std::string& layer_id = "") override {
            auto& weight_momentum = this->momentum_map[&weights];
            auto& weight_velocity = this->velocity_map[&weights];

            if (weight_momentum.rows() == 0) {
                weight_momentum = Matrix<T>(weights.rows(), weights.cols());
                weight_velocity = Matrix<T>(weights.rows(), weights.cols());
            }

            update(weights, weight_gradient, weight_momentum, weight_velocity, layer_id);
        }

        /**
         * @brief Update bias using Adam optimization algorithm
         * @param bias Reference to bias vector (modified in-place)
         * @param bias_gradient Gradient vector with same dimensions as bias
         * @param layer_id Identifier for the layer (used for stateful optimizers)
         */
        void update_bias(Matrix<T>& bias, const Matrix<T>& bias_gradient, const std::string& layer_id = "") override {
            auto& bias_momentum = this->momentum_map[&bias];
            auto& bias_velocity = this->velocity_map[&bias];

            if (bias_momentum.rows() == 0) {
                bias_momentum = Matrix<T>(1, bias.cols());
                bias_velocity = Matrix<T>(1, bias.cols());
            }

            update(bias, bias_gradient, bias_momentum, bias_velocity, layer_id);
        }


        /**
         * @brief Update normalization parameters using Adam optimization algorithm
         * @param norm_params Reference to normalization parameters matrix
         * @param norm_gradient Gradient matrix with same dimensions as norm_params
         * @param layer_id Identifier for the layer (used for stateful optimizers)
         */
        void update_normalization_params(Matrix<T>& norm_params, const Matrix<T>& norm_gradient, const std::string& layer_id = "") override {
            auto& norm_momentum = this->momentum_map[&norm_params];
            auto& norm_velocity = this->velocity_map[&norm_params];

            if (norm_momentum.rows() == 0) {
                norm_momentum = Matrix<T>(norm_params.rows(), norm_params.cols());
                norm_velocity = Matrix<T>(norm_params.rows(), norm_params.cols());
            }

            update(norm_params, norm_gradient, norm_momentum, norm_velocity, layer_id);
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

        std::unordered_map<const void*, Matrix<T>> momentum_map, velocity_map; ///< Map to track moments per layer for weights, biases, and normalization parameters
};