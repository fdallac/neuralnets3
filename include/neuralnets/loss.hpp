/**
 * @file loss.hpp
 * @brief Loss functions for neural network training
 * 
 * This file contains implementations of various loss functions used in
 * neural network training, including their forward computation and
 * backward gradient calculation.
 */

#pragma once

#include "matrix/matrix.hpp"
#include <cmath>
#include <stdexcept>

/**
 * @brief Abstract base class for loss functions
 * @tparam T Numeric type (float, double, etc.)
 * 
 * This class defines the interface that all loss functions must implement.
 * Loss functions compute both the forward loss value and backward gradients.
 */
template<typename T>
class Loss {
    public:
        /**
         * @brief Compute the forward pass of the loss function
         * @param predictions Matrix of predicted values from the model
         * @param targets Matrix of ground truth target values
         * @return Scalar loss value
         */
        virtual T forward(const Matrix<T>& predictions, const Matrix<T>& targets) = 0;
        
        /**
         * @brief Compute the backward pass (gradient) of the loss function
         * @param predictions Matrix of predicted values from the model
         * @param targets Matrix of ground truth target values
         * @return Matrix of gradients with respect to predictions
         */
        virtual Matrix<T> backward(const Matrix<T>& predictions, const Matrix<T>& targets) = 0;
        
        /**
         * @brief Virtual destructor
         */
        virtual ~Loss() = default;
};

/**
 * @brief Mean Squared Error (MSE) loss function
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Computes the mean squared error between predictions and targets:
 * MSE = (1/n) * Σ(predictions - targets)²
 * 
 * Commonly used for regression tasks.
 */
template<typename T>
class MSELoss : public Loss<T> {
    public:
        /**
         * @brief Compute MSE loss
         * @param predictions Matrix of predicted values
         * @param targets Matrix of target values
         * @return Mean squared error
         * @throws std::invalid_argument if dimensions don't match
         */
        T forward(const Matrix<T>& predictions, const Matrix<T>& targets) override {
            if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
                throw std::invalid_argument("Dimensions of predictions and targets do not match for MSE loss");
            }
            T loss = T{};
            std::size_t n_samples = predictions.rows();
            std::size_t n_outputs = predictions.cols();
            for (std::size_t i = 0; i < n_samples; ++i) {
                for (std::size_t j = 0; j < n_outputs; ++j) {
                    T diff = predictions(i, j) - targets(i, j);
                    loss += diff * diff;
                }
            }
            return loss / static_cast<T>(n_samples) / static_cast<T>(n_outputs);
        }  

        /**
         * @brief Compute gradient of MSE loss
         * 
         * Gradient: ∂L/∂pred = 2(predictions - targets) / n_samples
         * 
         * @param predictions Matrix of predicted values
         * @param targets Matrix of target values
         * @return Matrix of gradients
         * @throws std::invalid_argument if dimensions don't match
         */
        Matrix<T> backward(const Matrix<T>& predictions, const Matrix<T>& targets) override {
            if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
                throw std::invalid_argument("Dimensions of predictions and targets do not match for MSE loss backward");
            }
            std::size_t n_samples = predictions.rows();
            std::size_t n_outputs = predictions.cols();
            Matrix<T> grad(predictions.rows(), predictions.cols());
            for (std::size_t i = 0; i < n_samples; ++i) {
                for (std::size_t j = 0; j < n_outputs; ++j) {
                    grad(i, j) = T{2} * (predictions(i, j) - targets(i, j)) / static_cast<T>(n_samples);
                }
            }
            return grad;
        }
};


/**
 * @brief Binary Cross-Entropy loss function
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Computes binary cross-entropy loss for binary classification:
 * BCE = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]
 * 
 * where y is the target (0 or 1) and p is the predicted probability.
 * Uses epsilon (1e-12) to avoid log(0).
 */
template<typename T>
class BinaryCrossEntropyLoss : public Loss<T> {
    public:
        /**
         * @brief Compute binary cross-entropy loss
         * @param predictions Matrix of predicted probabilities (0 to 1)
         * @param targets Matrix of binary target values (0 or 1)
         * @return Binary cross-entropy loss value
         * @throws std::invalid_argument if dimensions don't match
         */
        T forward(const Matrix<T>& predictions, const Matrix<T>& targets) override  {
            if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
                throw std::invalid_argument("Dimensions of predictions and targets do not match for Binary Cross-Entropy loss");
            }
            T loss = T{};
            std::size_t n_samples = predictions.rows();
            std::size_t n_outputs = predictions.cols();
            for (std::size_t i = 0; i < n_samples; ++i) {
                for (std::size_t j = 0; j < n_outputs; ++j) {
                    T pred = predictions(i, j);
                    T target = targets(i, j);
                    // Avoid log(0) by adding a small epsilon
                    T epsilon = static_cast<T>(1e-12);
                    loss -= target * std::log(pred + epsilon) + (T{1} - target) * std::log(T{1} - pred + epsilon);
                }
            }
            return loss / static_cast<T>(n_samples) / static_cast<T>(n_outputs);
        }

        Matrix<T> backward(const Matrix<T>& predictions, const Matrix<T>& targets) override  {
            if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
                throw std::invalid_argument("Dimensions of predictions and targets do not match for Binary Cross-Entropy loss backward");
            }
            std::size_t n_samples = predictions.rows();
            std::size_t n_outputs = predictions.cols();
            Matrix<T> grad(predictions.rows(), predictions.cols());
            T epsilon = static_cast<T>(1e-12);
            for (std::size_t i = 0; i < n_samples; ++i) {
                for (std::size_t j = 0; j < n_outputs; ++j) {
                    T pred = predictions(i, j);
                    T target = targets(i, j);
                    grad(i, j) = -(target / (pred + epsilon) - (T{1} - target) / (T{1} - pred + epsilon)) / static_cast<T>(n_samples);
                }
            }
            return grad;
        }
};



/**
 * @brief Categorical Cross-Entropy loss function
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Computes categorical cross-entropy loss for multi-class classification:
 * CCE = -(1/n) * Σ Σ y_ij * log(p_ij)
 * 
 * where y is the one-hot encoded target and p is the predicted probability distribution.
 * Uses epsilon (1e-12) to avoid log(0).
 */
template<typename T>
class CategoricalCrossEntropyLoss : public Loss<T> {
    public:
        /**
         * @brief Compute categorical cross-entropy loss
         * @param predictions Matrix of predicted probabilities (should sum to 1 per sample)
         * @param targets Matrix of one-hot encoded target values
         * @return Categorical cross-entropy loss value
         * @throws std::invalid_argument if dimensions don't match
         */
        T forward(const Matrix<T>& predictions, const Matrix<T>& targets) override  {
            if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
                throw std::invalid_argument("Dimensions of predictions and targets do not match for Cross-Entropy loss");
            }
            T loss = T{};
            std::size_t n_samples = predictions.rows();
            std::size_t n_outputs = predictions.cols();
            for (std::size_t i = 0; i < n_samples; ++i) {
                for (std::size_t j = 0; j < n_outputs; ++j) {
                    T pred = predictions(i, j);
                    T target = targets(i, j);
                    // Avoid log(0) by adding a small epsilon
                    T epsilon = static_cast<T>(1e-12);
                    loss -= target * std::log(pred + epsilon);
                }
            }
            return loss / static_cast<T>(n_samples);
        }

        /**
         * @brief Compute gradient of categorical cross-entropy loss
         * 
         * Gradient: ∂L/∂p = -y/p / n_samples
         * 
         * @param predictions Matrix of predicted probabilities
         * @param targets Matrix of one-hot encoded target values
         * @return Matrix of gradients
         * @throws std::invalid_argument if dimensions don't match
         */
        Matrix<T> backward(const Matrix<T>& predictions, const Matrix<T>& targets) override  {
            if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
                throw std::invalid_argument("Dimensions of predictions and targets do not match for Cross-Entropy loss backward");
            }
            std::size_t n_samples = predictions.rows();
            std::size_t n_outputs = predictions.cols();
            Matrix<T> grad(predictions.rows(), predictions.cols());
            T epsilon = static_cast<T>(1e-12);
            for (std::size_t i = 0; i < n_samples; ++i) {
                for (std::size_t j = 0; j < n_outputs; ++j) {
                    T pred = predictions(i, j);
                    T target = targets(i, j);
                    grad(i, j) = -target / (pred + epsilon) / static_cast<T>(n_samples);
                }
            }
            return grad;
        }
};