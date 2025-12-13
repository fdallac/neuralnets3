#pragma once

#include "matrix.hpp"
#include <cmath>
#include <stdexcept>

template<typename T>
class Loss {
    public:
        virtual T forward(const Matrix<T>& predictions, const Matrix<T>& targets) = 0;
        virtual Matrix<T> backward(const Matrix<T>& predictions, const Matrix<T>& targets) = 0;
        virtual ~Loss() = default;
};

template<typename T>
class MSELoss : public Loss<T> {
    public:
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
            return loss / static_cast<T>(n_samples);
        }  

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


template<typename T>
class BinaryCrossEntropyLoss : public Loss<T> {
    public:
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
            return loss / static_cast<T>(n_samples);
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



template<typename T>
class CategoricalCrossEntropyLoss : public Loss<T> {
    public:
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
    