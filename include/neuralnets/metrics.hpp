#pragma once

#include "matrix/matrix.hpp"


template<typename T>
class Metrics {
    public:
        virtual double eval(const Matrix<T>& y_pred, const Matrix<T>& y_true) = 0;
        virtual double eval_probs(const Matrix<T>& y_pred, const Matrix<T>& y_true) {
            // Default implementation: threshold at 0.5 for binary classification
            Matrix<T> y_pred_labels = y_pred.apply([](T x) { return x >= 0.5 ? static_cast<T>(1) : static_cast<T>(0); });
            return eval(y_pred_labels, y_true);
        }
        virtual ~Metrics() = default;
};



template<typename T>
class Accuracy : public Metrics<T> {
    public:
        double eval (const Matrix<T>& y_pred, const Matrix<T>& y_true) override {
            if (y_pred.rows() != y_true.rows() || y_pred.cols() != y_true.cols()) {
                throw std::invalid_argument("Prediction and true label matrices must have the same dimensions for accuracy calculation");
            }
            std::size_t correct = 0;
            std::size_t total = y_true.rows();
            for (std::size_t i = 0; i < y_true.rows(); ++i) {
                // Assuming binary classification with threshold at 0.5
                T pred_label = y_pred(i, 0) >= static_cast<T>(0.5) ? static_cast<T>(1) : static_cast<T>(0);
                if (pred_label == y_true(i, 0)) {
                    ++correct;
                }
            }
            return static_cast<double>(correct) / static_cast<double>(total);
        }
};


template<typename T>
class MeanSquaredError : public Metrics<T> {
    public:
        double eval (const Matrix<T>& y_pred, const Matrix<T>& y_true) override {
            if (y_pred.rows() != y_true.rows() || y_pred.cols() != y_true.cols()) {
                throw std::invalid_argument("Prediction and true label matrices must have the same dimensions for MSE calculation");
            }
            double mse = 0.0;
            std::size_t n_samples = y_true.rows();
            std::size_t n_outputs = y_true.cols();
            for (std::size_t i = 0; i < n_samples; ++i) {
                for (std::size_t j = 0; j < n_outputs; ++j) {
                    double diff = static_cast<double>(y_pred(i, j)) - static_cast<double>(y_true(i, j));
                    mse += diff * diff;
                }
            }
            mse /= static_cast<double>(n_samples * n_outputs);
            return mse;
        }
};


template<typename T>
class MeanAbsoluteError : public Metrics<T> {
    public: 
        double eval (const Matrix<T>& y_pred, const Matrix<T>& y_true) override {
            if (y_pred.rows() != y_true.rows() || y_pred.cols() != y_true.cols()) {
                throw std::invalid_argument("Prediction and true label matrices must have the same dimensions for MAE calculation");
            }
            double mae = 0.0;
            std::size_t n_samples = y_true.rows();
            std::size_t n_outputs = y_true.cols();
            for (std::size_t i = 0; i < n_samples; ++i) {
                for (std::size_t j = 0; j < n_outputs; ++j) {
                    double diff = static_cast<double>(y_pred(i, j)) - static_cast<double>(y_true(i, j));
                    mae += std::abs(diff);
                }
            }
            mae /= static_cast<double>(n_samples * n_outputs);
            return mae;
        }
};


template<typename T>
class Precision : public Metrics<T> {
    public:
        double eval (const Matrix<T>& y_pred, const Matrix<T>& y_true) override {
            if (y_pred.rows() != y_true.rows() || y_pred.cols() != y_true.cols()) {
                throw std::invalid_argument("Prediction and true label matrices must have the same dimensions for precision calculation");
            }
            std::size_t true_positive = 0;
            std::size_t false_positive = 0;
            for (std::size_t i = 0; i < y_true.rows(); ++i) {
                T pred_label = y_pred(i, 0) >= static_cast<T>(0.5) ? static_cast<T>(1) : static_cast<T>(0);
                if (pred_label == static_cast<T>(1)) {
                    if (y_true(i, 0) == static_cast<T>(1)) {
                        ++true_positive;
                    } else {
                        ++false_positive;
                    }
                }
            }
            if (true_positive + false_positive == 0) {
                return 0.0; // Avoid division by zero
            }
            return static_cast<double>(true_positive) / static_cast<double>(true_positive + false_positive);
        }   
};


template<typename T>
class Recall : public Metrics<T> {
    public:
        double eval (const Matrix<T>& y_pred, const Matrix<T>& y_true) override {
            if (y_pred.rows() != y_true.rows() || y_pred.cols() != y_true.cols()) {
                throw std::invalid_argument("Prediction and true label matrices must have the same dimensions for recall calculation");
            }
            std::size_t true_positive = 0;
            std::size_t false_negative = 0;
            for (std::size_t i = 0; i < y_true.rows(); ++i) {
                T pred_label = y_pred(i, 0) >= static_cast<T>(0.5) ? static_cast<T>(1) : static_cast<T>(0);
                if (y_true(i, 0) == static_cast<T>(1)) {
                    if (pred_label == static_cast<T>(1)) {
                        ++true_positive;
                    } else {
                        ++false_negative;
                    }
                }
            }
            if (true_positive + false_negative == 0) {
                return 0.0; // Avoid division by zero
            }
            return static_cast<double>(true_positive) / static_cast<double>(true_positive + false_negative);
        }   
};