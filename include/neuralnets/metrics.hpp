/**
 * @file metrics.hpp
 * @brief Evaluation metrics for model performance
 * 
 * Provides various metrics to evaluate classification and regression models,
 * including accuracy, precision, recall, and mean squared error.
 */

#pragma once

#include "matrix/matrix.hpp"


/**
 * @brief Abstract base class for evaluation metrics
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Defines interface for computing model performance metrics.
 * Supports both discrete labels and probabilistic outputs.
 */
template<typename T>
class Metrics {
    public:
        /**
         * @brief Evaluate metric on discrete predictions
         * @param y_pred Predicted labels
         * @param y_true True labels
         * @return Metric value (higher is better for accuracy-like metrics)
         */
        virtual double eval(const Matrix<T>& y_pred, const Matrix<T>& y_true) = 0;
        
        /**
         * @brief Evaluate metric on probability outputs
         * @param y_pred Predicted probabilities
         * @param y_true True labels
         * @return Metric value
         * 
         * Default implementation: threshold at 0.5 for binary classification.
         */
        virtual double eval_probs(const Matrix<T>& y_pred, const Matrix<T>& y_true) {
            // Default implementation: threshold at 0.5 for binary classification
            Matrix<T> y_pred_labels = y_pred.apply([](T x) { return x >= 0.5 ? static_cast<T>(1) : static_cast<T>(0); });
            return eval(y_pred_labels, y_true);
        }
        
        /**
         * @brief Virtual destructor
         */
        virtual ~Metrics() = default;
};



/**
 * @brief Accuracy metric for classification
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Computes fraction of correct predictions:
 * Accuracy = (# correct) / (# total)
 * 
 * Uses threshold of 0.5 for binary classification.
 */
template<typename T>
class Accuracy : public Metrics<T> {
    public:
        /**
         * @brief Compute classification accuracy
         * @param y_pred Predicted labels (or probabilities to be thresholded)
         * @param y_true True labels
         * @return Accuracy in range [0, 1]
         * @throws std::invalid_argument if dimensions don't match
         */
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


/**
 * @brief Mean Squared Error metric for regression
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Computes average squared difference:
 * MSE = (1/n) * \u03a3(y_pred - y_true)\u00b2
 * 
 * Lower values indicate better performance.
 */
template<typename T>
class MeanSquaredError : public Metrics<T> {
    public:
        /**
         * @brief Compute mean squared error
         * @param y_pred Predicted values
         * @param y_true True values
         * @return MSE value (lower is better)
         * @throws std::invalid_argument if dimensions don't match
         */
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


/**
 * @brief Mean Absolute Error metric for regression
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Computes average absolute difference:
 * MAE = (1/n) * Σ|y_pred - y_true|
 * 
 * More robust to outliers than MSE.
 * Lower values indicate better performance.
 */
template<typename T>
class MeanAbsoluteError : public Metrics<T> {
    public:
        /**
         * @brief Compute mean absolute error
         * @param y_pred Predicted values
         * @param y_true True values
         * @return MAE value (lower is better)
         * @throws std::invalid_argument if dimensions don't match
         */
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


/**
 * @brief Precision metric for binary classification
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Computes precision (positive predictive value):
 * Precision = TP / (TP + FP)
 * 
 * Measures fraction of positive predictions that are correct.
 * Uses threshold of 0.5 for classification.
 */
template<typename T>
class Precision : public Metrics<T> {
    public:
        /**
         * @brief Compute precision
         * @param y_pred Predicted labels (or probabilities to be thresholded)
         * @param y_true True labels
         * @return Precision in range [0, 1]
         * @throws std::invalid_argument if dimensions don't match
         * @note Returns 0.0 if no positive predictions
         */
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


/**
 * @brief Recall (Sensitivity) metric for binary classification
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Computes recall (true positive rate):
 * Recall = TP / (TP + FN)
 * 
 * Measures fraction of actual positives that are correctly identified.
 * Uses threshold of 0.5 for classification.
 */
template<typename T>
class Recall : public Metrics<T> {
    public:
        /**
         * @brief Compute recall
         * @param y_pred Predicted labels (or probabilities to be thresholded)
         * @param y_true True labels
         * @return Recall in range [0, 1]
         * @throws std::invalid_argument if dimensions don't match
         * @note Returns 0.0 if no actual positives
         */
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