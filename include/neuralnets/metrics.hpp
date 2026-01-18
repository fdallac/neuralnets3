/**
 * @file metrics.hpp
 * @brief Evaluation metrics for model performance
 * 
 * Provides various metrics to evaluate classification and regression models,
 * including accuracy, precision, recall, and mean squared error.
 * Supports both binary and multi-class classification.
 */

#pragma once

#include "matrix/matrix.hpp"
#include <vector>
#include <cmath>


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


// ============================================================================
// Binary Classification Metrics
// ============================================================================

/**
 * @brief Binary accuracy metric for classification
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Computes fraction of correct predictions for binary classification:
 * Accuracy = (# correct) / (# total)
 * 
 * Uses threshold of 0.5 for binary classification.
 */
template<typename T>
class BinaryAccuracy : public Metrics<T> {
    public:
        /**
         * @brief Compute binary classification accuracy
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

/// @brief Backward compatibility alias
template<typename T>
using Accuracy = BinaryAccuracy<T>;


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
 * @brief Binary precision metric for classification
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Computes precision (positive predictive value):
 * Precision = TP / (TP + FP)
 * 
 * Measures fraction of positive predictions that are correct.
 * Uses threshold of 0.5 for classification.
 */
template<typename T>
class BinaryPrecision : public Metrics<T> {
    public:
        /**
         * @brief Compute binary precision
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

/// @brief Backward compatibility alias
template<typename T>
using Precision = BinaryPrecision<T>;


/**
 * @brief Binary recall (Sensitivity) metric for classification
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Computes recall (true positive rate):
 * Recall = TP / (TP + FN)
 * 
 * Measures fraction of actual positives that are correctly identified.
 * Uses threshold of 0.5 for classification.
 */
template<typename T>
class BinaryRecall : public Metrics<T> {
    public:
        /**
         * @brief Compute binary recall
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

/// @brief Backward compatibility alias
template<typename T>
using Recall = BinaryRecall<T>;


// ============================================================================
// Multi-Class Classification Metrics
// ============================================================================

/**
 * @brief Helper function to get argmax index from a row
 * @tparam T Numeric type
 * @param mat Matrix to search
 * @param row Row index
 * @return Column index of maximum value
 */
template<typename T>
inline std::size_t argmax_row(const Matrix<T>& mat, std::size_t row) {
    std::size_t max_idx = 0;
    T max_val = mat(row, 0);
    for (std::size_t j = 1; j < mat.cols(); ++j) {
        if (mat(row, j) > max_val) {
            max_val = mat(row, j);
            max_idx = j;
        }
    }
    return max_idx;
}


/**
 * @brief Multi-class accuracy metric
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Computes fraction of correct predictions for multi-class classification.
 * Uses argmax to determine predicted class from probability distributions.
 * 
 * Accuracy = (# correct) / (# total)
 */
template<typename T>
class MultiClassAccuracy : public Metrics<T> {
    public:
        /**
         * @brief Compute multi-class accuracy from class indices
         * @param y_pred Predicted class indices (n_samples x 1)
         * @param y_true True class indices (n_samples x 1)
         * @return Accuracy in range [0, 1]
         * @throws std::invalid_argument if dimensions don't match
         */
        double eval(const Matrix<T>& y_pred, const Matrix<T>& y_true) override {
            if (y_pred.rows() != y_true.rows()) {
                throw std::invalid_argument("Number of samples must match for multi-class accuracy");
            }
            
            std::size_t correct = 0;
            std::size_t total = y_true.rows();
            
            for (std::size_t i = 0; i < total; ++i) {
                // Compare class indices directly
                if (static_cast<int>(y_pred(i, 0)) == static_cast<int>(y_true(i, 0))) {
                    ++correct;
                }
            }
            
            return static_cast<double>(correct) / static_cast<double>(total);
        }

        /**
         * @brief Compute multi-class accuracy from probability distributions
         * @param y_pred Predicted probabilities (n_samples x n_classes)
         * @param y_true One-hot encoded true labels (n_samples x n_classes)
         * @return Accuracy in range [0, 1]
         * @throws std::invalid_argument if dimensions don't match
         */
        double eval_probs(const Matrix<T>& y_pred, const Matrix<T>& y_true) override {
            if (y_pred.rows() != y_true.rows() || y_pred.cols() != y_true.cols()) {
                throw std::invalid_argument("Dimensions must match for multi-class accuracy");
            }
            
            std::size_t correct = 0;
            std::size_t total = y_pred.rows();
            
            for (std::size_t i = 0; i < total; ++i) {
                std::size_t pred_class = argmax_row(y_pred, i);
                std::size_t true_class = argmax_row(y_true, i);
                if (pred_class == true_class) {
                    ++correct;
                }
            }
            
            return static_cast<double>(correct) / static_cast<double>(total);
        }
};


/**
 * @brief Multi-class precision metric (macro-averaged)
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Computes macro-averaged precision across all classes.
 * For each class c: Precision_c = TP_c / (TP_c + FP_c)
 * Macro precision = mean of all class precisions
 */
template<typename T>
class MultiClassPrecision : public Metrics<T> {
    public:
        /**
         * @brief Compute multi-class precision from class indices
         * @param y_pred Predicted class indices (n_samples x 1)
         * @param y_true True class indices (n_samples x 1)
         * @return Macro-averaged precision in range [0, 1]
         */
        double eval(const Matrix<T>& y_pred, const Matrix<T>& y_true) override {
            if (y_pred.rows() != y_true.rows()) {
                throw std::invalid_argument("Number of samples must match for multi-class precision");
            }
            
            // Find number of classes
            int max_class = 0;
            for (std::size_t i = 0; i < y_true.rows(); ++i) {
                max_class = std::max(max_class, static_cast<int>(y_true(i, 0)));
                max_class = std::max(max_class, static_cast<int>(y_pred(i, 0)));
            }
            std::size_t num_classes = static_cast<std::size_t>(max_class + 1);
            
            // Count TP and FP for each class
            std::vector<std::size_t> true_positives(num_classes, 0);
            std::vector<std::size_t> false_positives(num_classes, 0);
            
            for (std::size_t i = 0; i < y_true.rows(); ++i) {
                std::size_t pred_class = static_cast<std::size_t>(y_pred(i, 0));
                std::size_t true_class = static_cast<std::size_t>(y_true(i, 0));
                
                if (pred_class == true_class) {
                    true_positives[pred_class]++;
                } else {
                    false_positives[pred_class]++;
                }
            }
            
            // Compute macro-averaged precision
            double total_precision = 0.0;
            std::size_t valid_classes = 0;
            
            for (std::size_t c = 0; c < num_classes; ++c) {
                std::size_t total_predicted = true_positives[c] + false_positives[c];
                if (total_predicted > 0) {
                    total_precision += static_cast<double>(true_positives[c]) / static_cast<double>(total_predicted);
                    ++valid_classes;
                }
            }
            
            return valid_classes > 0 ? total_precision / static_cast<double>(valid_classes) : 0.0;
        }

        /**
         * @brief Compute multi-class precision from probability distributions
         * @param y_pred Predicted probabilities (n_samples x n_classes)
         * @param y_true One-hot encoded true labels (n_samples x n_classes)
         * @return Macro-averaged precision in range [0, 1]
         */
        double eval_probs(const Matrix<T>& y_pred, const Matrix<T>& y_true) override {
            if (y_pred.rows() != y_true.rows() || y_pred.cols() != y_true.cols()) {
                throw std::invalid_argument("Dimensions must match for multi-class precision");
            }
            
            std::size_t num_classes = y_pred.cols();
            std::vector<std::size_t> true_positives(num_classes, 0);
            std::vector<std::size_t> false_positives(num_classes, 0);
            
            for (std::size_t i = 0; i < y_pred.rows(); ++i) {
                std::size_t pred_class = argmax_row(y_pred, i);
                std::size_t true_class = argmax_row(y_true, i);
                
                if (pred_class == true_class) {
                    true_positives[pred_class]++;
                } else {
                    false_positives[pred_class]++;
                }
            }
            
            // Compute macro-averaged precision
            double total_precision = 0.0;
            std::size_t valid_classes = 0;
            
            for (std::size_t c = 0; c < num_classes; ++c) {
                std::size_t total_predicted = true_positives[c] + false_positives[c];
                if (total_predicted > 0) {
                    total_precision += static_cast<double>(true_positives[c]) / static_cast<double>(total_predicted);
                    ++valid_classes;
                }
            }
            
            return valid_classes > 0 ? total_precision / static_cast<double>(valid_classes) : 0.0;
        }
};


/**
 * @brief Multi-class recall metric (macro-averaged)
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Computes macro-averaged recall across all classes.
 * For each class c: Recall_c = TP_c / (TP_c + FN_c)
 * Macro recall = mean of all class recalls
 */
template<typename T>
class MultiClassRecall : public Metrics<T> {
    public:
        /**
         * @brief Compute multi-class recall from class indices
         * @param y_pred Predicted class indices (n_samples x 1)
         * @param y_true True class indices (n_samples x 1)
         * @return Macro-averaged recall in range [0, 1]
         */
        double eval(const Matrix<T>& y_pred, const Matrix<T>& y_true) override {
            if (y_pred.rows() != y_true.rows()) {
                throw std::invalid_argument("Number of samples must match for multi-class recall");
            }
            
            // Find number of classes
            int max_class = 0;
            for (std::size_t i = 0; i < y_true.rows(); ++i) {
                max_class = std::max(max_class, static_cast<int>(y_true(i, 0)));
                max_class = std::max(max_class, static_cast<int>(y_pred(i, 0)));
            }
            std::size_t num_classes = static_cast<std::size_t>(max_class + 1);
            
            // Count TP and FN for each class
            std::vector<std::size_t> true_positives(num_classes, 0);
            std::vector<std::size_t> false_negatives(num_classes, 0);
            
            for (std::size_t i = 0; i < y_true.rows(); ++i) {
                std::size_t pred_class = static_cast<std::size_t>(y_pred(i, 0));
                std::size_t true_class = static_cast<std::size_t>(y_true(i, 0));
                
                if (pred_class == true_class) {
                    true_positives[true_class]++;
                } else {
                    false_negatives[true_class]++;
                }
            }
            
            // Compute macro-averaged recall
            double total_recall = 0.0;
            std::size_t valid_classes = 0;
            
            for (std::size_t c = 0; c < num_classes; ++c) {
                std::size_t total_actual = true_positives[c] + false_negatives[c];
                if (total_actual > 0) {
                    total_recall += static_cast<double>(true_positives[c]) / static_cast<double>(total_actual);
                    ++valid_classes;
                }
            }
            
            return valid_classes > 0 ? total_recall / static_cast<double>(valid_classes) : 0.0;
        }

        /**
         * @brief Compute multi-class recall from probability distributions
         * @param y_pred Predicted probabilities (n_samples x n_classes)
         * @param y_true One-hot encoded true labels (n_samples x n_classes)
         * @return Macro-averaged recall in range [0, 1]
         */
        double eval_probs(const Matrix<T>& y_pred, const Matrix<T>& y_true) override {
            if (y_pred.rows() != y_true.rows() || y_pred.cols() != y_true.cols()) {
                throw std::invalid_argument("Dimensions must match for multi-class recall");
            }
            
            std::size_t num_classes = y_pred.cols();
            std::vector<std::size_t> true_positives(num_classes, 0);
            std::vector<std::size_t> false_negatives(num_classes, 0);
            
            for (std::size_t i = 0; i < y_pred.rows(); ++i) {
                std::size_t pred_class = argmax_row(y_pred, i);
                std::size_t true_class = argmax_row(y_true, i);
                
                if (pred_class == true_class) {
                    true_positives[true_class]++;
                } else {
                    false_negatives[true_class]++;
                }
            }
            
            // Compute macro-averaged recall
            double total_recall = 0.0;
            std::size_t valid_classes = 0;
            
            for (std::size_t c = 0; c < num_classes; ++c) {
                std::size_t total_actual = true_positives[c] + false_negatives[c];
                if (total_actual > 0) {
                    total_recall += static_cast<double>(true_positives[c]) / static_cast<double>(total_actual);
                    ++valid_classes;
                }
            }
            
            return valid_classes > 0 ? total_recall / static_cast<double>(valid_classes) : 0.0;
        }
};