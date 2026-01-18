/**
 * @file preprocessing.hpp
 * @brief Data preprocessing utilities for neural networks
 * 
 * This file contains preprocessing classes following sklearn-style API
 */

#pragma once

#include "matrix/matrix.hpp"
#include <vector>
#include <map>
#include <stdexcept>
#include <algorithm>


/**
 * @brief One-Hot Encoder for categorical variables (sklearn-style API)
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Converts integer class labels to one-hot encoded vectors.
 * Supports fit/transform pattern for consistent encoding across train/test sets.
 * 
 * Example:
 * @code
 * OneHotEncoder<double> encoder;
 * Matrix<double> y_encoded = encoder.fit_transform(y_train);
 * Matrix<double> y_test_encoded = encoder.transform(y_test);
 * @endcode
 */
template<typename T>
class OneHotEncoder {
    private:
        std::vector<T> classes_;           ///< Unique class values (sorted)
        std::map<T, std::size_t> class_to_index_;  ///< Mapping from class value to index
        bool fitted_ = false;              ///< Whether the encoder has been fitted
        std::size_t num_classes_ = 0;      ///< Number of unique classes

    public:
        /**
         * @brief Default constructor
         */
        OneHotEncoder() = default;

        /**
         * @brief Fit the encoder to the data
         * 
         * Learns the unique classes present in the input labels.
         * Classes are sorted to ensure consistent encoding.
         * 
         * @param y Input labels matrix (n_samples x 1)
         * @return Reference to this encoder (for method chaining)
         * @throws std::invalid_argument if y has more than 1 column
         */
        OneHotEncoder& fit(const Matrix<T>& y) {
            if (y.cols() != 1) {
                throw std::invalid_argument("OneHotEncoder expects a column vector (n_samples x 1)");
            }

            // Collect unique classes
            std::vector<T> unique_values;
            for (std::size_t i = 0; i < y.rows(); ++i) {
                T val = y(i, 0);
                if (std::find(unique_values.begin(), unique_values.end(), val) == unique_values.end()) {
                    unique_values.push_back(val);
                }
            }

            // Sort classes for consistent ordering
            std::sort(unique_values.begin(), unique_values.end());
            classes_ = unique_values;
            num_classes_ = classes_.size();

            // Build class-to-index mapping
            class_to_index_.clear();
            for (std::size_t i = 0; i < classes_.size(); ++i) {
                class_to_index_[classes_[i]] = i;
            }

            fitted_ = true;
            return *this;
        }

        /**
         * @brief Transform labels to one-hot encoded matrix
         * 
         * @param y Input labels matrix (n_samples x 1)
         * @return One-hot encoded matrix (n_samples x n_classes)
         * @throws std::runtime_error if encoder has not been fitted
         * @throws std::invalid_argument if y has more than 1 column
         * @throws std::out_of_range if y contains unknown class values
         */
        Matrix<T> transform(const Matrix<T>& y) const {
            if (!fitted_) {
                throw std::runtime_error("OneHotEncoder has not been fitted. Call fit() first.");
            }
            if (y.cols() != 1) {
                throw std::invalid_argument("OneHotEncoder expects a column vector (n_samples x 1)");
            }

            std::size_t n_samples = y.rows();
            Matrix<T> encoded(n_samples, num_classes_);

            // Initialize to zeros
            for (std::size_t i = 0; i < n_samples; ++i) {
                for (std::size_t j = 0; j < num_classes_; ++j) {
                    encoded(i, j) = T{0};
                }
            }

            // Set one-hot values
            for (std::size_t i = 0; i < n_samples; ++i) {
                T val = y(i, 0);
                auto it = class_to_index_.find(val);
                if (it == class_to_index_.end()) {
                    throw std::out_of_range("Unknown class value encountered during transform: " + std::to_string(static_cast<int>(val)));
                }
                encoded(i, it->second) = T{1};
            }

            return encoded;
        }

        /**
         * @brief Fit and transform in one step
         * 
         * Convenience method that combines fit() and transform().
         * 
         * @param y Input labels matrix (n_samples x 1)
         * @return One-hot encoded matrix (n_samples x n_classes)
         */
        Matrix<T> fit_transform(const Matrix<T>& y) {
            fit(y);
            return transform(y);
        }

        /**
         * @brief Inverse transform one-hot encoded matrix back to labels
         * 
         * Converts probability distributions or one-hot vectors back to class labels
         * using argmax (index of maximum value per row).
         * 
         * @param y_encoded One-hot or probability matrix (n_samples x n_classes)
         * @return Labels matrix (n_samples x 1) with original class values
         * @throws std::runtime_error if encoder has not been fitted
         * @throws std::invalid_argument if number of columns doesn't match n_classes
         */
        Matrix<T> inverse_transform(const Matrix<T>& y_encoded) const {
            if (!fitted_) {
                throw std::runtime_error("OneHotEncoder has not been fitted. Call fit() first.");
            }
            if (y_encoded.cols() != num_classes_) {
                throw std::invalid_argument("Number of columns must match number of classes");
            }

            std::size_t n_samples = y_encoded.rows();
            Matrix<T> labels(n_samples, 1);

            for (std::size_t i = 0; i < n_samples; ++i) {
                // Find argmax
                std::size_t max_idx = 0;
                T max_val = y_encoded(i, 0);
                for (std::size_t j = 1; j < num_classes_; ++j) {
                    if (y_encoded(i, j) > max_val) {
                        max_val = y_encoded(i, j);
                        max_idx = j;
                    }
                }
                labels(i, 0) = classes_[max_idx];
            }

            return labels;
        }

        /**
         * @brief Get the number of classes
         * @return Number of unique classes
         * @throws std::runtime_error if encoder has not been fitted
         */
        std::size_t get_num_classes() const {
            if (!fitted_) {
                throw std::runtime_error("OneHotEncoder has not been fitted. Call fit() first.");
            }
            return num_classes_;
        }

        /**
         * @brief Get the unique classes
         * @return Vector of unique class values (sorted)
         * @throws std::runtime_error if encoder has not been fitted
         */
        const std::vector<T>& get_classes() const {
            if (!fitted_) {
                throw std::runtime_error("OneHotEncoder has not been fitted. Call fit() first.");
            }
            return classes_;
        }

        /**
         * @brief Check if the encoder has been fitted
         * @return true if fitted, false otherwise
         */
        bool is_fitted() const {
            return fitted_;
        }
};
