#pragma once

#include "matrix/matrix.hpp"

template<typename T>
class Optimizer {
    public:
        virtual void update_weights(Matrix<T>& weights, const Matrix<T>& weight_gradient) = 0;
        virtual void update_bias(Matrix<T>& bias, const Matrix<T>& bias_gradient) = 0;
        virtual ~Optimizer() = default;
};


template<typename T>
class SGD : public Optimizer<T> {
    public:
        SGD(T learning_rate) : learning_rate(learning_rate) {}
        T get_learning_rate() const { return learning_rate; }

        void update_weights(Matrix<T>& weights, const Matrix<T>& weight_gradient) override {
            for (std::size_t i = 0; i < weights.rows(); ++i) {
                for (std::size_t j = 0; j < weights.cols(); ++j) {
                    weights(i, j) -= learning_rate * weight_gradient(i, j);
                }
            }
        }

        void update_bias(Matrix<T>& bias, const Matrix<T>& bias_gradient) override {
            for (std::size_t j = 0; j < bias.cols(); ++j) {
                bias(0, j) -= learning_rate * bias_gradient(0, j);
            }
        }

    private:
        T learning_rate;
};