#include "matrix.hpp"
#include "neuralnets.hpp"
#include "matmul.hpp"
#include "activation.hpp"
#include "loss.hpp"


template<typename T>
NeuralNets<T>::NeuralNets(std::string optimizer, T learning_rate) {
    n_layers = 0;
    // Initialize empty weights, bias and activations
    weights.clear();
    bias.clear();
    activations.clear();
}


template<typename T>
void NeuralNets<T>::add_layer(size_t input_size, size_t output_size, const std::string& activation) {
    // Initialize weights and bias matrices // TODO better initialization (e.g., Xavier, He)
    Matrix<T> W(input_size, output_size);
    Matrix<T> b(1, output_size);
    weights.push_back(W);
    bias.push_back(b);
    activations.push_back(activation);
    n_layers++;
}


template<typename T>
void NeuralNets<T>::train(const Matrix<T>& input, const Matrix<T>& target, std::size_t epochs, T learning_rate) {
    // Simple training loop
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        Matrix<T> output = input;
        // Forward pass through all layers
        for (std::size_t i = 0; i < n_layers; ++i) {
            output = forward_pass(output, weights[i], bias[i], activations[i]);
        }
        // Backward pass through all layers
        for (std::size_t i = n_layers; i > 0; --i) {
            output = backward_pass(output, target, weights[i - 1], bias[i - 1], activations[i - 1]);
        }
    }
}


template<typename T>
Matrix<T> NeuralNets<T>::predict(const Matrix<T>& input) {
    Matrix<T> output = input;
    for (std::size_t i = 0; i < n_layers; ++i) {
        output = forward_pass(output, weights[i], bias[i], activations[i]);
    }
    return output;
}


template<typename T>
Matrix<T> NeuralNets<T>::forward_pass(const Matrix<T>& input, const Matrix<T>& weights, const Matrix<T>& bias, const std::string& activation) {
    if (input.cols() != weights.rows()) {
        throw std::invalid_argument("Incompatible dimensions for layer forward pass");
    }
    Matrix<T> z = MatMul<T>::mm(input, weights);
    for (std::size_t i = 0; i < z.rows(); ++i) {
        for (std::size_t j = 0; j < z.cols(); ++j) {
            z(i, j) += bias(0, j);
        }
    }
    return apply_activation(z, activation);
}


template<typename T>
Matrix<T> NeuralNets<T>::backward_pass(const Matrix<T>& output, const Matrix<T>& target, const Matrix<T>& weights, const Matrix<T>& bias, const std::string& activation) {
    if (output.rows() != target.rows() || output.cols() != target.cols()) {
        throw std::invalid_argument("Incompatible dimensions for layer backward pass");
    }
    // Stub implementation: return zero matrix of same size as output
    return Matrix<T>(output.rows(), output.cols());
}


template<typename T>
Matrix<T> NeuralNets<T>::apply_activation(const Matrix<T>& M, const std::string& activation) {
    if (activation == "relu") {
        return relu(M);
    } else if (activation == "sigmoid") {
        return sigmoid(M);
    } else if (activation == "tanh") {
        return tanh(M);
    } else {
        throw std::invalid_argument("Unknown activation function: " + activation);
    }
}


template<typename T>
Matrix<T> NeuralNets<T>::apply_activation_derivative(const Matrix<T>& M, const std::string& activation) {
    if (activation == "relu") {
        return drelu(M);
    } else if (activation == "sigmoid") {
        return dsigmoid(M);
    } else if (activation == "tanh") {
        return dtanh(M);
    } else {
        throw std::invalid_argument("Unknown activation function: " + activation);
    }
}


