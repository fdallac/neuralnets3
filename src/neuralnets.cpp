#include "matrix.hpp"
#include "neuralnets.hpp"
#include "matmul.hpp"
#include "activation.hpp"
#include "loss.hpp"


template<typename T>
NeuralNets<T>::NeuralNets(Optimizer<T> optimizer, Loss<T> loss_function) {
    n_layers = 0;
    // Initialize empty weights, bias and activations
    weights.clear();
    bias.clear();
    activations.clear();
    this->optimizer = optimizer;
    this->loss_function = loss_function;
}


template<typename T>
void NeuralNets<T>::add_layer(size_t input_size, size_t output_size, const Activation<T>& activation) {
    // Initialize weights and bias matrices // TODO better initialization (e.g., Xavier, He)
    Matrix<T> W(input_size, output_size);
    Matrix<T> b(1, output_size); // row vector for bias
    weights.push_back(W);
    bias.push_back(b);
    activations.push_back(activation);
    n_layers++;
}


template<typename T>
void NeuralNets<T>::train(const Matrix<T>& X, const Matrix<T>& y, std::size_t epochs, bool verbose) {
    // Simple training loop
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        Matrix<T> Z = this->predict(X);

        // Calculate loss
        T loss = this->loss_function.forward(Z, y);

        // Log progress if verbose
        if (verbose) std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << loss << std::endl;

        // Calculate gradient of the loss
        Matrix<T> gradient = this->loss_function.backward(Z, y);

        // Backward propagation through all layers
        for (std::size_t i = n_layers; i > 0; --i) {
            gradient = backward_pass(gradient, this->weights[i - 1], this->bias[i - 1], this->activations[i - 1]);
        }
    }
}


template<typename T>
Matrix<T> NeuralNets<T>::predict(const Matrix<T>& X) {
    // Forward pass through first layer
    Matrix<T> Z = forward_pass(X, this->weights[0], this->bias[0], this->activations[0]);
    // Forward pass through remaining layers
    for (std::size_t i = 1; i < n_layers; ++i) {
        Z = forward_pass(Z, this->weights[i], this->bias[i], this->activations[i]);
    }
    return Z;
}


template<typename T>
Matrix<T> NeuralNets<T>::forward_pass(const Matrix<T>& X, const Matrix<T>& weights, const Matrix<T>& bias, const Activation<T>& activation) {
    if (X.cols() != weights.rows()) {
        throw std::invalid_argument("Incompatible dimensions for layer forward pass");
    }
    // Z = W * X_in
    Matrix<T> Z = X * weights;
    // Z' = Z + b
    Z +_= bias;

    // X_out = activation(Z')
    return activation.forward(Z);
}


template<typename T>
Matrix<T> NeuralNets<T>::backward_pass(const Matrix<T>& gradient, const Matrix<T>& weights, const Matrix<T>& bias, const Activation<T>& activation) {
    if (output.rows() != target.rows() || output.cols() != target.cols()) {
        throw std::invalid_argument("Incompatible dimensions for layer backward pass");
    }
    // 
    Matrix<T> activation_derivative = activation.backward(output);
    
    // delta = gradient * activation_derivative
    Matrix<T> delta = gradient *. activation_derivative;

    // prev_gradient = delta * W^T 
    Matrix<T> prev_gradient = delta * weights.transpose();

    // Update weights and bias using optimizer
    Matrix<T> weight_gradient = input.transpose() * delta;   

    optimizer->update_weights(weights, weight_gradient);

    optimizer->update_bias(bias, delta);


    return prev_gradient;
}


