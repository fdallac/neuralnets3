#include "matrix.hpp"
#include "neuralnets.hpp"
#include "matmul.hpp"
#include "activation.hpp"
#include "loss.hpp"

template<typename T>
NeuralLayer<T>::NeuralLayer(std::size_t input_size, std::size_t output_size, const Activation<T>& activation) {
    // Weights and bias are initialized to zero by default
    this->W = Matrix<T>(input_size, output_size);
    this->b = Matrix<T>(1, output_size);
    this->dW = Matrix<T>(input_size, output_size);
    this->db = Matrix<T>(1, output_size);
    this->Z = Matrix<T>(input_size, output_size);
    this->activation = activation;
}


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
    this->layers.push_back(NeuralLayer<T>(input_size, output_size, activation));
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
        for (std::size_t i = n_layers - 1; i >= 0; --i) {
            gradient = backward_pass(gradient, layers[i]);
        }

        optimizer->update_weights(this->weights, this->weight_gradients);
        optimizer->update_bias(this->bias, this->bias_gradients);
    }
}


template<typename T>
Matrix<T> NeuralNets<T>::predict(const Matrix<T>& X) {
    // Forward pass through first layer
    Matrix<T> Z = forward_pass(X, this->layers[0]);
    // Forward pass through remaining layers
    for (std::size_t i = 1; i < n_layers; ++i) {
        Z = forward_pass(Z, this->layers[i]);
    }
    return Z;
}


template<typename T>
Matrix<T> NeuralNets<T>::forward_pass(const Matrix<T>& X, const NeuralLayer<T>& layer) {
    if (X.cols() != layer.W.rows()) {
        throw std::invalid_argument("Incompatible dimensions for layer forward pass");
    }
    // Cache input for backpropagation
    layer.X = X;
    // Z = W * X_in
    layer.Z = X * layer.W;
    // Z' = Z + b
    layer.Z +_= layer.b;

    // X_out = activation(Z')
    return layer.activation.forward(layer.Z);
}


template<typename T>
Matrix<T> NeuralNets<T>::backward_pass(const Matrix<T>& gradient, const NeuralLayer<T>& layer) {
    // dZ = gradient * activation_derivative (element-wise)
    Matrix<T> dZ = gradient .* layer.activation.backward(layer.Z);

    // prev_gradient = delta * W^T 
    Matrix<T> prev_gradient = dZ * layer.W.transpose();

    // Get gradients for weights and bias updates
    this->db = dZ.horizontal_sum(); // Sum over rows to get bias gradient
    this->dW = layer.X.transpose() * dZ;   

    return prev_gradient;
}


