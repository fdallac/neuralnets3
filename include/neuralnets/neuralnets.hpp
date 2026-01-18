/**
 * @file neuralnets.hpp
 * @brief Neural network implementation with backpropagation
 * 
 * This file contains the core neural network classes implementing
 * feedforward architecture with backpropagation training.
 */

#pragma once

#include "matrix/matrix.hpp"
#include "neuralnets/activation.hpp"
#include "neuralnets/loss.hpp"
#include "neuralnets/optimizer.hpp"
#include "neuralnets/normalization.hpp"
#include <iostream>
#include <cmath>

/**
 * @brief Represents a single layer in a neural network
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Stores weights, biases, gradients, and cached values for forward/backward passes.
 * Uses He initialization for weights to prevent gradient vanishing with ReLU.
 */
template<typename T>
class NeuralLayer {
    public:
        Matrix<T> W;  ///< Weight matrix (input_size x output_size)
        Matrix<T> b;  ///< Bias vector (1 x output_size)
        Matrix<T> dW; ///< Weight gradient (same shape as W)
        Matrix<T> db; ///< Bias gradient (same shape as b)
        Matrix<T> X;  ///< Cached input from forward pass
        Matrix<T> Z;  ///< Cached pre-activation values (X*W + b)
        Matrix<T> A;  ///< Cached post-activation output
        Activation<T>& activation; ///< Reference to activation function
        Normalization<T>* normalization; ///< Pointer to normalization (nullptr if none)
        std::string _id;  ///< Layer identifier

        /**
         * @brief Construct a neural network layer
         * @param input_size Number of input features
         * @param output_size Number of output neurons
         * @param activation Reference to activation function
         * @param normalization Pointer to normalization layer (optional, nullptr for none)
         * @param id Layer identifier (optional)
         * 
         * Initializes weights using He initialization: stddev = sqrt(2/n_inputs)
         * This helps prevent gradient vanishing/exploding with ReLU activations.
         */
        NeuralLayer(std::size_t input_size, std::size_t output_size, Activation<T>& activation, Normalization<T>* normalization = nullptr, std::string id = "") 
            : activation(activation), normalization(normalization), _id(id) {
            // Weights and bias are initialized to zero by default
            this->W = Matrix<T>(input_size, output_size);
            this->b = Matrix<T>(1, output_size);
            this->dW = Matrix<T>(input_size, output_size);
            this->db = Matrix<T>(1, output_size);
            this->X = Matrix<T>(1, input_size);
            this->Z = Matrix<T>(input_size, output_size);

            // Initialize weights with He initialization (good for ReLU)
            // stddev = sqrt(2 / n_inputs) for better gradient flow
            T he_stddev = std::sqrt(static_cast<T>(2.0) / static_cast<T>(input_size));
            this->normally_initialize_weights(static_cast<T>(0), he_stddev);
            this->normally_initialize_bias(static_cast<T>(0), static_cast<T>(0.01));

            // If id not provided, generate random integer id (must be very unlikely to collide)
            if (this->_id.empty()) {
                this->_id = std::to_string(rand());
            }
        }


        /**
         * @brief Initialize weights with normal distribution
         * @param mean Mean of the distribution
         * @param stddev Standard deviation of the distribution
         * 
         * Generates random values in [-stddev, +stddev] range centered at mean.
         * Uses simple uniform random generation (not true normal distribution).
         */
        void normally_initialize_weights(T mean, T stddev) {
            for (std::size_t i = 0; i < W.rows(); ++i) {
                for (std::size_t j = 0; j < W.cols(); ++j) {
                    // Random value in [-1, 1] range, then scale by stddev
                    T random_val = static_cast<T>(2.0) * static_cast<T>(rand()) / static_cast<T>(RAND_MAX) - static_cast<T>(1.0);
                    W(i, j) = mean + stddev * random_val;
                }
            }
        }

        /**
         * @brief Initialize bias with normal distribution
         * @param mean Mean of the distribution
         * @param stddev Standard deviation of the distribution
         * 
         * Generates random values in [-stddev, +stddev] range centered at mean.
         */
        void normally_initialize_bias(T mean, T stddev) {
            for (std::size_t j = 0; j < b.cols(); ++j) {
                // Random value in [-1, 1] range, then scale by stddev
                T random_val = static_cast<T>(2.0) * static_cast<T>(rand()) / static_cast<T>(RAND_MAX) - static_cast<T>(1.0);
                b(0, j) = mean + stddev * random_val;
            }
        }
};


/**
 * @brief Feedforward neural network with backpropagation
 * @tparam T Numeric type (float, double, etc.)
 * 
 * Implements a multi-layer perceptron (MLP) with:
 * - Arbitrary number of layers with configurable sizes and activations
 * - Backpropagation algorithm for training
 * - Pluggable optimizer and loss function
 * - He initialization for weights
 */
template<typename T>
class NeuralNets {

    public:
        /**
         * @brief Construct a neural network
         * @param optimizer Reference to optimizer for weight updates
         * @param loss_function Reference to loss function for training
         * 
         * Creates an empty network. Use add_layer() to build architecture.
         */
        NeuralNets(Optimizer<T>& optimizer, Loss<T>& loss_function)
            : optimizer(optimizer), loss_function(loss_function), n_layers(0) {
            // Initialize empty weights, bias and activations
            this->layers = std::vector<NeuralLayer<T>>();
        }

        /**
         * @brief Add a new layer to the neural network
         * @param input_size Number of input features to this layer
         * @param output_size Number of neurons in this layer
         * @param activation Reference to activation function for this layer
         * @param normalization Pointer to normalization layer (optional, nullptr for none)
         * @param id Layer identifier (optional)
         * 
         * Layers must be added in order from input to output.
         * The input_size of layer i+1 should match output_size of layer i.
         */
        void add_layer(size_t input_size, size_t output_size, Activation<T>& activation, Normalization<T>* normalization = nullptr, std::string id = "") {
            // Initialize weights and bias matrices // TODO better initialization (e.g., Xavier, He)
            this->layers.push_back(NeuralLayer<T>(input_size, output_size, activation, normalization, id));
            n_layers++;
        }

        /**
         * @brief Train the neural network using backpropagation
         * @param X Input data matrix (batch_size x input_features)
         * @param y Target data matrix (batch_size x output_features)
         * @param epochs Number of training epochs
         * @param verbose Whether to print loss each epoch (default: true)
         * 
         * Performs full-batch gradient descent:
         * 1. Forward pass through all layers
         * 2. Compute loss
         * 3. Backward pass to compute gradients
         * 4. Update weights and biases using optimizer
         * 
         * Prints "Epoch X/Y, Loss: Z" if verbose is true.
         */
        void train(const Matrix<T>& X, const Matrix<T>& y, int epochs, bool verbose = true)  {
                        
            // Simple training loop
            for (int epoch = 0; epoch < epochs; ++epoch) {
                // Forward pass
                Matrix<T> Z = this->predict(X);

                // Calculate loss
                T loss = this->loss_function.forward(Z, y);

                // Log progress if verbose
                if (verbose) std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << loss << std::endl;

                // Calculate gradient of the loss
                Matrix<T> gradient = this->loss_function.backward(Z, y);

                // Backward propagation through all layers
                for (int i = n_layers - 1; i >= 0; --i) {
                    gradient = backward_pass(gradient, layers[i]);
                }

                // Update weights and biases for all layers
                for (int i = n_layers - 1; i >= 0; --i) {
                    optimizer.update_weights(layers[i].W, layers[i].dW, layers[i]._id);
                    optimizer.update_bias(layers[i].b, layers[i].db, layers[i]._id);
                    
                    // Update normalization parameters if present
                    if (layers[i].normalization != nullptr) {
                        // Check if it's LayerNormalization (has gamma/beta)
                        auto* layer_norm = dynamic_cast<LayerNormalization<T>*>(layers[i].normalization);
                        if (layer_norm != nullptr) {
                            optimizer.update_weights(layer_norm->gamma, layer_norm->d_gamma, layers[i]._id + "_gamma");
                            optimizer.update_bias(layer_norm->beta, layer_norm->d_beta, layers[i]._id + "_beta");
                        }
                    }
                    
                    optimizer.next_step();
                }
            }
        }

        /**
         * @brief Predict outputs for input data
         * @param X Input data matrix (batch_size x input_features)
         * @return Predicted outputs (batch_size x output_features)
         * 
         * Performs forward pass through all layers without caching gradients.
         * Does not modify network weights.
         */
        Matrix<T> predict(const Matrix<T>& X) {
            // Forward pass through first layer
            Matrix<T> Z = forward_pass(X, this->layers[0]);
            // Forward pass through remaining layers
            for (int i = 1; i < n_layers; ++i) {
                Z = forward_pass(Z, this->layers[i]);
            }
            return Z;
        }



        /**
         * @brief Set the optimizer for the neural network
         * @param optimizer New optimizer to use for weight updates
         */
        void set_optimizer(const Optimizer<T>& optimizer) {
            this->optimizer = optimizer;
        }


        /**
         * @brief Set the loss function for the neural network
         * @param loss_function New loss function to use for training
         */
        void set_loss_function(const Loss<T>& loss_function) {
            this->loss_function = loss_function;
        }


        /**
         * @brief Get number of layers in the network
         * @return Number of layers
         */
        int get_num_layers() const {
            return n_layers;
        }


        /**
         * @brief Get a specific layer by index
         * @param index Layer index (0-based)
         * @return Reference to the layer
         */
        NeuralLayer<T>& get_layer(int index) {
            return layers[index];
        }


    private:
        int n_layers; ///< Number of layers in the network
        std::vector<NeuralLayer<T>> layers; ///< Vector of all layers
        
        Optimizer<T>& optimizer; ///< Reference to optimizer
        Loss<T>& loss_function; ///< Reference to loss function


        /**
         * @brief Forward pass through a single layer
         * @param X Input matrix (batch_size x input_features)
         * @param layer Reference to layer to pass through
         * @return Activation output (batch_size x output_features)
         * @throws std::invalid_argument if input dimensions don't match layer weights
         * 
         * Computes:
         * 1. Z = X * W + b (pre-activation)
         * 2. Z_norm = normalization(Z) (if normalization enabled)
         * 3. A = activation(Z_norm) (post-activation)
         * 
         * Caches X, Z, A for backward pass.
         */
        static Matrix<T> forward_pass(const Matrix<T>& X, NeuralLayer<T>& layer) {
            if (X.cols() != layer.W.rows()) {
                throw std::invalid_argument("Incompatible dimensions for layer forward pass");
            }
            // Cache input for back-propagation
            layer.X = X;
            // Z = X * W + b
            layer.Z = X * layer.W;
            layer.Z.broadcast_horizontal_sum_inplace(layer.b);

            // Apply normalization if present
            Matrix<T> Z_normalized = layer.Z;
            if (layer.normalization != nullptr) {
                Z_normalized = layer.normalization->forward(layer.Z);
            }

            // A = activation(Z) - cache the activation output
            layer.A = layer.activation.forward(Z_normalized);
            return layer.A;
        }


        /**
         * @brief Backward pass through a single layer
         * @param gradient Gradient from next layer (batch_size x current_layer_output)
         * @param layer Reference to layer to backpropagate through
         * @return Gradient to propagate to previous layer (batch_size x current_layer_input)
         * 
         * Computes:
         * 1. dA = gradient (from upstream)
         * 2. dZ_norm = activation.backward(Z, dA)
         * 3. dZ = normalization.backward(dZ_norm) (if normalization enabled)
         * 4. dW = X^T * dZ (weight gradient)
         * 5. db = sum(dZ) over batch (bias gradient)
         * 6. prev_gradient = dZ * W^T (gradient for previous layer)
         * 
         * Updates layer.dW and layer.db for optimizer to use.
         */ 
        static Matrix<T> backward_pass(Matrix<T>& gradient, NeuralLayer<T>& layer)  {
            // dZ = dL/dZ (through activation)
            // Note: If normalization was applied, we need to use the normalized Z
            Matrix<T> Z_for_backward = layer.Z;
            if (layer.normalization != nullptr) {
                Z_for_backward = layer.normalization->forward(layer.Z);  // Re-compute normalized Z
            }
            
            Matrix<T> dZ = layer.activation.backward(Z_for_backward, gradient);

            // Backprop through normalization if present
            if (layer.normalization != nullptr) {
                dZ = layer.normalization->backward(dZ);
            }

            // prev_gradient = delta * W^T 
            Matrix<T> prev_gradient = dZ * layer.W.transpose();

            // Get gradients for weights and bias updates
            layer.db = dZ.horizontal_sum(); // Sum over rows to get bias gradient
            layer.dW = layer.X.transpose() * dZ;   

            return prev_gradient;
        }

};  