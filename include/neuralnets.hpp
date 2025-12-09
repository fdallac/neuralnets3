#pragma once
#include "matrix.hpp"
#include "activation.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include <iostream>

template<typename T>
class NeuralLayer {
    public:
        Matrix<T> W, b, dW, db, X, Z;
        Activation<T>& activation;

        NeuralLayer(std::size_t input_size, std::size_t output_size, Activation<T>& activation) 
            : activation(activation) {
            // Weights and bias are initialized to zero by default
            this->W = Matrix<T>(input_size, output_size);
            this->b = Matrix<T>(1, output_size);
            this->dW = Matrix<T>(input_size, output_size);
            this->db = Matrix<T>(1, output_size);
            this->X = Matrix<T>(1, input_size);
            this->Z = Matrix<T>(input_size, output_size);
        }
};

template<typename T>
class NeuralNets {

    public:
        NeuralNets(Optimizer<T>& optimizer, Loss<T>& loss_function)
            : optimizer(optimizer), loss_function(loss_function), n_layers(0) {
            // Initialize empty weights, bias and activations
            this->layers = std::vector<NeuralLayer<T>>();
        }

        /// @brief Add a new layer to the neural network   
        /// @param input_size 
        /// @param output_size 
        /// @param activation 
        void add_layer(size_t input_size, size_t output_size, Activation<T>& activation) {
            // Initialize weights and bias matrices // TODO better initialization (e.g., Xavier, He)
            this->layers.push_back(NeuralLayer<T>(input_size, output_size, activation));
            n_layers++;
        }


        /// @brief Train the neural network
        /// @param X Input data matrix
        /// @param y Target data matrix
        /// @param epochs Number of training epochs
        /// @param verbose Whether to print training progress
        void train(const Matrix<T>& X, const Matrix<T>& y, std::size_t epochs, bool verbose = true)  {
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

                // Update weights and biases for all layers
                for (std::size_t i = 0; i < n_layers; ++i) {
                    optimizer.update_weights(layers[i].W, layers[i].dW);
                    optimizer.update_bias(layers[i].b, layers[i].db);
                }
            }
        }


        /// @brief Predict using the trained neural network
        /// @param X Input data matrix
        /// @return Predicted output matrix
        Matrix<T> predict(const Matrix<T>& X) {
            // Forward pass through first layer
            Matrix<T> Z = forward_pass(X, this->layers[0]);
            // Forward pass through remaining layers
            for (std::size_t i = 1; i < n_layers; ++i) {
                Z = forward_pass(Z, this->layers[i]);
            }
            return Z;
        }



        /// @brief Set the optimizer for the neural network
        /// @param optimizer
        void set_optimizer(const Optimizer<T>& optimizer) {
            this->optimizer = optimizer;
        }


        /// @brief Set the loss function for the neural network
        /// @param loss_function
        void set_loss_function(const Loss<T>& loss_function) {
            this->loss_function = loss_function;
        }


        /// @brief Get number of layers
        /// @return Number of layers
        std::size_t get_num_layers() const {
            return n_layers;
        }

        /// @brief Get a specific layer
        /// @param index Layer index
        NeuralLayer<T>& get_layer(std::size_t index) {
            return layers[index];
        }


    private:
        int n_layers;
        std::vector<NeuralLayer<T>> layers;
        
        Optimizer<T>& optimizer;
        Loss<T>& loss_function;


        /// @brief Example public method: Forward pass through a layer
        /// @param X Input matrix
        /// @param layer 
        /// @return Output matrix after forward pass
        static Matrix<T> forward_pass(const Matrix<T>& X, NeuralLayer<T>& layer) {
            if (X.cols() != layer.W.rows()) {
                throw std::invalid_argument("Incompatible dimensions for layer forward pass");
            }
            // Cache input for back-propagation
            layer.X = X;
            // Z = W * X_in
            layer.Z = X * layer.W;
            // Z' = Z + b
            layer.Z.broadcast_horizontal_sum_inplace(layer.b);

            // X_out = activation(Z')
            return layer.activation.forward(layer.Z);
        }


        /// @brief  Example method: Backward pass through a layer (stub implementation)
        /// @param output 
        /// @param target 
        /// @param weights 
        /// @param bias 
        /// @param activation 
        static Matrix<T> backward_pass(Matrix<T>& gradient, NeuralLayer<T>& layer)  {
            // dZ = gradient * activation_derivative (element-wise)
            Matrix<T> dZ = gradient.elementwise_multiply(layer.activation.backward(layer.Z));

            // prev_gradient = delta * W^T 
            Matrix<T> prev_gradient = dZ * layer.W.transpose();

            // Get gradients for weights and bias updates
            layer.db = dZ.horizontal_sum(); // Sum over rows to get bias gradient
            layer.dW = layer.X.transpose() * dZ;   

            return prev_gradient;
        }

};  