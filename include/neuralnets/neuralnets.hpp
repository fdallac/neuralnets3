#pragma once

#include "matrix/matrix.hpp"
#include "neuralnets/activation.hpp"
#include "neuralnets/loss.hpp"
#include "neuralnets/optimizer.hpp"
#include <iostream>
#include <cmath>

template<typename T>
class NeuralLayer {
    public:
        Matrix<T> W, b, dW, db, X, Z, A;  // Added A to cache activation output
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

            // Initialize weights with He initialization (good for ReLU)
            // stddev = sqrt(2 / n_inputs) for better gradient flow
            T he_stddev = std::sqrt(static_cast<T>(2.0) / static_cast<T>(input_size));
            this->normally_initialize_weights(static_cast<T>(0), he_stddev);
            this->normally_initialize_bias(static_cast<T>(0), static_cast<T>(0.01));
        }


        void normally_initialize_weights(T mean, T stddev) {
            for (std::size_t i = 0; i < W.rows(); ++i) {
                for (std::size_t j = 0; j < W.cols(); ++j) {
                    // Random value in [-1, 1] range, then scale by stddev
                    T random_val = static_cast<T>(2.0) * static_cast<T>(rand()) / static_cast<T>(RAND_MAX) - static_cast<T>(1.0);
                    W(i, j) = mean + stddev * random_val;
                }
            }
        }

        void normally_initialize_bias(T mean, T stddev) {
            for (std::size_t j = 0; j < b.cols(); ++j) {
                // Random value in [-1, 1] range, then scale by stddev
                T random_val = static_cast<T>(2.0) * static_cast<T>(rand()) / static_cast<T>(RAND_MAX) - static_cast<T>(1.0);
                b(0, j) = mean + stddev * random_val;
            }
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
        void train(const Matrix<T>& X, const Matrix<T>& y, int epochs, bool verbose = true)  {
            
            // // DEBUGGING
            // std::cout << "=========="  << std::endl;
            // std::cout << "Layer 0 weights and bias: " << std::endl;
            // std::cout << "--- W0 ---"  << std::endl;
            // layers[0].W.display();
            // std::cout << "--- b0 ---" << std::endl;
            // layers[0].b.display();
            // std::cout << "--- dW0 ---" << std::endl;
            // layers[0].dW.display();
            // std::cout << "--- db0 ---" << std::endl;
            // layers[0].db.display();
            // std::cout << "==========" << std::endl;
            // std::cout << "Layer 1 weights and bias: " << std::endl;
            // std::cout << "--- W1 ---"  << std::endl;
            // layers[1].W.display();
            // std::cout << "--- b1 ---" << std::endl;
            // layers[1].b.display();
            // std::cout << "--- dW1 ---" << std::endl;
            // layers[1].dW.display();
            // std::cout << "--- db1 ---" << std::endl;
            // layers[1].db.display();
            // std::cout << "==========" << std::endl;
            // std::cout << "==========" << std::endl;
            
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
                    optimizer.update_weights(layers[i].W, layers[i].dW);
                    optimizer.update_bias(layers[i].b, layers[i].db);
                }

                // // DEBUGGING
                // std::cout << "=========="  << std::endl;
                // std::cout << "Layer 0 weights and bias: " << std::endl;
                // std::cout << "--- W0 ---"  << std::endl;
                // layers[0].W.display();
                // std::cout << "--- b0 ---" << std::endl;
                // layers[0].b.display();
                // std::cout << "--- dW0 ---" << std::endl;
                // layers[0].dW.display();
                // std::cout << "--- db0 ---" << std::endl;
                // layers[0].db.display();
                // std::cout << "==========" << std::endl;
                // std::cout << "Layer 1 weights and bias: " << std::endl;
                // std::cout << "--- W1 ---"  << std::endl;
                // layers[1].W.display();
                // std::cout << "--- b1 ---" << std::endl;
                // layers[1].b.display();
                // std::cout << "--- dW1 ---" << std::endl;
                // layers[1].dW.display();
                // std::cout << "--- db1 ---" << std::endl;
                // layers[1].db.display();
                // std::cout << "==========" << std::endl;
                // std::cout << "==========" << std::endl;
            }
        }


        /// @brief Predict using the trained neural network
        /// @param X Input data matrix
        /// @return Predicted output matrix
        Matrix<T> predict(const Matrix<T>& X) {
            // Forward pass through first layer
            Matrix<T> Z = forward_pass(X, this->layers[0]);
            // Forward pass through remaining layers
            for (int i = 1; i < n_layers; ++i) {
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
        int get_num_layers() const {
            return n_layers;
        }


        /// @brief Get a specific layer
        /// @param index Layer index
        NeuralLayer<T>& get_layer(int index) {
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
            // Z = X * W + b
            layer.Z = X * layer.W;
            layer.Z.broadcast_horizontal_sum_inplace(layer.b);

            // A = activation(Z) - cache the activation output
            layer.A = layer.activation.forward(layer.Z);
            return layer.A;
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