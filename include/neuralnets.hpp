#pragma once
#include "matrix.hpp"
#include "activation.hpp"
#include "loss.hpp"
#include "optimizer.hpp"

template<typename T>
class NeuralLayer {
    public:
        Matrix<T> W, b, dW, db, Z;
        Activation<T> activation;
        NeuralLayer(std::size_t input_size, std::size_t output_size, const Activation<T>& activation);
};

template<typename T>
class NeuralNets {

    public:
        NeuralNets(Optimizer<T> optimizer, Loss<T> loss_function);

        /// @brief Add a new layer to the neural network   
        /// @param input_size 
        /// @param output_size 
        /// @param activation 
        static void add_layer(size_t input_size, size_t output_size, const Activation& activation);


        /// @brief Train the neural network
        /// @param X Input data matrix
        /// @param y Target data matrix
        /// @param epochs Number of training epochs
        /// @param verbose Whether to print training progress
        static void train(const Matrix<T>& X, const Matrix<T>& y, std::size_t epochs, bool verbose = true);


        /// @brief Predict using the trained neural network
        /// @param X Input data matrix
        /// @return Predicted output matrix
        static Matrix<T> predict(const Matrix<T>& X);


        /// @brief Set the optimizer for the neural network
        /// @param optimizer
        static void set_optimizer(const Optimizer<T>& optimizer) {
            this->optimizer = optimizer;
        }


        /// @brief Set the loss function for the neural network
        /// @param loss_function
        static void set_loss_function(const Loss<T>& loss_function) {
            this->loss_function = loss_function;
        }


    private:
        static int n_layers;
        static std::vector<NeuralLayer<T>> layers;
        
        static Optimizer<T> optimizer;
        static Loss<T> loss_function;


        /// @brief Example public method: Forward pass through a layer
        /// @param X Input matrix
        /// @param layer 
        /// @return Output matrix after forward pass
        static Matrix<T> forward_pass(const Matrix<T>& X, const NeuralLayer<T>& layer);


        /// @brief  Example method: Backward pass through a layer (stub implementation)
        /// @param output 
        /// @param target 
        /// @param weights 
        /// @param bias 
        /// @param activation 
        static Matrix<T> backward_pass(const Matrix<T>& gradient, const NeuralLayer<T>& layer);

};  