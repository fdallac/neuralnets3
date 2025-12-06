#pragma once
#include "matrix.hpp"
#include "activations.hpp"


template<typename T>
class NeuralNets {

    public:
        NeuralNets(std::string optimizer = "sgd", T learning_rate = T{0.01});

        static void add_layer(size_t input_size, size_t output_size, const std::string& activation);

        static void train(const Matrix<T>& input, const Matrix<T>& target, std::size_t epochs, T learning_rate);

        static Matrix<T> predict(const Matrix<T>& input);



    private:
        static int n_layers;
        static std::vector<Matrix<T>> weights;
        static std::vector<Matrix<T>> bias;
        static std::vector<std::string> activations;


        /// @brief Example public method: Forward pass through a layer
        /// @param input Input matrix
        /// @param weights Weights matrix
        /// @param bias Bias matrix
        /// @param activation Activation function to apply ("relu", "sigmoid", "tanh", "identity")
        /// @return Output matrix after applying weights, bias, and activation
        static Matrix<T> forward_pass(const Matrix<T>& input, const Matrix<T>& weights, const Matrix<T>& bias, const std::string& activation);


        /// @brief  Example method: Backward pass through a layer (stub implementation)
        /// @param output 
        /// @param target 
        /// @param weights 
        /// @param bias 
        /// @param activation 
        static Matrix<T> backward_pass(const Matrix<T>& output, const Matrix<T>& target, const Matrix<T>& weights, const Matrix<T>& bias, const std::string& activation);


        /// @brief Example method: Apply activation function to a matrix
        /// @param M Input matrix
        /// @param activation Activation function ("relu", "sigmoid", "tanh", "identity")
        /// @return Matrix after applying activation function
        static Matrix<T> apply_activation(const Matrix<T>& M, const std::string& activation);


        /// @brief Apply the derivative of an activation function to a matrix
        /// @param M Input matrix
        /// @param activation Activation function ("relu", "sigmoid", "tanh", "identity")
        /// @return Matrix after applying the derivative of the activation function
        static Matrix<T> apply_activation_derivative(const Matrix<T>& M, const std::string& activation);
    










  };