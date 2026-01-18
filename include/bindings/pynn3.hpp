/**
 * @file pynn3.hpp
 * @brief High-level Python interface for NeuralNets
 * 
 * This header defines the PyNN3 class which provides a clean,
 * NumPy-based interface for Python users
 */
    
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>
#include <vector>

#include "neuralnets/neuralnets.hpp"
#include "neuralnets/activation.hpp"
#include "neuralnets/loss.hpp"
#include "neuralnets/optimizer.hpp"
#include "numpy_matrix_helper.hpp"

namespace py = pybind11;

namespace pynn3 {

/**
 * @brief Python-friendly wrapper for NeuralNets
 * @tparam T Numeric type (float, double)
 * 
 * This class provides a high-level API for Python users:
 * - Accepts NumPy arrays for train/predict
 * - Hides internal Matrix implementation
 * - Manages ownership of all C++ components
 * - Releases GIL during long computations
 */
template<typename T>
class PyNN3 {
private:
    std::unique_ptr<NeuralNets<T>> net_;
    
    // Owned components - NeuralNets holds references to these
    std::shared_ptr<Optimizer<T>> optimizer_;
    std::shared_ptr<Loss<T>> loss_;
    std::vector<std::shared_ptr<Activation<T>>> activations_;

public:
    /**
     * @brief Construct a new neural network
     * @param optimizer Shared pointer to optimizer (SGD, Adam, etc.)
     * @param loss Shared pointer to loss function (MSE, CrossEntropy, etc.)
     */
    PyNN3(std::shared_ptr<Optimizer<T>> optimizer, std::shared_ptr<Loss<T>> loss)
        : optimizer_(optimizer), loss_(loss) {
        net_ = std::make_unique<NeuralNets<T>>(*optimizer_, *loss_);
    }

    /**
     * @brief Add a dense layer to the network
     * @param input_size Number of input features
     * @param output_size Number of output neurons
     * @param activation Shared pointer to activation function
     */
    void add_layer(int input_size, int output_size, std::shared_ptr<Activation<T>> activation) {
        activations_.push_back(activation);
        net_->add_layer(static_cast<size_t>(input_size), 
                        static_cast<size_t>(output_size), 
                        *activation);
    }

    /**
     * @brief Train the network on provided data
     * @param X_np Input features as NumPy array (n_samples x n_features)
     * @param y_np Target values as NumPy array (n_samples x n_outputs)
     * @param epochs Number of training epochs
     * @param verbose Whether to print loss each epoch
     * 
     * Releases the GIL during training for better Python threading.
     */
    void train(py::array_t<T> X_np, py::array_t<T> y_np, int epochs, bool verbose = true) {
        // Convert NumPy -> C++ Matrix
        Matrix<T> X = internal::numpy_to_matrix<T>(X_np);
        Matrix<T> y = internal::numpy_to_matrix<T>(y_np);

        // Release GIL for potentially long-running C++ training
        {
            py::gil_scoped_release release;
            net_->train(X, y, epochs, verbose);
        }
    }

    /**
     * @brief Predict outputs for input data
     * @param X_np Input features as NumPy array (n_samples x n_features)
     * @return Predictions as NumPy array (n_samples x n_outputs)
     */
    py::array_t<T> predict(py::array_t<T> X_np) {
        // Convert NumPy -> C++ Matrix
        Matrix<T> X = internal::numpy_to_matrix<T>(X_np);
        
        // Run prediction
        Matrix<T> result = net_->predict(X);

        // Return as zero-copy NumPy array
        return internal::matrix_to_numpy_zerocopy<T>(std::move(result));
    }

    /**
     * @brief Get number of layers in the network
     * @return Number of layers
     */
    size_t num_layers() const {
        return activations_.size();
    }
};

} // namespace pynn3
