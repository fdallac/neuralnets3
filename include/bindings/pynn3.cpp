/**
 * @file pynn3.cpp
 * @brief Python module definition for PyNN3
 * 
 * This file defines the pybind11 module that exposes the high-level
 * neural network API to Python. Users work with NumPy arrays - the
 * internal C++ Matrix implementation is hidden.
 * 
 * Module structure:
 * - numpy_matrix_helper.hpp: Internal NumPy <-> Matrix conversion utilities
 * - pynn3.hpp: High-level PyNN3 wrapper class
 * - pynn3.cpp: Module definition and Python bindings (this file)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "pynn3.hpp"

namespace py = pybind11;

using namespace pynn3;


// =============================================================================
// Module Definition
// =============================================================================

PYBIND11_MODULE(pynn3, m) {
    m.doc() = R"pbdoc(
        PyNN3 - Neural Networks with C++ Backend
        -----------------------------------------
        
        A C++ backed neural network library with NumPy integration.
        
        Example:
            >>> import pynn3 as nn
            >>> import numpy as np
            >>> 
            >>> # Create network
            >>> model = nn.NeuralNets(nn.SGD(0.1), nn.MSE())
            >>> model.add_layer(2, 4, nn.ReLU())
            >>> model.add_layer(4, 1, nn.Sigmoid())
            >>> 
            >>> # Train
            >>> X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
            >>> y = np.array([[0], [1], [1], [0]], dtype=np.float32)
            >>> model.train(X, y, epochs=1000)
            >>> 
            >>> # Predict
            >>> predictions = model.predict(X)
    )pbdoc";

    // =========================================================================
    // Abstract Base Classes
    // =========================================================================
    
    py::class_<Optimizer<float>, std::shared_ptr<Optimizer<float>>>(m, "Optimizer",
        "Abstract base class for optimizers (do not instantiate directly)");
    
    py::class_<Loss<float>, std::shared_ptr<Loss<float>>>(m, "Loss",
        "Abstract base class for loss functions (do not instantiate directly)");
    
    py::class_<Activation<float>, std::shared_ptr<Activation<float>>>(m, "Activation",
        "Abstract base class for activation functions (do not instantiate directly)");

    // =========================================================================
    // Optimizers
    // =========================================================================
    
    py::class_<SGD<float>, Optimizer<float>, std::shared_ptr<SGD<float>>>(m, "SGD",
        R"pbdoc(
            Stochastic Gradient Descent optimizer.
            
            Updates parameters using: θ = θ - learning_rate * gradient
            
            Args:
                learning_rate: Step size for gradient descent (typically 0.001 - 0.1)
        )pbdoc")
        .def(py::init<float>(), py::arg("learning_rate") = 0.01f,
             "Create SGD optimizer with specified learning rate")
        .def_property_readonly("learning_rate", &SGD<float>::get_learning_rate,
             "Get the current learning rate");

    py::class_<Adam<float>, Optimizer<float>, std::shared_ptr<Adam<float>>>(m, "Adam",
        R"pbdoc(
            Adam optimizer with adaptive learning rates.
            
            Combines momentum and RMSprop for robust optimization.
            
            Args:
                learning_rate: Initial learning rate (typically 0.001)
                beta1: Exponential decay rate for first moment (default: 0.9)
                beta2: Exponential decay rate for second moment (default: 0.999)
                epsilon: Small constant for numerical stability (default: 1e-8)
        )pbdoc")
        .def(py::init<float, float, float, float>(),
             py::arg("learning_rate") = 0.001f,
             py::arg("beta1") = 0.9f,
             py::arg("beta2") = 0.999f,
             py::arg("epsilon") = 1e-8f,
             "Create Adam optimizer with specified hyperparameters")
        .def_property_readonly("learning_rate", &Adam<float>::get_learning_rate,
             "Get the current learning rate");

    // =========================================================================
    // Loss Functions
    // =========================================================================
    
    py::class_<MSELoss<float>, Loss<float>, std::shared_ptr<MSELoss<float>>>(m, "MSE",
        R"pbdoc(
            Mean Squared Error loss function.
            
            MSE = (1/n) * Σ(predictions - targets)²
            
            Best for: Regression tasks with continuous outputs.
        )pbdoc")
        .def(py::init<>(), "Create MSE loss function");

    py::class_<BinaryCrossEntropyLoss<float>, Loss<float>, 
               std::shared_ptr<BinaryCrossEntropyLoss<float>>>(m, "BinaryCrossEntropy",
        R"pbdoc(
            Binary Cross-Entropy loss function.
            
            BCE = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]
            
            Best for: Binary classification (0/1 labels).
            Use with Sigmoid activation on output layer.
        )pbdoc")
        .def(py::init<>(), "Create Binary Cross-Entropy loss function");

    py::class_<CategoricalCrossEntropyLoss<float>, Loss<float>, 
               std::shared_ptr<CategoricalCrossEntropyLoss<float>>>(m, "CategoricalCrossEntropy",
        R"pbdoc(
            Categorical Cross-Entropy loss function.
            
            CCE = -(1/n) * Σ Σ y_ij * log(p_ij)
            
            Best for: Multi-class classification with one-hot encoded labels.
            Use with Softmax activation on output layer.
        )pbdoc")
        .def(py::init<>(), "Create Categorical Cross-Entropy loss function");

    // =========================================================================
    // Activation Functions
    // =========================================================================
    
    py::class_<ReLU<float>, Activation<float>, std::shared_ptr<ReLU<float>>>(m, "ReLU",
        R"pbdoc(
            Rectified Linear Unit activation: f(x) = max(0, x)
            
            Best for: Hidden layers in most networks.
            Fast to compute and helps with gradient flow.
        )pbdoc")
        .def(py::init<>(), "Create ReLU activation");

    py::class_<LeakyReLU<float>, Activation<float>, std::shared_ptr<LeakyReLU<float>>>(m, "LeakyReLU",
        R"pbdoc(
            Leaky ReLU activation: f(x) = x if x > 0, else 0.01*x
            
            Best for: Hidden layers when ReLU causes "dead neurons".
            Allows small gradient for negative inputs.
        )pbdoc")
        .def(py::init<>(), "Create Leaky ReLU activation");

    py::class_<Sigmoid<float>, Activation<float>, std::shared_ptr<Sigmoid<float>>>(m, "Sigmoid",
        R"pbdoc(
            Sigmoid activation: f(x) = 1 / (1 + exp(-x))
            
            Output range: (0, 1)
            Best for: Binary classification output layers.
        )pbdoc")
        .def(py::init<>(), "Create Sigmoid activation");

    py::class_<Tanh<float>, Activation<float>, std::shared_ptr<Tanh<float>>>(m, "Tanh",
        R"pbdoc(
            Hyperbolic tangent activation: f(x) = tanh(x)
            
            Output range: (-1, 1)
            Best for: Hidden layers when zero-centered output is desired.
        )pbdoc")
        .def(py::init<>(), "Create Tanh activation");

    py::class_<Linear<float>, Activation<float>, std::shared_ptr<Linear<float>>>(m, "Linear",
        R"pbdoc(
            Linear (identity) activation: f(x) = x
            
            Output range: unbounded
            Best for: Regression output layers.
        )pbdoc")
        .def(py::init<>(), "Create Linear activation");

    py::class_<Softmax<float>, Activation<float>, std::shared_ptr<Softmax<float>>>(m, "Softmax",
        R"pbdoc(
            Softmax activation: f(x_i) = exp(x_i) / Σ exp(x_j)
            
            Outputs probability distribution (sums to 1).
            Best for: Multi-class classification output layers.
            
            Note: Backward pass not fully implemented yet.
        )pbdoc")
        .def(py::init<>(), "Create Softmax activation");

    // =========================================================================
    // Main Neural Network Class
    // =========================================================================
    
    py::class_<PyNN3<float>>(m, "NeuralNets",
        R"pbdoc(
            Neural Network with configurable architecture.
            
            Build networks by adding layers with different activations,
            then train using NumPy arrays.
            
            Args:
                optimizer: Optimizer for weight updates (SGD, Adam)
                loss: Loss function for training (MSE, CrossEntropy)
            
            Example:
                >>> model = nn.NeuralNets(nn.Adam(0.001), nn.MSE())
                >>> model.add_layer(input_size=10, output_size=64, activation=nn.ReLU())
                >>> model.add_layer(64, 32, nn.ReLU())
                >>> model.add_layer(32, 1, nn.Linear())
        )pbdoc")
        .def(py::init<std::shared_ptr<Optimizer<float>>, std::shared_ptr<Loss<float>>>(),
             py::arg("optimizer"), py::arg("loss"),
             "Create a neural network with specified optimizer and loss function")
        
        .def("add_layer", &PyNN3<float>::add_layer,
             py::arg("input_size"), py::arg("output_size"), py::arg("activation"),
             R"pbdoc(
                 Add a dense (fully connected) layer to the network.
                 
                 Args:
                     input_size: Number of input features for this layer
                     output_size: Number of neurons (output features) for this layer
                     activation: Activation function (ReLU, Sigmoid, etc.)
                 
                 Note: Layers must be added in order. First layer's input_size should
                 match your data's feature count.
             )pbdoc")
        
        .def("train", &PyNN3<float>::train,
             py::arg("X"), py::arg("y"), py::arg("epochs"), py::arg("verbose") = true,
             R"pbdoc(
                 Train the network using backpropagation.
                 
                 Args:
                     X: Input features as NumPy array (n_samples x n_features)
                     y: Target values as NumPy array (n_samples x n_outputs)
                     epochs: Number of training epochs
                     verbose: If True, print loss each epoch (default: True)
                 
                 The training uses full-batch gradient descent.
             )pbdoc")
        
        .def("predict", &PyNN3<float>::predict,
             py::arg("X"),
             R"pbdoc(
                 Predict outputs for input data.
                 
                 Args:
                     X: Input features as NumPy array (n_samples x n_features)
                 
                 Returns:
                     NumPy array of predictions (n_samples x n_outputs)
             )pbdoc")
        
        .def_property_readonly("num_layers", &PyNN3<float>::num_layers,
             "Get the number of layers in the network");

    // =========================================================================
    // Module-level version info
    // =========================================================================
    
    m.attr("__version__") = "0.1.0";
}
