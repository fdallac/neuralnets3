/**
 * @file test_regression_neuralnet.cpp
 * @brief Regression test using Wine Quality dataset
 * 
 * This test demonstrates regression using:
 * - Linear activation for output layer
 * - Mean Squared Error loss
 * - Regression metrics (MSE, MAE)
 */

#include "matrix/matrix.hpp"
#include "neuralnets/activation.hpp"
#include "neuralnets/loss.hpp" 
#include "neuralnets/optimizer.hpp"
#include "neuralnets/neuralnets.hpp"
#include "neuralnets/metrics.hpp"
#include "matrix/iohelper.hpp"
#include <memory>
#include <iostream>
#include <cmath>

int main(int argc, char **argv) {
    // Parse command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <optimizer: adam|sgd>" << std::endl;
        return 1;
    }
    std::string opt = argv[1];

    std::cout << "=== Wine Quality Regression ===" << std::endl;
    std::cout << "Predicting wine quality score (continuous) using Linear + MSE" << std::endl;
    std::cout << std::endl;

    // =========================================================================
    // Load Data
    // =========================================================================
    std::cout << "Loading data..." << std::endl;
    
    Matrix<double> X_train = IOHelper<double>::read_csv("../data/preprocessed/wine_quality/X_train.csv", ',', true);
    Matrix<double> y_train = IOHelper<double>::read_csv("../data/preprocessed/wine_quality/y_train.csv", ',', true);
    Matrix<double> X_test = IOHelper<double>::read_csv("../data/preprocessed/wine_quality/X_test.csv", ',', true);
    Matrix<double> y_test = IOHelper<double>::read_csv("../data/preprocessed/wine_quality/y_test.csv", ',', true);

    std::cout << "  X_train shape: " << X_train.rows() << " x " << X_train.cols() << std::endl;
    std::cout << "  y_train shape: " << y_train.rows() << " x " << y_train.cols() << std::endl;
    std::cout << "  X_test shape:  " << X_test.rows() << " x " << X_test.cols() << std::endl;
    std::cout << "  y_test shape:  " << y_test.rows() << " x " << y_test.cols() << std::endl;

    // =========================================================================
    // Build Neural Network
    // =========================================================================
    std::cout << "\nBuilding neural network..." << std::endl;

    // Create optimizer based on command line argument
    std::unique_ptr<Optimizer<double>> optimizer;
    if (opt == "adam") {
        optimizer = std::make_unique<Adam<double>>(0.1, 0.9, 0.999, 1e-8);
        std::cout << "  Optimizer: Adam (lr=0.1)" << std::endl;
    } else if (opt == "sgd") {
        optimizer = std::make_unique<SGD<double>>(0.1);
        std::cout << "  Optimizer: SGD (lr=0.1)" << std::endl;
    } else {
        std::cerr << "Unsupported optimizer: " << opt << std::endl;
        return 1;
    }

    // Use Mean Squared Error for regression
    MSELoss<double> loss_function;

    // Create neural network
    NeuralNets<double> nn(*optimizer, loss_function);
    
    // Activation functions
    LeakyReLU<double> leaky_relu;
    Linear<double> linear;

    // Network architecture: 12 -> 32 -> 32 -> 16 -> 16 -> 1
    std::size_t input_size = X_train.cols();  // 12 features
    nn.add_layer(input_size, 32, leaky_relu);  // Hidden layer 1
    nn.add_layer(32, 32, leaky_relu);          // Hidden layer 2
    nn.add_layer(32, 16, leaky_relu);          // Hidden layer 3
    nn.add_layer(16, 16, leaky_relu);          // Hidden layer 4
    nn.add_layer(16, 1, linear);               // Output layer with Linear

    std::cout << "  Architecture: " << input_size << " -> 32 -> 32 -> 16 -> 16 -> 1" << std::endl;
    std::cout << "  Hidden activations: LeakyReLU" << std::endl;
    std::cout << "  Output activation: Linear" << std::endl;
    std::cout << "  Loss function: Mean Squared Error" << std::endl;

    // =========================================================================
    // Train Neural Network
    // =========================================================================
    std::cout << "\nTraining neural network..." << std::endl;
    
    int epochs = 100;
    nn.train(X_train, y_train, epochs, true);

    // =========================================================================
    // Evaluate Model
    // =========================================================================
    std::cout << "\n=== Evaluation Results ===" << std::endl;

    // Get predictions
    Matrix<double> train_pred = nn.predict(X_train);
    Matrix<double> test_pred = nn.predict(X_test);

    // Regression metrics
    MeanAbsoluteError<double> mae_metric;
    MeanSquaredError<double> mse_metric;

    // Evaluate on training set
    double train_mae = mae_metric.eval(train_pred, y_train);
    double train_mse = mse_metric.eval(train_pred, y_train);
    double train_rmse = std::sqrt(train_mse);

    // Evaluate on test set
    double test_mae = mae_metric.eval(test_pred, y_test);
    double test_mse = mse_metric.eval(test_pred, y_test);
    double test_rmse = std::sqrt(test_mse);

    std::cout << "\nTraining Set:" << std::endl;
    std::cout << "  MAE:  " << train_mae << std::endl;
    std::cout << "  MSE:  " << train_mse << std::endl;
    std::cout << "  RMSE: " << train_rmse << std::endl;

    std::cout << "\nTest Set:" << std::endl;
    std::cout << "  MAE:  " << test_mae << std::endl;
    std::cout << "  MSE:  " << test_mse << std::endl;
    std::cout << "  RMSE: " << test_rmse << std::endl;

    // =========================================================================
    // Sample Predictions
    // =========================================================================
    std::cout << "\n=== Sample Predictions (first 10 test samples) ===" << std::endl;
    
    for (std::size_t i = 0; i < 10 && i < X_test.rows(); ++i) {
        double pred_value = test_pred(i, 0);
        double true_value = y_test(i, 0);
        double error = std::abs(pred_value - true_value);
        
        std::cout << "  Sample " << i + 1 << ": ";
        std::cout << "True=" << true_value << ", ";
        std::cout << "Pred=" << pred_value;
        std::cout << " (error=" << error << ")";
        if (error < 0.5) {
            std::cout << " ✓";
        } else if (error < 1.0) {
            std::cout << " ~";
        } else {
            std::cout << " ✗";
        }
        std::cout << std::endl;
    }

    std::cout << "\n=== Test Complete ===" << std::endl;

    return 0;
}