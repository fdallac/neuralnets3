/**
 * @file test_classification_neuralnet.cpp
 * @brief Binary classification test using Breast Cancer Wisconsin dataset
 * 
 * This test demonstrates binary classification using:
 * - Sigmoid activation for output layer
 * - Binary Cross-Entropy loss
 * - Binary classification metrics (accuracy, precision, recall)
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

int main(int argc, char **argv) {
    // Parse command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <optimizer: adam|sgd>" << std::endl;
        return 1;
    }
    std::string opt = argv[1];

    std::cout << "=== Breast Cancer Binary Classification ===" << std::endl;
    std::cout << "Predicting malignant (1) or benign (0) tumors using Sigmoid + BCE" << std::endl;
    std::cout << std::endl;

    // =========================================================================
    // Load Data
    // =========================================================================
    std::cout << "Loading data..." << std::endl;
    
    Matrix<double> X_train = IOHelper<double>::read_csv("../data/preprocessed/breast_cancer/X_train.csv", ',', true);
    Matrix<double> y_train = IOHelper<double>::read_csv("../data/preprocessed/breast_cancer/y_train.csv", ',', true);
    Matrix<double> X_test = IOHelper<double>::read_csv("../data/preprocessed/breast_cancer/X_test.csv", ',', true);
    Matrix<double> y_test = IOHelper<double>::read_csv("../data/preprocessed/breast_cancer/y_test.csv", ',', true);

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

    // Use Binary Cross-Entropy for binary classification
    BinaryCrossEntropyLoss<double> loss_function;

    // Create neural network
    NeuralNets<double> nn(*optimizer, loss_function);
    
    // Activation functions
    LeakyReLU<double> leaky_relu;
    Sigmoid<double> sigmoid;

    // Network architecture: 30 -> 128 -> 64 -> 32 -> 1
    std::size_t input_size = X_train.cols();  // 30 features
    nn.add_layer(input_size, 128, leaky_relu);  // Hidden layer 1
    nn.add_layer(128, 64, leaky_relu);          // Hidden layer 2
    nn.add_layer(64, 32, leaky_relu);           // Hidden layer 3
    nn.add_layer(32, 1, sigmoid);               // Output layer with Sigmoid

    std::cout << "  Architecture: " << input_size << " -> 128 -> 64 -> 32 -> 1" << std::endl;
    std::cout << "  Hidden activations: LeakyReLU" << std::endl;
    std::cout << "  Output activation: Sigmoid" << std::endl;
    std::cout << "  Loss function: Binary Cross-Entropy" << std::endl;

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

    // Get predictions (probabilities)
    Matrix<double> train_pred = nn.predict(X_train);
    Matrix<double> test_pred = nn.predict(X_test);

    // Binary classification metrics
    BinaryAccuracy<double> accuracy_metric;
    BinaryPrecision<double> precision_metric;
    BinaryRecall<double> recall_metric;

    // Evaluate on training set
    double train_accuracy = accuracy_metric.eval_probs(train_pred, y_train);
    double train_precision = precision_metric.eval_probs(train_pred, y_train);
    double train_recall = recall_metric.eval_probs(train_pred, y_train);

    // Evaluate on test set
    double test_accuracy = accuracy_metric.eval_probs(test_pred, y_test);
    double test_precision = precision_metric.eval_probs(test_pred, y_test);
    double test_recall = recall_metric.eval_probs(test_pred, y_test);

    std::cout << "\nTraining Set:" << std::endl;
    std::cout << "  Accuracy:  " << train_accuracy * 100 << "%" << std::endl;
    std::cout << "  Precision: " << train_precision * 100 << "%" << std::endl;
    std::cout << "  Recall:    " << train_recall * 100 << "%" << std::endl;

    std::cout << "\nTest Set:" << std::endl;
    std::cout << "  Accuracy:  " << test_accuracy * 100 << "%" << std::endl;
    std::cout << "  Precision: " << test_precision * 100 << "%" << std::endl;
    std::cout << "  Recall:    " << test_recall * 100 << "%" << std::endl;

    // =========================================================================
    // Sample Predictions
    // =========================================================================
    std::cout << "\n=== Sample Predictions (first 10 test samples) ===" << std::endl;
    
    for (std::size_t i = 0; i < 10 && i < X_test.rows(); ++i) {
        double prob = test_pred(i, 0);
        int pred_class = prob >= 0.5 ? 1 : 0;
        int true_class = static_cast<int>(y_test(i, 0));
        
        std::cout << "  Sample " << i + 1 << ": ";
        std::cout << "True=" << true_class << " (" << (true_class == 1 ? "malignant" : "benign") << "), ";
        std::cout << "Pred=" << pred_class << " (" << (pred_class == 1 ? "malignant" : "benign") << ")";
        std::cout << " (prob=" << prob * 100 << "%)";
        if (pred_class == true_class) {
            std::cout << " ✓";
        } else {
            std::cout << " ✗";
        }
        std::cout << std::endl;
    }

    std::cout << "\n=== Test Complete ===" << std::endl;

    return 0;
}