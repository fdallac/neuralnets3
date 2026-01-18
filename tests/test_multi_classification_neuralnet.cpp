/**
 * @file test_multi_classification_neuralnet.cpp
 * @brief Multi-class classification test using Wine Quality dataset
 * 
 * This test demonstrates multi-class classification using:
 * - Softmax activation for output layer
 * - Categorical Cross-Entropy loss
 * - Multi-class classification metrics (accuracy, precision, recall)
 */

#include "matrix/matrix.hpp"
#include "neuralnets/activation.hpp"
#include "neuralnets/loss.hpp" 
#include "neuralnets/optimizer.hpp"
#include "neuralnets/neuralnets.hpp"
#include "neuralnets/metrics.hpp"
#include "neuralnets/preprocessing.hpp"
#include "neuralnets/normalization.hpp"
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

    std::cout << "=== Wine Quality Multi-Class Classification ===" << std::endl;
    std::cout << "Predicting wine quality level (3-9) using Softmax + CCE" << std::endl;
    std::cout << std::endl;

    // =========================================================================
    // Load Data
    // =========================================================================
    std::cout << "Loading data..." << std::endl;
    
    Matrix<double> X_train = IOHelper<double>::read_csv("../data/preprocessed/wine_quality/X_train.csv", ',', true);
    Matrix<double> y_train_raw = IOHelper<double>::read_csv("../data/preprocessed/wine_quality/y_train.csv", ',', true);
    Matrix<double> X_test = IOHelper<double>::read_csv("../data/preprocessed/wine_quality/X_test.csv", ',', true);
    Matrix<double> y_test_raw = IOHelper<double>::read_csv("../data/preprocessed/wine_quality/y_test.csv", ',', true);

    std::cout << "  X_train shape: " << X_train.rows() << " x " << X_train.cols() << std::endl;
    std::cout << "  y_train shape: " << y_train_raw.rows() << " x " << y_train_raw.cols() << std::endl;
    std::cout << "  X_test shape:  " << X_test.rows() << " x " << X_test.cols() << std::endl;
    std::cout << "  y_test shape:  " << y_test_raw.rows() << " x " << y_test_raw.cols() << std::endl;

    // =========================================================================
    // Preprocess Labels: One-Hot Encoding
    // =========================================================================
    std::cout << "\nOne-hot encoding labels..." << std::endl;
    
    OneHotEncoder<double> encoder;
    Matrix<double> y_train = encoder.fit_transform(y_train_raw);
    Matrix<double> y_test = encoder.transform(y_test_raw);

    std::size_t num_classes = encoder.get_num_classes();
    std::cout << "  Number of classes: " << num_classes << std::endl;
    std::cout << "  Classes: ";
    for (const auto& c : encoder.get_classes()) {
        std::cout << static_cast<int>(c) << " ";
    }
    std::cout << std::endl;
    std::cout << "  y_train encoded shape: " << y_train.rows() << " x " << y_train.cols() << std::endl;
    std::cout << "  y_test encoded shape:  " << y_test.rows() << " x " << y_test.cols() << std::endl;

    // =========================================================================
    // Build Neural Network
    // =========================================================================
    std::cout << "\nBuilding neural network..." << std::endl;

    // Create optimizer based on command line argument
    std::unique_ptr<Optimizer<double>> optimizer;
    if (opt == "adam") {
        optimizer = std::make_unique<Adam<double>>(0.01, 0.9, 0.999, 1e-8);
        std::cout << "  Optimizer: Adam (lr=0.01)" << std::endl;
    } else if (opt == "sgd") {
        optimizer = std::make_unique<SGD<double>>(0.1);
        std::cout << "  Optimizer: SGD (lr=0.1)" << std::endl;
    } else {
        std::cerr << "Unsupported optimizer: " << opt << std::endl;
        return 1;
    }

    // Use Categorical Cross-Entropy for multi-class classification
    CategoricalCrossEntropyLoss<double> loss_function;

    // Create neural network
    NeuralNets<double> nn(*optimizer, loss_function);
    
    // Activation functions
    LeakyReLU<double> leaky_relu;
    Softmax<double> softmax;

    // Layer normalization for hidden layers
    LayerNormalization<double> ln1(128); // After first hidden layer
    LayerNormalization<double> ln2(64); // After second hidden layer
    LayerNormalization<double> ln3(32); // After third hidden layer

    // Network architecture: 12 -> 128 (LN) -> 64 (LN) -> 32 (LN) -> 7 (num_classes)
    std::size_t input_size = X_train.cols();  // 12 features
    nn.add_layer(input_size, 128, leaky_relu, &ln1);  // Hidden layer 1 with LayerNorm
    nn.add_layer(128, 64, leaky_relu, &ln2);          // Hidden layer 2 with LayerNorm
    nn.add_layer(64,  32, leaky_relu, &ln3);          // Hidden layer 3 with LayerNorm
    nn.add_layer(32, num_classes, softmax);           // Output layer with Softmax (no normalization)

    std::cout << "  Architecture: " << input_size << " -> 128 (LN) -> 64 (LN) -> 32 (LN) -> " << num_classes << std::endl;
    std::cout << "  Hidden activations: LeakyReLU with LayerNormalization" << std::endl;
    std::cout << "  Output activation: Softmax" << std::endl;
    std::cout << "  Loss function: Categorical Cross-Entropy" << std::endl;

    // =========================================================================
    // Train Neural Network
    // =========================================================================
    std::cout << "\nTraining neural network..." << std::endl;
    
    int epochs = 200;
    nn.train(X_train, y_train, epochs, true);

    // =========================================================================
    // Evaluate Model
    // =========================================================================
    std::cout << "\n=== Evaluation Results ===" << std::endl;

    // Get predictions (probability distributions)
    Matrix<double> train_pred = nn.predict(X_train);
    Matrix<double> test_pred = nn.predict(X_test);

    // Multi-class metrics
    MultiClassAccuracy<double> accuracy_metric;
    MultiClassPrecision<double> precision_metric;
    MultiClassRecall<double> recall_metric;

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
    std::cout << "  Precision: " << train_precision * 100 << "% (macro-averaged)" << std::endl;
    std::cout << "  Recall:    " << train_recall * 100 << "% (macro-averaged)" << std::endl;

    std::cout << "\nTest Set:" << std::endl;
    std::cout << "  Accuracy:  " << test_accuracy * 100 << "%" << std::endl;
    std::cout << "  Precision: " << test_precision * 100 << "% (macro-averaged)" << std::endl;
    std::cout << "  Recall:    " << test_recall * 100 << "% (macro-averaged)" << std::endl;

    // =========================================================================
    // Sample Predictions
    // =========================================================================
    std::cout << "\n=== Sample Predictions (first 5 test samples) ===" << std::endl;
    
    for (std::size_t i = 0; i < 5 && i < X_test.rows(); ++i) {
        // Get predicted class (argmax)
        std::size_t pred_class_idx = 0;
        double max_prob = test_pred(i, 0);
        for (std::size_t j = 1; j < num_classes; ++j) {
            if (test_pred(i, j) > max_prob) {
                max_prob = test_pred(i, j);
                pred_class_idx = j;
            }
        }
        
        // Get true class (argmax of one-hot)
        std::size_t true_class_idx = 0;
        for (std::size_t j = 0; j < num_classes; ++j) {
            if (y_test(i, j) > 0.5) {
                true_class_idx = j;
                break;
            }
        }
        
        // Map back to original class values
        int pred_quality = static_cast<int>(encoder.get_classes()[pred_class_idx]);
        int true_quality = static_cast<int>(encoder.get_classes()[true_class_idx]);
        
        std::cout << "  Sample " << i + 1 << ": ";
        std::cout << "True=" << true_quality << ", Pred=" << pred_quality;
        std::cout << " (prob=" << max_prob * 100 << "%)";
        if (pred_quality == true_quality) {
            std::cout << " ✓";
        } else {
            std::cout << " ✗";
        }
        std::cout << std::endl;
    }

    std::cout << "\n=== Test Complete ===" << std::endl;

    return 0;
}
