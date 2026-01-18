#include "matrix/matrix.hpp"
#include "neuralnets/activation.hpp"
#include "neuralnets/loss.hpp" 
#include "neuralnets/optimizer.hpp"
#include "neuralnets/neuralnets.hpp"
#include "neuralnets/metrics.hpp"
#include "matrix/iohelper.hpp"
#include <memory>

int main(int argc, char **argv) {
    // Create a simple neural network for binary classification
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <optimizer: adam|sgd>" << std::endl;
        return 1;
    }
    std::string opt = argv[1];

    // Use smart pointer to base class for runtime polymorphism
    std::unique_ptr<Optimizer<double>> optimizer;
    
    if (opt == "adam") {
        optimizer = std::make_unique<Adam<double>>(0.1, 0.9, 0.999, 1e-8);
    } else if (opt == "sgd") {
        optimizer = std::make_unique<SGD<double>>(0.1);
    } else {
        std::cerr << "Unsupported optimizer: " << opt << std::endl;
        return 1;
    }

    BinaryCrossEntropyLoss<double> loss_function;
    NeuralNets<double> nn(*optimizer, loss_function);
    LeakyReLU<double> leaky_relu;  // Use LeakyReLU to prevent dead neurons
    Sigmoid<double> sigmoid;
    nn.add_layer(30, 128, leaky_relu);  // Input layer to hidden layer
    nn.add_layer(128, 64, leaky_relu);   // Hidden layer to hidden layer
    nn.add_layer(64, 32, leaky_relu);   // Hidden layer to hidden layer
    nn.add_layer(32, 1, sigmoid);       // Hidden layer to output layer

    // Load training data from CSV files
    Matrix<double> X_train = IOHelper<double>::read_csv("../data/preprocessed/breast_cancer/X_train.csv", ',', true);
    Matrix<double> y_train = IOHelper<double>::read_csv("../data/preprocessed/breast_cancer/y_train.csv", ',', true);

    // Train the neural network
    nn.train(X_train, y_train, 100, true);

    // Load test data from CSV files
    Matrix<double> X_test = IOHelper<double>::read_csv("../data/preprocessed/breast_cancer/X_test.csv", ',', true);
    Matrix<double> y_test = IOHelper<double>::read_csv("../data/preprocessed/breast_cancer/y_test.csv", ',', true);

    // Evaluate the neural network on test data
    Matrix<double> train_pred = nn.predict(X_train);
    Matrix<double> test_pred = nn.predict(X_test);
    BinaryAccuracy<double> accuracy_metric;
    BinaryPrecision<double> precision_metric;
    BinaryRecall<double> recall_metric;

    std::cout << "====================" << std::endl;
    double train_accuracy = accuracy_metric.eval_probs(train_pred, y_train);
    std::cout << "Train Accuracy: " << train_accuracy << std::endl;
    double test_accuracy = accuracy_metric.eval_probs(test_pred, y_test);
    std::cout << "Test Accuracy: " << test_accuracy << std::endl;
    double train_recall = recall_metric.eval_probs(train_pred, y_train);
    std::cout << "Train Recall: " << train_recall << std::endl;
    double test_recall = recall_metric.eval_probs(test_pred, y_test);
    std::cout << "Test Recall: " << test_recall << std::endl;
    double train_precision = precision_metric.eval_probs(train_pred, y_train);
    std::cout << "Train Precision: " << train_precision << std::endl;
    double test_precision = precision_metric.eval_probs(test_pred, y_test);
    std::cout << "Test Precision: " << test_precision << std::endl;

    return 0;
}