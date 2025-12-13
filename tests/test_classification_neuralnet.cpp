#include "matrix.hpp"
#include "activation.hpp"
#include "loss.hpp" 
#include "optimizer.hpp"
#include "neuralnets.hpp"
#include "metrics.hpp"
#include "utils/iohelper.hpp"

int main(int argc, char **argv) {
    // Create a simple neural network for binary classification
    BinaryCrossEntropyLoss<double> loss_function;
    SGD<double> sgd_optimizer(0.1);
    NeuralNets<double> nn(sgd_optimizer, loss_function);
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
    Matrix<double> predictions = nn.predict(X_test);
    double test_loss = loss_function.forward(predictions, y_test);
    std::cout << "====================" << std::endl;
    Accuracy<double> accuracy_metric;
    Precision<double> precision_metric;
    Recall<double> recall_metric;
    double train_accuracy = accuracy_metric.eval_probs(nn.predict(X_train), y_train);
    std::cout << "Train Accuracy: " << train_accuracy << std::endl;
    double test_accuracy = accuracy_metric.eval_probs(predictions, y_test);
    std::cout << "Test Accuracy: " << test_accuracy << std::endl;
    double train_recall = recall_metric.eval_probs(nn.predict(X_train), y_train);
    std::cout << "Train Recall: " << train_recall << std::endl;
    double test_recall = recall_metric.eval_probs(predictions, y_test);
    std::cout << "Test Recall: " << test_recall << std::endl;
    double train_precision = precision_metric.eval_probs(nn.predict(X_train), y_train);
    std::cout << "Train Precision: " << train_precision << std::endl;
    double test_precision = precision_metric.eval_probs(predictions, y_test);
    std::cout << "Test Precision: " << test_precision << std::endl;

    return 0;
}