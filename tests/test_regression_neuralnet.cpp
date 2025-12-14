#include "matrix.hpp"
#include "activation.hpp"
#include "loss.hpp" 
#include "optimizer.hpp"
#include "neuralnets.hpp"
#include "metrics.hpp"
#include "utils/iohelper.hpp"

int main(int argc, char **argv) {
    // Create a simple neural network for binary classification
    MSELoss<double> loss_function;
    SGD<double> sgd_optimizer(0.1);
    NeuralNets<double> nn(sgd_optimizer, loss_function);
    LeakyReLU<double> leaky_relu;  // Use LeakyReLU to prevent dead neurons
    Linear<double> linear;
    nn.add_layer(12, 32, leaky_relu);   // Input layer to hidden layer
    nn.add_layer(32, 32, leaky_relu);   // Hidden layer to hidden layer
    nn.add_layer(32, 16, leaky_relu);   // Hidden layer to hidden layer
    nn.add_layer(16, 16, leaky_relu);   // Hidden layer to hidden layer
    nn.add_layer(16, 1, linear);        // Hidden layer to output layer

    // Load training data from CSV files
    Matrix<double> X_train = IOHelper<double>::read_csv("../data/preprocessed/wine_quality/X_train.csv", ',', true);
    Matrix<double> y_train = IOHelper<double>::read_csv("../data/preprocessed/wine_quality/y_train.csv", ',', true);

    // Train the neural network
    nn.train(X_train, y_train, 100, true);

    // Load test data from CSV files
    Matrix<double> X_test = IOHelper<double>::read_csv("../data/preprocessed/wine_quality/X_test.csv", ',', true);
    Matrix<double> y_test = IOHelper<double>::read_csv("../data/preprocessed/wine_quality/y_test.csv", ',', true);

    // Evaluate the neural network on test data
    Matrix<double> train_pred = nn.predict(X_train);
    Matrix<double> test_pred = nn.predict(X_test);
    MeanAbsoluteError<double> mae;
    MeanSquaredError<double> mse;

    std::cout << "====================" << std::endl;
    double train_mae = mae.eval(train_pred, y_train);
    std::cout << "Train MAE: " << train_mae << std::endl;
    double test_mae = mae.eval(test_pred, y_test);
    std::cout << "Test MAE: " << test_mae << std::endl;
    double train_mse = mse.eval(train_pred, y_train);
    std::cout << "Train MSE: " << train_mse << std::endl;
    double test_mse = mse.eval(test_pred, y_test);
    std::cout << "Test MSE: " << test_mse << std::endl;

    return 0;
}