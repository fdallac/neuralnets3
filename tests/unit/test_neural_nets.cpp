// Code for testing matrix constructors

#include "matrix.hpp"
#include "neuralnets.hpp"
#include <gtest/gtest.h>


TEST(NeuralNets, AddLayer) {
    MSELoss<double> mse_loss;
    SGD<double> sgd_optimizer(0.01);
    NeuralNets<double> nn(sgd_optimizer, mse_loss);
    ReLU<double> relu;
    Sigmoid<double> sigmoid;
    nn.add_layer(3, 5, relu);
    nn.add_layer(5, 2, sigmoid);
    EXPECT_EQ(nn.get_num_layers(), 2);
    EXPECT_EQ(nn.get_layer(0).W.rows(), 3);
    EXPECT_EQ(nn.get_layer(0).W.cols(), 5);
    EXPECT_EQ(nn.get_layer(1).W.rows(), 5);
    EXPECT_EQ(nn.get_layer(1).W.cols(), 2);
}


TEST(NeuralNets, ForwardPass) {
    MSELoss<double> mse_loss;
    SGD<double> sgd_optimizer(0.01);
    NeuralNets<double> nn(sgd_optimizer, mse_loss);
    ReLU<double> relu;
    Sigmoid<double> sigmoid;
    nn.add_layer(2, 3, relu);
    nn.add_layer(3, 1, sigmoid);

    Matrix<double> X(4, 2, {0.1, 0.2,
                            0.3, 0.4,
                            0.5, 0.6,
                            0.7, 0.8});
    Matrix<double> output = nn.predict(X);
    EXPECT_EQ(output.rows(), 4);
    EXPECT_EQ(output.cols(), 1);
}


TEST(NeuralNets, Train) {
    MSELoss<double> mse_loss;
    SGD<double> sgd_optimizer(0.01);
    NeuralNets<double> nn(sgd_optimizer, mse_loss);
    ReLU<double> relu;
    Sigmoid<double> sigmoid;
    nn.add_layer(2, 3, relu);
    nn.add_layer(3, 1, sigmoid);

    Matrix<double> X(4, 2, {0.1, 0.2,
                            0.3, 0.4,
                            0.5, 0.6,
                            0.7, 0.8});
    Matrix<double> y(4, 1, {0.0,
                            1.0,
                            1.0,
                            0.0});
    nn.train(X, y, 10, false);
    Matrix<double> output = nn.predict(X);
    EXPECT_EQ(output.rows(), 4);
    EXPECT_EQ(output.cols(), 1);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}