// Code for testing matrix constructors

#include "matrix/matrix.hpp"
#include "neuralnets/neuralnets.hpp"
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

TEST(Activations, ReLUForward) {
    ReLU<double> relu;
    Matrix<double> input(2, 3, {-1.0, 0.0, 1.0,
                                -2.0, 0.5, 2.0});
    Matrix<double> output = relu.forward(input);
    EXPECT_EQ(output(0, 0), 0.0);  // max(0, -1)
    EXPECT_EQ(output(0, 1), 0.0);  // max(0, 0)
    EXPECT_EQ(output(0, 2), 1.0);  // max(0, 1)
    EXPECT_EQ(output(1, 0), 0.0);  // max(0, -2)
    EXPECT_EQ(output(1, 1), 0.5);  // max(0, 0.5)
    EXPECT_EQ(output(1, 2), 2.0);  // max(0, 2)
}

TEST(Activations, ReLUBackward) {
    ReLU<double> relu;
    Matrix<double> input(2, 3, {-1.0, 0.0, 1.0,
                                -2.0, 0.5, 2.0});
    Matrix<double> grad = relu.backward(input);
    EXPECT_EQ(grad(0, 0), 0.0);  // x <= 0
    EXPECT_EQ(grad(0, 1), 0.0);  // x <= 0
    EXPECT_EQ(grad(0, 2), 1.0);  // x > 0
    EXPECT_EQ(grad(1, 0), 0.0);  // x <= 0
    EXPECT_EQ(grad(1, 1), 1.0);  // x > 0
    EXPECT_EQ(grad(1, 2), 1.0);  // x > 0
}

TEST(Activations, LeakyReLUForward) {
    LeakyReLU<double> leaky_relu;
    Matrix<double> input(1, 3, {-1.0, 0.0, 1.0});
    Matrix<double> output = leaky_relu.forward(input);
    EXPECT_DOUBLE_EQ(output(0, 0), -0.01);  // 0.01 * -1
    EXPECT_DOUBLE_EQ(output(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(output(0, 2), 1.0);
}

TEST(Activations, SigmoidForward) {
    Sigmoid<double> sigmoid;
    Matrix<double> input(1, 2, {0.0, 1000.0});
    Matrix<double> output = sigmoid.forward(input);
    EXPECT_NEAR(output(0, 0), 0.5, 1e-6);      // sigmoid(0) ≈ 0.5
    EXPECT_NEAR(output(0, 1), 1.0, 1e-6);      // sigmoid(large) ≈ 1.0
}

TEST(Activations, TanhForward) {
    Tanh<double> tanh_act;
    Matrix<double> input(1, 3, {-1.0, 0.0, 1.0});
    Matrix<double> output = tanh_act.forward(input);
    EXPECT_NEAR(output(0, 0), -0.7615941559557649, 1e-6);  // tanh(-1)
    EXPECT_NEAR(output(0, 1), 0.0, 1e-6);                   // tanh(0)
    EXPECT_NEAR(output(0, 2), 0.7615941559557649, 1e-6);   // tanh(1)
}

TEST(Loss, MSEForward) {
    MSELoss<double> mse;
    Matrix<double> predictions(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix<double> targets(2, 2, {1.5, 2.5, 2.5, 3.5});
    double loss = mse.forward(predictions, targets);
    // MSE = mean((pred - target)^2) = mean(0.25, 0.25, 0.25, 0.25) = 0.25
    EXPECT_NEAR(loss, 0.25, 1e-6);
}

TEST(Loss, MSEBackward) {
    MSELoss<double> mse;
    Matrix<double> predictions(2, 1, {2.0, 4.0});
    Matrix<double> targets(2, 1, {1.0, 3.0});
    Matrix<double> grad = mse.backward(predictions, targets);
    // grad = 2 * (pred - target) / n = 2 * (1, 1) / 2 = (1, 1)
    EXPECT_NEAR(grad(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(grad(1, 0), 1.0, 1e-6);
}

TEST(Loss, BinaryCrossEntropyForward) {
    BinaryCrossEntropyLoss<double> bce;
    Matrix<double> predictions(2, 1, {0.9, 0.1});
    Matrix<double> targets(2, 1, {1.0, 0.0});
    double loss = bce.forward(predictions, targets);
    // Should be a small positive value
    EXPECT_GT(loss, 0.0);
    EXPECT_LT(loss, 1.0);
}

TEST(NeuralNets, MultiLayerNetwork) {
    MSELoss<double> mse_loss;
    SGD<double> sgd_optimizer(0.1);
    NeuralNets<double> nn(sgd_optimizer, mse_loss);
    LeakyReLU<double> leaky_relu;
    Sigmoid<double> sigmoid;
    
    nn.add_layer(4, 8, leaky_relu);
    nn.add_layer(8, 4, leaky_relu);
    nn.add_layer(4, 2, leaky_relu);
    nn.add_layer(2, 1, sigmoid);
    
    EXPECT_EQ(nn.get_num_layers(), 4);
    EXPECT_EQ(nn.get_layer(0).W.rows(), 4);
    EXPECT_EQ(nn.get_layer(0).W.cols(), 8);
    EXPECT_EQ(nn.get_layer(3).W.rows(), 2);
    EXPECT_EQ(nn.get_layer(3).W.cols(), 1);
}

TEST(NeuralNets, WeightsInitialization) {
    MSELoss<double> mse_loss;
    SGD<double> sgd_optimizer(0.01);
    NeuralNets<double> nn(sgd_optimizer, mse_loss);
    ReLU<double> relu;
    
    nn.add_layer(10, 20, relu);
    
    // Check that weights are initialized (not all zeros)
    bool has_non_zero = false;
    for (std::size_t i = 0; i < nn.get_layer(0).W.rows() && !has_non_zero; ++i) {
        for (std::size_t j = 0; j < nn.get_layer(0).W.cols() && !has_non_zero; ++j) {
            if (nn.get_layer(0).W(i, j) != 0.0) {
                has_non_zero = true;
            }
        }
    }
    EXPECT_TRUE(has_non_zero);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}