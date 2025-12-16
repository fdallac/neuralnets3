// Unit tests for optimizers

#include "matrix/matrix.hpp"
#include "neuralnets/optimizer.hpp"
#include <gtest/gtest.h>

TEST(Optimizer, SGDUpdateWeights) {
    SGD<double> sgd(0.1);  // learning rate = 0.1
    Matrix<double> weights(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix<double> gradients(2, 2, {0.1, 0.2, 0.3, 0.4});
    
    sgd.update_weights(weights, gradients);
    
    // weights = weights - lr * gradients
    EXPECT_NEAR(weights(0, 0), 0.99, 1e-6);  // 1.0 - 0.1*0.1
    EXPECT_NEAR(weights(0, 1), 1.98, 1e-6);  // 2.0 - 0.1*0.2
    EXPECT_NEAR(weights(1, 0), 2.97, 1e-6);  // 3.0 - 0.1*0.3
    EXPECT_NEAR(weights(1, 1), 3.96, 1e-6);  // 4.0 - 0.1*0.4
}

TEST(Optimizer, SGDUpdateBias) {
    SGD<double> sgd(0.1);
    Matrix<double> bias(1, 3, {1.0, 2.0, 3.0});
    Matrix<double> bias_grad(1, 3, {0.5, 1.0, 1.5});
    
    sgd.update_bias(bias, bias_grad);
    
    EXPECT_NEAR(bias(0, 0), 0.95, 1e-6);  // 1.0 - 0.1*0.5
    EXPECT_NEAR(bias(0, 1), 1.90, 1e-6);  // 2.0 - 0.1*1.0
    EXPECT_NEAR(bias(0, 2), 2.85, 1e-6);  // 3.0 - 0.1*1.5
}

TEST(Optimizer, SGDZeroGradient) {
    SGD<double> sgd(0.1);
    Matrix<double> weights(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix<double> zero_grad(2, 2, {0.0, 0.0, 0.0, 0.0});
    
    sgd.update_weights(weights, zero_grad);
    
    // Weights should remain unchanged with zero gradient
    EXPECT_EQ(weights(0, 0), 1.0);
    EXPECT_EQ(weights(0, 1), 2.0);
    EXPECT_EQ(weights(1, 0), 3.0);
    EXPECT_EQ(weights(1, 1), 4.0);
}

TEST(Optimizer, SGDLargeGradient) {
    SGD<double> sgd(0.01);  // Small learning rate
    Matrix<double> weights(1, 2, {1.0, 1.0});
    Matrix<double> large_grad(1, 2, {100.0, 200.0});
    
    sgd.update_weights(weights, large_grad);
    
    EXPECT_NEAR(weights(0, 0), 0.0, 1e-6);   // 1.0 - 0.01*100
    EXPECT_NEAR(weights(0, 1), -1.0, 1e-6);  // 1.0 - 0.01*200
}

TEST(Optimizer, SGDDifferentLearningRates) {
    Matrix<double> weights1(1, 2, {1.0, 1.0});
    Matrix<double> weights2(1, 2, {1.0, 1.0});
    Matrix<double> grad(1, 2, {1.0, 1.0});
    
    SGD<double> sgd_small(0.01);
    SGD<double> sgd_large(0.1);
    
    sgd_small.update_weights(weights1, grad);
    sgd_large.update_weights(weights2, grad);
    
    // Small learning rate: smaller change
    EXPECT_NEAR(weights1(0, 0), 0.99, 1e-6);
    EXPECT_NEAR(weights1(0, 1), 0.99, 1e-6);
    
    // Large learning rate: larger change
    EXPECT_NEAR(weights2(0, 0), 0.9, 1e-6);
    EXPECT_NEAR(weights2(0, 1), 0.9, 1e-6);
}

TEST(Optimizer, SGDGetLearningRate) {
    SGD<double> sgd(0.123);
    EXPECT_DOUBLE_EQ(sgd.get_learning_rate(), 0.123);
}

TEST(Optimizer, SGDMultipleUpdates) {
    SGD<double> sgd(0.1);
    Matrix<double> weights(1, 1, {10.0});
    Matrix<double> grad(1, 1, {1.0});
    
    // Multiple gradient descent steps
    for (int i = 0; i < 5; ++i) {
        sgd.update_weights(weights, grad);
    }
    
    // After 5 steps: 10.0 - 5 * 0.1 * 1.0 = 9.5
    EXPECT_NEAR(weights(0, 0), 9.5, 1e-6);
}

TEST(Optimizer, SGDNegativeGradient) {
    SGD<double> sgd(0.1);
    Matrix<double> weights(1, 2, {1.0, 2.0});
    Matrix<double> grad(1, 2, {-0.5, -1.0});
    
    sgd.update_weights(weights, grad);
    
    // Negative gradient means weights should increase
    EXPECT_NEAR(weights(0, 0), 1.05, 1e-6);  // 1.0 - 0.1*(-0.5)
    EXPECT_NEAR(weights(0, 1), 2.10, 1e-6);  // 2.0 - 0.1*(-1.0)
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
