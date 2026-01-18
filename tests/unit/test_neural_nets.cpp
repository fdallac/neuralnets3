// Code for testing matrix constructors

#include "matrix/matrix.hpp"
#include "neuralnets/neuralnets.hpp"
#include "neuralnets/normalization.hpp"
#include <gtest/gtest.h>
#include <cmath>


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
    Matrix<double> d_A(2, 3, {1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0});  // Upstream gradient (all ones)
    Matrix<double> grad = relu.backward(input, d_A);
    EXPECT_EQ(grad(0, 0), 0.0);  // x <= 0, so gradient is 0
    EXPECT_EQ(grad(0, 1), 0.0);  // x <= 0, so gradient is 0
    EXPECT_EQ(grad(0, 2), 1.0);  // x > 0, so gradient is d_A(0,2) = 1.0
    EXPECT_EQ(grad(1, 0), 0.0);  // x <= 0, so gradient is 0
    EXPECT_EQ(grad(1, 1), 1.0);  // x > 0, so gradient is d_A(1,1) = 1.0
    EXPECT_EQ(grad(1, 2), 1.0);  // x > 0, so gradient is d_A(1,2) = 1.0
}

TEST(Activations, LeakyReLUForward) {
    LeakyReLU<double> leaky_relu;
    Matrix<double> input(1, 3, {-1.0, 0.0, 1.0});
    Matrix<double> output = leaky_relu.forward(input);
    EXPECT_DOUBLE_EQ(output(0, 0), -0.01);  // 0.01 * -1
    EXPECT_DOUBLE_EQ(output(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(output(0, 2), 1.0);
}

TEST(Activations, LeakyReLUBackward) {
    LeakyReLU<double> leaky_relu;
    Matrix<double> input(1, 3, {-1.0, 0.0, 1.0});
    Matrix<double> d_A(1, 3, {1.0, 1.0, 1.0});  // Upstream gradient (all ones)
    Matrix<double> grad = leaky_relu.backward(input, d_A);
    EXPECT_DOUBLE_EQ(grad(0, 0), 0.01);  // x < 0, so gradient is 0.01 * d_A(0,0) = 0.01
    EXPECT_DOUBLE_EQ(grad(0, 1), 0.01);  // x <= 0, so gradient is 0.01 * d_A(0,1) = 0.01
    EXPECT_DOUBLE_EQ(grad(0, 2), 1.0);   // x > 0, so gradient is d_A(0,2) = 1.0
}

TEST(Activations, SigmoidForward) {
    Sigmoid<double> sigmoid;
    Matrix<double> input(1, 2, {0.0, 1000.0});
    Matrix<double> output = sigmoid.forward(input);
    EXPECT_NEAR(output(0, 0), 0.5, 1e-6);      // sigmoid(0) ≈ 0.5
    EXPECT_NEAR(output(0, 1), 1.0, 1e-6);      // sigmoid(large) ≈ 1.0
}

TEST(Activations, SigmoidBackward) {
    Sigmoid<double> sigmoid;
    Matrix<double> input(1, 2, {0.0, 2.0});
    Matrix<double> d_A(1, 2, {1.0, 1.0});  // Upstream gradient (all ones)
    Matrix<double> grad = sigmoid.backward(input, d_A);
    // sigmoid(0) = 0.5, derivative = 0.5 * (1 - 0.5) * d_A = 0.25
    EXPECT_NEAR(grad(0, 0), 0.25, 1e-6);
    // sigmoid(2) ≈ 0.8808, derivative ≈ 0.8808 * 0.1192 * d_A ≈ 0.105
    EXPECT_NEAR(grad(0, 1), 0.10499, 1e-4);
}

TEST(Activations, TanhForward) {
    Tanh<double> tanh_act;
    Matrix<double> input(1, 3, {-1.0, 0.0, 1.0});
    Matrix<double> output = tanh_act.forward(input);
    EXPECT_NEAR(output(0, 0), -0.7615941559557649, 1e-6);  // tanh(-1)
    EXPECT_NEAR(output(0, 1), 0.0, 1e-6);                   // tanh(0)
    EXPECT_NEAR(output(0, 2), 0.7615941559557649, 1e-6);   // tanh(1)
}

TEST(Activations, TanhBackward) {
    Tanh<double> tanh_act;
    Matrix<double> input(1, 3, {-1.0, 0.0, 1.0});
    Matrix<double> d_A(1, 3, {1.0, 1.0, 1.0});  // Upstream gradient (all ones)
    Matrix<double> grad = tanh_act.backward(input, d_A);
    // tanh(-1) ≈ -0.7616, derivative ≈ 1 - 0.7616^2 ≈ 0.4200
    EXPECT_NEAR(grad(0, 0), 0.41997, 1e-4);
    // tanh(0) = 0, derivative = 1 - 0^2 = 1.0
    EXPECT_NEAR(grad(0, 1), 1.0, 1e-6);
    // tanh(1) ≈ 0.7616, derivative ≈ 1 - 0.7616^2 ≈ 0.4200
    EXPECT_NEAR(grad(0, 2), 0.41997, 1e-4);
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

// ===========================================================================
// LayerNormalization Tests
// ===========================================================================

TEST(LayerNormalization, Initialization) {
    LayerNormalization<double> ln(5);
    
    // Check gamma initialized to 1
    for (std::size_t j = 0; j < 5; ++j) {
        EXPECT_DOUBLE_EQ(ln.gamma(0, j), 1.0);
    }
    
    // Check beta initialized to 0
    for (std::size_t j = 0; j < 5; ++j) {
        EXPECT_DOUBLE_EQ(ln.beta(0, j), 0.0);
    }
}

TEST(LayerNormalization, ForwardNormalization) {
    LayerNormalization<double> ln(4);
    
    // Input: single sample with features [2, 4, 6, 8]
    // Mean = 5, Variance = 5, Std = sqrt(5) ≈ 2.236
    Matrix<double> input(1, 4, {2.0, 4.0, 6.0, 8.0});
    Matrix<double> output = ln.forward(input);
    
    // Normalized values should have mean ≈ 0 and variance ≈ 1
    double mean = 0.0;
    for (std::size_t j = 0; j < 4; ++j) {
        mean += output(0, j);
    }
    mean /= 4.0;
    EXPECT_NEAR(mean, 0.0, 1e-5);
    
    double variance = 0.0;
    for (std::size_t j = 0; j < 4; ++j) {
        variance += (output(0, j) - mean) * (output(0, j) - mean);
    }
    variance /= 4.0;
    EXPECT_NEAR(variance, 1.0, 1e-5);
}

TEST(LayerNormalization, ForwardBatch) {
    LayerNormalization<double> ln(3);
    
    // Batch of 2 samples
    Matrix<double> input(2, 3, {1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0});
    Matrix<double> output = ln.forward(input);
    
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 3);
    
    // Each row should be independently normalized
    // Row 0: mean=2, var=2/3, std≈0.816
    // Row 1: mean=5, var=2/3, std≈0.816
    for (std::size_t i = 0; i < 2; ++i) {
        double row_mean = 0.0;
        for (std::size_t j = 0; j < 3; ++j) {
            row_mean += output(i, j);
        }
        row_mean /= 3.0;
        EXPECT_NEAR(row_mean, 0.0, 1e-5);
    }
}

TEST(LayerNormalization, BackwardGradient) {
    LayerNormalization<double> ln(3);
    
    Matrix<double> input(1, 3, {1.0, 2.0, 3.0});
    ln.forward(input);  // Populate cache
    
    // Gradient from upstream (all ones)
    Matrix<double> d_out(1, 3, {1.0, 1.0, 1.0});
    Matrix<double> d_input = ln.backward(d_out);
    
    // Backward should return gradient with same shape as input
    EXPECT_EQ(d_input.rows(), 1);
    EXPECT_EQ(d_input.cols(), 3);
    
    // Gradients should sum to approximately 0 (due to normalization)
    double grad_sum = 0.0;
    for (std::size_t j = 0; j < 3; ++j) {
        grad_sum += d_input(0, j);
    }
    EXPECT_NEAR(grad_sum, 0.0, 1e-5);
}

TEST(NeuralNets, AddLayerWithNormalization) {
    MSELoss<double> mse_loss;
    SGD<double> sgd_optimizer(0.01);
    NeuralNets<double> nn(sgd_optimizer, mse_loss);
    ReLU<double> relu;
    LayerNormalization<double> ln(5);
    
    nn.add_layer(3, 5, relu, &ln);
    
    EXPECT_EQ(nn.get_num_layers(), 1);
    EXPECT_NE(nn.get_layer(0).normalization, nullptr);
}

TEST(NeuralNets, ForwardPassWithNormalization) {
    MSELoss<double> mse_loss;
    SGD<double> sgd_optimizer(0.01);
    NeuralNets<double> nn(sgd_optimizer, mse_loss);
    ReLU<double> relu;
    Sigmoid<double> sigmoid;
    LayerNormalization<double> ln(4);
    
    nn.add_layer(2, 4, relu, &ln);   // With normalization
    nn.add_layer(4, 1, sigmoid);      // Without normalization
    
    Matrix<double> X(3, 2, {0.1, 0.2,
                            0.3, 0.4,
                            0.5, 0.6});
    Matrix<double> output = nn.predict(X);
    
    EXPECT_EQ(output.rows(), 3);
    EXPECT_EQ(output.cols(), 1);
    
    // Output should be in valid sigmoid range [0, 1]
    for (std::size_t i = 0; i < output.rows(); ++i) {
        EXPECT_GE(output(i, 0), 0.0);
        EXPECT_LE(output(i, 0), 1.0);
    }
}

TEST(NeuralNets, TrainWithNormalization) {
    MSELoss<double> mse_loss;
    SGD<double> sgd_optimizer(0.1);
    NeuralNets<double> nn(sgd_optimizer, mse_loss);
    ReLU<double> relu;
    Sigmoid<double> sigmoid;
    LayerNormalization<double> ln(4);
    
    nn.add_layer(2, 4, relu, &ln);
    nn.add_layer(4, 1, sigmoid);
    
    Matrix<double> X(4, 2, {0.1, 0.2,
                            0.3, 0.4,
                            0.5, 0.6,
                            0.7, 0.8});
    Matrix<double> y(4, 1, {0.0,
                            1.0,
                            1.0,
                            0.0});
    
    // Save initial gamma/beta
    Matrix<double> initial_gamma = ln.gamma;
    Matrix<double> initial_beta = ln.beta;
    
    nn.train(X, y, 50, false);
    
    // After training, gamma/beta should be updated
    bool gamma_changed = false;
    bool beta_changed = false;
    for (std::size_t j = 0; j < ln.gamma.cols(); ++j) {
        if (std::abs(ln.gamma(0, j) - initial_gamma(0, j)) > 1e-6) {
            gamma_changed = true;
        }
        if (std::abs(ln.beta(0, j) - initial_beta(0, j)) > 1e-6) {
            beta_changed = true;
        }
    }
    EXPECT_TRUE(gamma_changed || beta_changed);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}