// Unit tests for ADAM optimizer
// Tests specifically target ADAM-specific features:
// - Momentum (first moment estimates via beta1)
// - Adaptive learning rates (second moment estimates via beta2)
// - Bias correction over time
// - Convergence behavior

#include "matrix/matrix.hpp"
#include "neuralnets/optimizer.hpp"
#include <gtest/gtest.h>
#include <cmath>

// Test ADAM's first update with bias correction
TEST(AdamOptimizer, FirstUpdateBiasCorrection) {
    Adam<double> adam(0.001, 0.9, 0.999, 1e-8);
    Matrix<double> weights(1, 1, {1.0});
    Matrix<double> grad(1, 1, {1.0});
    
    double initial_weight = weights(0, 0);
    adam.update_weights(weights, grad);
    
    // First update: bias correction is applied
    // m_hat = m / (1 - beta1^1) = 0.1 / 0.1 = 1.0
    // v_hat = v / (1 - beta2^1) = 0.001 / 0.001 = 1.0
    // Expected update: -0.001 * 1.0 / (sqrt(1.0) + 1e-8) ≈ -0.001
    EXPECT_NEAR(weights(0, 0), initial_weight - 0.001, 1e-5);
}

// Test ADAM momentum accumulation
TEST(AdamOptimizer, MomentumAccumulation) {
    Adam<double> adam(0.001, 0.9, 0.999, 1e-8);
    Matrix<double> weights(1, 1, {1.0});
    Matrix<double> grad(1, 1, {1.0});
    
    // First update
    adam.update_weights(weights, grad);
    double after_first = weights(0, 0);
    
    // Second update with same gradient - momentum should cause different behavior
    adam.update_weights(weights, grad);
    double after_second = weights(0, 0);
    
    double first_change = 1.0 - after_first;
    double second_change = after_first - after_second;
    
    // Due to momentum accumulation, changes should be similar but not identical
    EXPECT_GT(second_change, 0.0);
    EXPECT_NEAR(first_change, second_change, 5e-4);  // Similar magnitude
}

// Test that ADAM handles consistent gradients with momentum
TEST(AdamOptimizer, ConsistentGradientMomentum) {
    Adam<double> adam(0.01, 0.9, 0.999, 1e-8);
    Matrix<double> weights(1, 1, {10.0});
    Matrix<double> grad(1, 1, {1.0});
    
    // Apply same gradient multiple times
    for (int i = 0; i < 10; ++i) {
        adam.update_weights(weights, grad);
    }
    
    // With momentum, the optimizer should make steady progress
    EXPECT_LT(weights(0, 0), 10.0);
    EXPECT_GT(weights(0, 0), 9.8);  // Should decrease but not as much as vanilla SGD
}

// Test ADAM with oscillating gradients (momentum should stabilize)
TEST(AdamOptimizer, OscillatingGradientDamping) {
    Adam<double> adam(0.01, 0.9, 0.999, 1e-8);
    Matrix<double> weights(1, 1, {5.0});
    
    // Apply oscillating gradients
    for (int i = 0; i < 20; ++i) {
        Matrix<double> grad(1, 1, {i % 2 == 0 ? 1.0 : -1.0});
        adam.update_weights(weights, grad);
    }
    
    // ADAM's momentum should dampen oscillations
    // Weight should remain close to initial value
    EXPECT_NEAR(weights(0, 0), 5.0, 0.5);
}

// Test ADAM with different beta1 values (momentum parameter)
TEST(AdamOptimizer, DifferentBeta1Values) {
    Matrix<double> weights_low_momentum(1, 1, {1.0});
    Matrix<double> weights_high_momentum(1, 1, {1.0});
    Matrix<double> grad(1, 1, {1.0});
    
    Adam<double> adam_low(0.001, 0.5, 0.999, 1e-8);   // Low momentum
    Adam<double> adam_high(0.001, 0.99, 0.999, 1e-8); // High momentum
    
    // Single update
    adam_low.update_weights(weights_low_momentum, grad);
    adam_high.update_weights(weights_high_momentum, grad);
    
    double change_low = 1.0 - weights_low_momentum(0, 0);
    double change_high = 1.0 - weights_high_momentum(0, 0);
    
    // Both should decrease but with slightly different magnitudes
    EXPECT_GT(change_low, 0.0);
    EXPECT_GT(change_high, 0.0);
}

// Test ADAM with different beta2 values (adaptive learning rate parameter)
TEST(AdamOptimizer, DifferentBeta2Values) {
    Matrix<double> weights1(1, 1, {1.0});
    Matrix<double> weights2(1, 1, {1.0});
    Matrix<double> grad(1, 1, {1.0});
    
    Adam<double> adam_low_beta2(0.001, 0.9, 0.9, 1e-8);    // Lower beta2
    Adam<double> adam_high_beta2(0.001, 0.9, 0.9999, 1e-8); // Higher beta2
    
    adam_low_beta2.update_weights(weights1, grad);
    adam_high_beta2.update_weights(weights2, grad);
    
    // Both should update, differences due to second moment estimation
    EXPECT_LT(weights1(0, 0), 1.0);
    EXPECT_LT(weights2(0, 0), 1.0);
}

// Test ADAM handles zero gradients correctly
TEST(AdamOptimizer, ZeroGradientHandling) {
    Adam<double> adam(0.001, 0.9, 0.999, 1e-8);
    Matrix<double> weights(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix<double> zero_grad(2, 2, {0.0, 0.0, 0.0, 0.0});
    
    adam.update_weights(weights, zero_grad);
    
    // With zero gradient, moments build toward zero, weights should stay nearly same
    EXPECT_NEAR(weights(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(weights(0, 1), 2.0, 1e-6);
    EXPECT_NEAR(weights(1, 0), 3.0, 1e-6);
    EXPECT_NEAR(weights(1, 1), 4.0, 1e-6);
}

// Test ADAM with large gradients (adaptive learning should help)
TEST(AdamOptimizer, LargeGradientAdaptation) {
    Adam<double> adam(0.001, 0.9, 0.999, 1e-8);
    Matrix<double> weights(1, 2, {1.0, 1.0});
    Matrix<double> large_grad(1, 2, {100.0, 200.0});
    
    adam.update_weights(weights, large_grad);
    
    // ADAM should adapt to large gradients, providing smaller effective steps
    // Much smaller changes than vanilla SGD would give (0.1, 0.2)
    EXPECT_LT(weights(0, 0), 1.0);
    EXPECT_LT(weights(0, 1), 1.0);
    EXPECT_GT(weights(0, 0), 0.99);  // Should be close to original
    EXPECT_GT(weights(0, 1), 0.99);
}

// Test ADAM with sparse gradients
TEST(AdamOptimizer, SparseGradients) {
    Adam<double> adam(0.001, 0.9, 0.999, 1e-8);
    Matrix<double> weights(2, 2, {1.0, 2.0, 3.0, 4.0});
    
    // First update: only some gradients are non-zero
    Matrix<double> sparse_grad1(2, 2, {1.0, 0.0, 0.0, 1.0});
    adam.update_weights(weights, sparse_grad1);
    
    // Elements with non-zero gradients should change
    EXPECT_LT(weights(0, 0), 1.0);
    EXPECT_NEAR(weights(0, 1), 2.0, 1e-6);  // No gradient
    EXPECT_NEAR(weights(1, 0), 3.0, 1e-6);  // No gradient
    EXPECT_LT(weights(1, 1), 4.0);
}

// Test ADAM bias updates
TEST(AdamOptimizer, BiasUpdateWithMomentum) {
    Adam<double> adam(0.001, 0.9, 0.999, 1e-8);
    Matrix<double> bias(1, 3, {1.0, 2.0, 3.0});
    Matrix<double> bias_grad(1, 3, {0.5, 1.0, 1.5});
    
    double initial_bias_0 = bias(0, 0);
    adam.update_bias(bias, bias_grad);
    
    // Bias should decrease due to positive gradient
    EXPECT_LT(bias(0, 0), initial_bias_0);
    EXPECT_LT(bias(0, 1), 2.0);
    EXPECT_LT(bias(0, 2), 3.0);
    
    // Changes should be small due to adaptive learning and bias correction
    EXPECT_NEAR(bias(0, 0), initial_bias_0, 0.01);
}

// Test ADAM with negative gradients
TEST(AdamOptimizer, NegativeGradients) {
    Adam<double> adam(0.001, 0.9, 0.999, 1e-8);
    Matrix<double> weights(1, 2, {1.0, 2.0});
    Matrix<double> neg_grad(1, 2, {-1.0, -2.0});
    
    adam.update_weights(weights, neg_grad);
    
    // Negative gradients should increase weights
    EXPECT_GT(weights(0, 0), 1.0);
    EXPECT_GT(weights(0, 1), 2.0);
}

// Test learning rate getter
TEST(AdamOptimizer, GetLearningRate) {
    Adam<double> adam(0.123, 0.9, 0.999, 1e-8);
    EXPECT_DOUBLE_EQ(adam.get_learning_rate(), 0.123);
}

// Test ADAM convergence with consistent gradients
TEST(AdamOptimizer, ConvergenceBehavior) {
    Adam<double> adam(0.01, 0.9, 0.999, 1e-8);
    Matrix<double> weights(1, 1, {10.0});
    Matrix<double> grad(1, 1, {1.0});
    
    double prev_weight = weights(0, 0);
    
    // Apply multiple updates
    for (int i = 0; i < 20; ++i) {
        adam.update_weights(weights, grad);
    }
    
    // Should converge steadily downward
    EXPECT_LT(weights(0, 0), prev_weight);
    EXPECT_GT(weights(0, 0), 9.5);  // Momentum prevents overshoot
}

// Test epsilon prevents division by zero
TEST(AdamOptimizer, EpsilonPreventsZeroDivision) {
    Adam<double> adam(0.001, 0.9, 0.999, 1e-8);
    Matrix<double> weights(1, 1, {1.0});
    
    // Apply many zero gradients followed by non-zero
    Matrix<double> zero_grad(1, 1, {0.0});
    for (int i = 0; i < 5; ++i) {
        adam.update_weights(weights, zero_grad);
    }
    
    Matrix<double> grad(1, 1, {1.0});
    adam.update_weights(weights, grad);
    
    // Should not crash and weight should change
    EXPECT_LT(weights(0, 0), 1.0);
}

// Test ADAM with varying gradients (simulating noisy optimization)
TEST(AdamOptimizer, NoisyGradientsSmoothing) {
    Adam<double> adam(0.01, 0.9, 0.999, 1e-8);
    Matrix<double> weights(1, 1, {5.0});
    
    // Simulate noisy gradients around mean of 1.0
    double gradients[] = {1.5, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7, 1.0};
    
    for (double g : gradients) {
        Matrix<double> grad(1, 1, {g});
        adam.update_weights(weights, grad);
    }
    
    // ADAM should smooth out noise and make steady progress
    EXPECT_LT(weights(0, 0), 5.0);
    EXPECT_GT(weights(0, 0), 4.5);
}

// Test ADAM with different matrix dimensions
TEST(AdamOptimizer, VariousMatrixDimensions) {
    Adam<double> adam(0.001, 0.9, 0.999, 1e-8);
    
    // Test different sized weight matrices
    Matrix<double> weights_small(1, 1, {1.0});
    Matrix<double> grad_small(1, 1, {1.0});
    
    Matrix<double> weights_large(3, 4, std::vector<double>(12, 1.0));
    Matrix<double> grad_large(3, 4, std::vector<double>(12, 1.0));
    
    // Both should update without issues
    adam.update_weights(weights_small, grad_small);
    EXPECT_LT(weights_small(0, 0), 1.0);
    
    adam.update_weights(weights_large, grad_large);
    for (std::size_t i = 0; i < weights_large.rows(); ++i) {
        for (std::size_t j = 0; j < weights_large.cols(); ++j) {
            EXPECT_LT(weights_large(i, j), 1.0);
        }
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

