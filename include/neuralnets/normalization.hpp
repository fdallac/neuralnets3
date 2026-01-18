/**
 * @file layernorm.hpp
 * @brief Layer Normalization
 */
#pragma once
#include "matrix/matrix.hpp"
#include <cmath>


template<typename T>
class Normalization {
public:
    virtual Matrix<T> forward(const Matrix<T>& X) = 0;
    virtual Matrix<T> backward(const Matrix<T>& d_Out) = 0;
    virtual ~Normalization() = default;
};


template<typename T>
class None : public Normalization<T> {
public:
    Matrix<T> forward(const Matrix<T>& X) override {
        return X;
    }

    Matrix<T> backward(const Matrix<T>& d_Out) override {
        return d_Out;
    }
};


template<typename T>
class LayerNormalization : public Normalization<T> {
public:
    // Learnable Parameters
    Matrix<T> gamma; // Scale
    Matrix<T> beta;  // Shift

    // Gradients
    Matrix<T> d_gamma;
    Matrix<T> d_beta;

    // Cache for backward pass
    Matrix<T> cache_X_norm; // Normalized input
    Matrix<T> cache_std_inv; // 1 / sigma

    T epsilon = static_cast<T>(1e-5);

    LayerNormalization(size_t features) {
        // Gamma initialized to 1, Beta to 0
        gamma = Matrix<T>(1, features);
        beta = Matrix<T>(1, features);
        
        gamma.fill_ones(); // Ensure you implement fill_ones in Matrix
        beta.fill_uniform_noise(0, 0); // Effectively zero
        
        d_gamma = Matrix<T>(1, features);
        d_beta = Matrix<T>(1, features);
    }

    // Forward: Y = (X - mean) / std * gamma + beta
    Matrix<T> forward(const Matrix<T>& X) {
        size_t batch_size = X.rows();
        size_t features = X.cols();

        Matrix<T> Out(batch_size, features);
        cache_X_norm = Matrix<T>(batch_size, features);
        cache_std_inv = Matrix<T>(batch_size, 1);

        for(size_t i = 0; i < batch_size; ++i) {
            // 1. Calculate Mean
            T sum = 0;
            for(size_t j = 0; j < features; ++j) sum += X(i,j);
            T mean = sum / features;

            // 2. Calculate Variance
            T var_sum = 0;
            for(size_t j = 0; j < features; ++j) {
                T diff = X(i,j) - mean;
                var_sum += diff * diff;
            }
            T variance = var_sum / features;
            
            // 3. Inverse Std Dev
            T std_inv = static_cast<T>(1) / std::sqrt(variance + epsilon);
            cache_std_inv(i, 0) = std_inv;

            // 4. Normalize and Scale
            for(size_t j = 0; j < features; ++j) {
                T x_norm = (X(i,j) - mean) * std_inv;
                cache_X_norm(i, j) = x_norm;
                
                // Apply Gamma and Beta
                Out(i, j) = (gamma(0, j) * x_norm) + beta(0, j);
            }
        }
        return Out;
    }

    // Backward Pass
    Matrix<T> backward(const Matrix<T>& d_Out) {
        size_t batch = d_Out.rows();
        size_t feat = d_Out.cols();
        
        Matrix<T> d_X(batch, feat);

        // Reset parameter gradients
        d_gamma.fill_uniform_noise(0, 0); // Reset to 0
        d_beta.fill_uniform_noise(0, 0);

        for(size_t i = 0; i < batch; ++i) {
            T d_std_inv = 0;
            T d_mean = 0;
            T sum_d_out_gamma = 0;

            for(size_t j = 0; j < feat; ++j) {
                T dout = d_Out(i, j);
                
                // 1. Gradients for Gamma and Beta
                d_gamma(0, j) += dout * cache_X_norm(i, j);
                d_beta(0, j)  += dout;

                // 2. Backprop through Scale/Shift
                T d_x_norm = dout * gamma(0, j);

                // Accumulate internals
                sum_d_out_gamma += d_x_norm;
                d_std_inv += d_x_norm * (cache_X_norm(i, j) / cache_std_inv(i, 0)); // Approx reconstruction
            }
            
            // Exact LayerNorm Gradient formulation
            T inv_N = static_cast<T>(1) / feat;
            T term1 = sum_d_out_gamma; // Sum(dy * gamma)
            
            // Compute Sum(dy * gamma * x_norm)
            T term2 = 0;
            for(size_t j = 0; j < feat; ++j) {
                term2 += (d_Out(i, j) * gamma(0, j)) * cache_X_norm(i, j);
            }

            // Final dX calculation
            for(size_t j = 0; j < feat; ++j) {
                T d_x_norm = d_Out(i, j) * gamma(0, j);
                T val = (feat * d_x_norm) - term1 - (cache_X_norm(i, j) * term2);
                d_X(i, j) = val * (cache_std_inv(i, 0) * inv_N);
            }
        }
        
        return d_X;
    }
    
    // Helper to update params (used by Optimizer)
    void update(const Optimizer<T>& opt) {
        // You'll need to expose these to your Optimizer
        // optimizer.update(gamma, d_gamma);
        // optimizer.update(beta, d_beta);
    }
};