C++ implementation of matrix multiplication algorithms leveraged into a flexible neural network framework, built from scratch. This project demonstrates various optimization techniques including loop unrolling, cache tiling, SIMD vectorization (AVX-512), and OpenMP parallelization.

This project consists of two main components:

- **Optimized Matrix Multiplication Library**: Multiple implementations of matrix multiplication with varying optimization strategies, evaluated against OpenBLAS as a reference implementation
- **Neural Network Framework**: A flexible, template-based neural network library supporting:
    - Dense layers with customizable activations
    - Multiple activation functions (ReLU, LeakyReLU, Sigmoid, Tanh, Softmax)
    - Loss functions (MSE, Cross-Entropy)
    - Optimizers (SGD with configurable learning rate)
    - Backpropagation
    
The library is designed for educational purposes and performance experimentation, demonstrating how low-level optimizations can dramatically improve computational efficiency.