# NN3: Optimized Matrix Multiplication and Neural Network Framework

C++ implementation of matrix multiplication algorithms leveraged into a custom neural network framework, built from scratch. This project demonstrates various optimization techniques including loop unrolling, cache tiling, SIMD vectorization, OpenMP parallelization and GPU acceleration (using CUDA).

Moreover, it implements a flexible and friendly tool for experimenting DNN architectures, also usable by full Python API.

This project consists of the following main components:

1. **Optimized Matrix Multiplication Library**: Matrix structure (with basic/algebraic operations, I/O, etc.) and multiple implementations of matrix multiplication with varying optimization strategies, including GPU acceleration using CUDA. All the implementations are evaluated against OpenBLAS and cuBLAS as a reference implementations
2. **Neural Network Framework**: A flexible, template-based neural network library supporting:
   - Dense layers with customizable activations
   - Multiple activation functions (ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, Linear)
   - Loss functions (MSE, Binary Cross-Entropy, Categorical Cross-Entropy)
   - Optimizers (SGD, Adam)
   - Preprocessing utilities (One-hot encoding)
   - Metrics for binary and multi-class classification, and regression
3. **Python Bindings (PyNN3)**: High-level Python interface using pybind11, allowing seamless integration with NumPy

The library is designed for educational purposes and performance experimentation, demonstrating how low-level optimizations can dramatically improve computational efficiency.