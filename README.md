# NN3: Optimized Matrix Multiplication and Neural Network Framework

C++ implementation of matrix multiplication algorithms leveraged into a flexible neural network framework, built from scratch. This project demonstrates various optimization techniques including loop unrolling, cache tiling, SIMD vectorization (AVX-512), and OpenMP parallelization.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Matrix Multiplication Optimization](#matrix-multiplication-optimization)
  - [Implemented Algorithms](#implemented-algorithms)
  - [Benchmarking Results](#benchmarking-results)
- [Neural Network Framework](#neural-network-framework)
  - [Architecture](#architecture)
  - [Supported Components](#supported-components)
  - [Training Examples](#training-examples)
- [Building the Project](#building-the-project)
- [Running Tests and Benchmarks](#running-tests-and-benchmarks)
- [Dependencies](#dependencies)
- [Performance Analysis](#performance-analysis)
- [Future Improvements](#future-improvements)
- [References](#references)
- [Contributors](#contributors)

---

## Project Overview

This project consists of two main components:

1. **Optimized Matrix Multiplication Library**: Multiple implementations of matrix multiplication with varying optimization strategies, evaluated against OpenBLAS as a reference implementation
2. **Neural Network Framework**: A flexible, template-based neural network library supporting:
   - Dense layers with customizable activations
   - Multiple activation functions (ReLU, LeakyReLU, Sigmoid, Tanh, Softmax)
   - Loss functions (MSE, Cross-Entropy)
   - Optimizers (SGD with configurable learning rate)
   - Backpropagation

The library is designed for educational purposes and performance experimentation, demonstrating how low-level optimizations can dramatically improve computational efficiency.

---

## Project Structure

```
neuralnets-3-neuralnets/
├── include/
│   ├── matrix/
│   │   ├── matrix.hpp          # Matrix class with basic operations
│   │   ├── matmul.hpp          # Matrix multiplication implementations
│   │   └── iohelper.hpp        # CSV I/O utilities
│   ├── neuralnets/
│   │   ├── neuralnets.hpp      # Neural network and layer classes
│   │   ├── activation.hpp      # Activation functions
│   │   ├── loss.hpp            # Loss functions
│   │   ├── optimizer.hpp       # Optimization algorithms
│   │   └── metrics.hpp         # Performance metrics
│   └── utils/
│       └── bench.hpp           # Benchmarking utilities
├── tests/
│   ├── unit/                   # Unit tests
│   ├── benchmark/              # Benchmark tests
│   ├── test_classification_neuralnet.cpp
│   └── test_regression_neuralnet.cpp
├── output/
│   └── benchmark/
│       ├── matrix_mult_benchmark_logs.csv
│       └── plot_benchmark.ipynb    # Benchmark visualization
├── data/                       # Sample datasets
├── docs/                       # Documentation
└── CMakeLists.txt
```

The core library is implemented as header-only modules to simplify experimentation and inlining of performance-critical code.

---

## Matrix Multiplication Optimization

### Implemented Algorithms

The project implements several matrix multiplication strategies, each demonstrating specific optimization techniques:

#### 1. **Vanilla (Baseline)**

```cpp
C[i][j] = Σ A[i][k] * B[k][j]
```
- Standard triple-nested loop implementation ~ O(n³)
- No optimizations applied
- Serves as baseline for comparison

To call it, use:
```cpp
static Matrix<T> mm_vanilla(const Matrix<T> &A, const Matrix<T> &B)
```

#### 2. **Loop Unrolling (4x and 8x)**
```cpp
// Process 8 elements per iteration
for (k = 0; k + 7 < N; k += 8) {
    sum += A[i][k+0] * B[k+0][j] + A[i][k+1] * B[k+1][j] + ...
}
```
- Reduces loop overhead by processing multiple iterations together
- Improves instruction-level parallelism (ILP)
- Efficient register utilization
- Reduce branch predictions with fewer loop iterations

To call it, use:
```cpp
// 4-way loop unrolling
static Matrix<T> mm_unrolled4(const Matrix<T> &A, const Matrix<T> &B)

// 8-way loop unrolling
static Matrix<T> mm_unrolled8(const Matrix<T> &A, const Matrix<T> &B)

// Custom N-way loop unrolling
template<std::size_t N_UNROLL>
static Matrix<T> mm_unrolled(const Matrix<T> &A, const Matrix<T> &B)
```

#### 3. **Cache Tiling (Blocking)**
```cpp
// Process tile_size x tile_size blocks
for (ii = 0; ii < N; ii += tile_size)
    for (jj = 0; jj < N; jj += tile_size)
        for (kk = 0; kk < N; kk += tile_size)
            // Inner computation on tiles
```
- Improves cache locality by processing matrix in blocks
- Reduces cache misses significantly (tile size typically matched to L1/L2 cache size)

To call it, use:
```cpp
template<std::size_t TILE_SIZE>
static Matrix<T> mm_tiled(const Matrix<T> &A, const Matrix<T> &B)
```

#### 4. **SIMD AVX-512 Vectorization**
```cpp
// Process 8 doubles (512 bits) simultaneously
__m512d va = _mm512_loadu_pd(&A[i*K + k]);
__m512d vb = _mm512_loadu_pd(&B_t[j*K + k]);
vc = _mm512_fmadd_pd(va, vb, vc);
```
- Exploits AVX-512 SIMD instructions for data parallelism
- Processes 8 double-precision floats per instruction
- Requires B matrix transposition for efficient contiguous memory access

To call it, use:
```cpp
static Matrix<T> mm_avx512(const Matrix<T> &A, const Matrix<T> &B)
```

#### 5. **OpenMP Parallelization**
```cpp
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < M; ++i) {
    // Matrix multiplication per row
}
```
- Thread-level parallelism across CPU cores
- Static scheduling for load balancing and minimal overhead
- Ideally scales with number of cores (matrix multiplication is *embarrassingly parallel* across rows or blocks)

To call it, use:
```cpp
static Matrix<T> mm_openmp(const Matrix<T> &A, const Matrix<T> &B)
```

#### 6. **Optimized (Combined)**

- Combines multiple optimization techniques:
  - Multi-threading (OpenMP)
  - Vectorization (AVX-512)
  - Cache optimization (Tiling)
- Best performance among custom implementations

To call it, use:
```cpp
static Matrix<T> mm(const Matrix<T> &A, const Matrix<T> &B) // default method
```

#### 7. **OpenBLAS (Reference)**
- Industry-standard optimized BLAS library
- Highly tuned assembly code for specific architectures
- Serves as performance ceiling for comparison

### Benchmarking Results

The benchmark suite (`test_benchmark`) measures performance across different matrix sizes and generates detailed performance reports.

#### Running the Benchmark

```bash
cd build
./test_benchmark <matrix_size>

# Example: Benchmark 512x512 matrices
./test_benchmark 512
```

#### Benchmark Script

For comprehensive testing across multiple sizes:
```bash
bash tests/run_benchmark.sh
```

This runs benchmarks for matrix sizes: 8, 10, 16, 32, 64, 128, 200, 256, 512, 1024 and logs results to `output/benchmark/matrix_mult_benchmark_logs.csv`.

#### Results

Detailed numerical results are logged to CSV files and performance analysis is visualized by:
```
output/benchmark/plot_benchmark.ipynb
```

The Jupyter notebook provides:
- **Linear scale plots**: Absolute performance comparison
![Alt text](output/benchmark/images/benchmark_img.png)


- **Log scale plots**: Better visualization across different matrix sizes
![Alt text](output/benchmark/images/benchmark_img_log_scale.png)

**Key Observations**:
1. **Small matrices (< 32×32)**: Simple algorithms often faster due to overhead (thread creation cost)
2. **Medium matrices (64-256)**: Vectorization and parallelization show clear benefits
3. **Large matrices (> 512)**: Combined optimizations approach OpenBLAS performance
4. **OpenBLAS**: Consistently fastest for all the matrix sizes (highly tuned)

---

## Neural Network Framework

### Architecture

The framework follows a modular, template-based design:

```cpp
template<typename T>
class NeuralNets {
    std::vector<NeuralLayer<T>> layers;
    Optimizer<T>& optimizer;
    Loss<T>& loss_function;
    
    void train(const Matrix<T>& X, const Matrix<T>& y, int epochs);
    Matrix<T> predict(const Matrix<T>& X);
};
```

**Key Features**:
- Template-based for type flexibility (float, double)
- Layer-wise forward and backward propagation
- Automatic gradient computation
- Support common initialization strategy (e.g., He initialization)

### Supported Components

#### Activation Functions

| Function | Forward | Derivative | Use Case |
|----------|---------|------------|----------|
| **ReLU** | max(0, x) | 1 if x>0 else 0 | Hidden layers|
| **LeakyReLU** | x if x>0 else 0.01x | 1 if x>0 else 0.01 | Hidden layers (preventing dead neurons) |
| **Tanh** | tanh(x) | 1 - tanh²(x) | Hidden layers |
| **Sigmoid** | 1/(1+e^(-x)) | σ(x)(1-σ(x)) | Binary classification output |
| **Linear** | x | 1 | Regression output |

#### Loss Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| **MSE** | (1/n)Σ(ŷ - y)² | Regression |
| **Cross-Entropy** | -(1/n)Σy·log(ŷ) | Multi-class classification |
| **Binary Cross-Entropy** | -(1/n)Σ[y·log(ŷ) + (1-y)·log(1-ŷ)] | Binary classification |

#### Optimizers

| Optimizer | Update Rule | Parameters |
|-----------|-------------|------------|
| **SGD** | W ← W - η·∇W | Learning rate (η) |

### Training Examples

#### Binary Classification

```cpp
#include "matrix/matrix.hpp"
#include "neuralnets/neuralnets.hpp"

int main() {
    // Setup
    BinaryCrossEntropyLoss<double> loss;
    SGD<double> optimizer(0.1);  // Learning rate = 0.1
    NeuralNets<double> nn(optimizer, loss);
    
    // Architecture: 2 inputs → 3 hidden (LeakyReLU) → 1 output (Sigmoid)
    LeakyReLU<double> leaky_relu;
    Sigmoid<double> sigmoid;
    nn.add_layer(2, 3, leaky_relu);
    nn.add_layer(3, 1, sigmoid);
    
    // Training data (10 samples, 2 features)
    Matrix<double> X(10, 2, {0.1, 0.2, 0.3, 0.4, ...});
    Matrix<double> y(10, 1, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1});
    
    // Train
    nn.train(X, y, 100, true);  // 100 epochs, verbose
    
    // Predict
    Matrix<double> predictions = nn.predict(X);
    predictions.display();
    
    return 0;
}
```

#### Regression Example

```cpp
#include "matrix/matrix.hpp"
#include "neuralnets/neuralnets.hpp"

// Similar setup with different loss
MSELoss<double> mse_loss;
SGD<double> sgd(0.01);
NeuralNets<double> nn(sgd, mse_loss);

// Architecture for regression: no sigmoid at output
LeakyReLU<double> leaky_relu;
Linear<double> linear;
nn.add_layer(input_dim, 64, leaky_relu);
nn.add_layer(64, 32, leaky_relu);
nn.add_layer(32, output_dim, linear);

nn.train(X_train, y_train, 1000);
```


---

## Building the Project

### Prerequisites

- **C++ Compiler**: GCC 7+ or Clang 6+ (C++17 support required)
- **CMake**: Version 3.10 or higher
- **OpenMP**: For parallel matrix multiplication
- **OpenBLAS**: For benchmark comparison
- **AVX-512**: CPU support required for SIMD optimizations
- **Doxygen**: (Optional) For generating interactive documentation

### Build Instructions

```bash
# Clone repository
git clone https://github.com/AMSC-25-26/neuralnets-3-neuralnets.git
cd neuralnets-3-neuralnets

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build all targets
cmake --build .
```


The project automatically enables:
- `-std=c++17`: C++17 standard
- `-mavx512f`: AVX-512 SIMD instructions
- `-fopenmp`: OpenMP parallelization


### Generate Documentation

```bash
doxygen Doxyfile
```
Once documentation has been generated, it can be accessed from `docs/docs.html`

---

## Running Tests and Benchmarks

### Unit Tests

```bash
cd build

# Run all tests with CTest
ctest

# Or run individual test suites
./test_matrix_construct   # Matrix constructors tests
./test_matrix_ops         # Matrix operations tests
./test_matrix_mult        # Matrix multiplication tests
./test_neural_nets        # Neural network tests
./test_io_helper          # CSV I/O tests
```

### Benchmark Suite

#### Single Matrix Size
```bash
./test_benchmark 256
```
Output logged to `output/benchmark/matrix_mult_benchmark_logs.csv`

#### Comprehensive Benchmark
```bash
cd tests
bash run_benchmark.sh
```
Runs benchmarks for sizes: 8, 10, 16, 32, 64, 128, 200, 256, 512, 1024

#### Analyze Results
```bash
cd output/benchmark
jupyter notebook plot_benchmark.ipynb
```

### Neural Network Examples

#### Classification Task
```bash
./test_classification_neuralnet
```

#### Regression Task
```bash
./test_regression_neuralnet
```

---

## Dependencies

### Required

- **OpenBLAS**: High-performance BLAS library (as benchmark reference)
  ```bash
  sudo apt-get install libopenblas-dev
  ```

- **OpenMP**: Included with GCC/Clang
  ```bash
  sudo apt-get install libomp-dev
  ```

### Optional

- **GoogleTest**: For unit testing (included as submodule)

- **Jupyter**: For benchmark visualization
  ```bash
  pip install jupyter pandas matplotlib
  ```

---

## Key Insights (from performance analysis)

The benchmark results demonstrate several key principles for matrix multiplication:

1. **Algorithm selection matters**

   No single implementation is optimal across all problem sizes. Simple algorithms tend to perform well for small matrices due to low overhead, while more sophisticated optimization strategies become advantageous as problem size increases.

2. **Data locality is critical**

   Performance is strongly influenced by memory access patterns. Techniques such as cache blocking and data layout transformations improve spatial and temporal locality, reducing memory latency and improving effective throughput.

3. **Parallelization involves trade-offs**

   Multi-threading introduces non-negligible overhead and is therefore most effective for sufficiently large workloads. When the problem size justifies it, parallel execution can significantly improve performance by exploiting available CPU cores.

4. **Vectorization enables hardware-level parallelism**

   SIMD instructions allow multiple arithmetic operations to be performed simultaneously. Their effectiveness depends on regular data access patterns and appropriate memory organization, making them particularly well-suited for dense, compute-intensive kernels.




---

## Future Improvements

- [x] Improve code documentation with Doxygen
- [ ] Additional optimizers (e.g., Adam)
- [ ] GPU acceleration (CUDA)
- [ ] Refactor `activation.hpp` to include also non-diagonal Jacobian (e.g., SoftMax)
- [ ] ...
<!-- - [ ] Convolutional layers
- [ ] Mini-batch gradient descent
- [ ] Model serialization (save/load)
- [ ] Regularization (L1, L2, Dropout)
- [ ] Learning rate scheduling
- [ ] More loss functions
- [ ] Data augmentation utilities -->

---

## References

- **SIMD**  [[1](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)]
- **Loop unrolling** [[2](https://en.wikipedia.org/wiki/Loop_unrolling)]
- **Cache performance** [[3](https://en.wikipedia.org/wiki/Cache_performance_measurement_and_metric)]
- **Cache hierarchy** [[4](https://en.wikipedia.org/wiki/Cache_hierarchy)]



---


## Contributors

Filippo Dalla Chiara (filippo.dalla@mail.polimi.it | filippo.dallac@live.it)