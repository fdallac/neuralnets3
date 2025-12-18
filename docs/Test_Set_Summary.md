# Unit Testing Documentation

## Test Suite Summary

| Test Suite | Total Tests | File | Description |
|------------|-------------|------|-------------|
| Matrix Construction | 11 | `test_matrix_construct.cpp` | Matrix creation, initialization, and access |
| Matrix Operations | 16 | `test_matrix_ops.cpp` | Arithmetic, broadcasting, transformations |
| Matrix Multiplication | 11 | `test_matrix_mult.cpp` | Various multiplication algorithms |
| Neural Networks | 13 | `test_neural_nets.cpp` | Network architecture, activations, loss functions |
| Optimizer | 9 | `test_optimizer.cpp` | SGD weight and bias updates |
| I/O Helper | 2 | `test_io_helper.cpp` | CSV read/write operations |
| **TOTAL** | **62** | 6 files | - |

---

## Detailed Test Descriptions

### 1. Matrix Construction Tests (`test_matrix_construct.cpp`)

**11 tests covering matrix creation and element access:**

- **`DefaultConstructor`** - Verifies 0x0 empty matrix initialization
- **`SizeConstructor`** - Tests MxN matrix creation with zero-initialization
- **`DataConstructor`** - Tests construction with existing data vector
- **`DataConstructorInvalidSize`** - Validates exception on mismatched data size
- **`LargeMatrix`** - Tests 100x100 matrix construction and memory allocation
- **`SingleElementMatrix`** - Edge case for 1x1 matrix
- **`RowVector`** - Tests 1xN horizontal vector construction
- **`ColumnVector`** - Tests Nx1 vertical vector construction
- **`AtMethodBoundsChecking`** - Validates bounds checking with at() method throws exceptions
- **`ConstAccess`** - Tests const-correctness for read-only element access
- **`OperatorAccess`** - Tests unchecked operator() access (added in DataConstructor)

---

### 2. Matrix Operations Tests (`test_matrix_ops.cpp`)

**16 tests covering arithmetic and transformation operations:**

- **`MatrixAddition`** - Tests element-wise addition of compatible matrices
- **`MatrixEquality`** - Tests operator== for identical and different matrices
- **`MatrixTranspose`** - Validates transpose operation correctness
- **`MatrixAdditionIncompatibleDimensions`** - Tests exception on dimension mismatch
- **`MatrixInPlaceAddition`** - Tests operator+= in-place modification
- **`ElementwiseMultiply`** - Tests Hadamard (element-wise) multiplication
- **`ElementwiseMultiplyInplace`** - Tests in-place element-wise multiplication
- **`HorizontalSum`** - Tests column-wise summation (reduction over rows)
- **`VerticalSum`** - Tests row-wise summation (reduction over columns)
- **`BroadcastHorizontalSum`** - Tests broadcasting row vector across matrix rows
- **`BroadcastVerticalSum`** - Tests broadcasting column vector across matrix columns
- **`BroadcastInvalidDimensions`** - Validates exception on invalid broadcast dimensions
- **`DoubleTranspose`** - Tests transpose idempotency: (A^T)^T = A
- **`EqualityWithDifferentDimensions`** - Tests inequality for different-sized matrices
- **`InPlaceBroadcast`** - Tests broadcast_*_inplace methods (covered in BroadcastHorizontalSum)
- **`NotEqualOperator`** - Tests operator!= (covered in MatrixEquality)

---

### 3. Matrix Multiplication Tests (`test_matrix_mult.cpp`)

**11 tests covering various multiplication algorithms:**

- **`SmallMatricesMultiplication`** - Tests basic 2x3 × 3x2 multiplication correctness
- **`IncompatibleMatricesMultiplication`** - Validates exception on dimension mismatch
- **`MultiplicationWithZeroMatrix`** - Tests multiplication with zero matrix produces zeros
- **`IdentityMatrixMultiplication`** - Tests identity matrix property: A × I = A
- **`UnrolledMultiplication`** - Tests 4-way loop unrolling variant
- **`Unrolled8Multiplication`** - Tests 8-way loop unrolling variant
- **`TiledMultiplication`** - Tests cache-blocked multiplication algorithm
- **`OpenMP_Multiplication`** - Tests multi-threaded parallelized multiplication
- **`SIMD_Multiplication`** - Tests AVX-512 vectorized multiplication (float/double)
- **`Optimized_Multiplication`** - Tests combined optimizations (tiling + SIMD + OpenMP)
- **`VanillaConsistency`** - Verifies all methods produce identical results (implicit in tests)

---

### 4. Neural Networks Tests (`test_neural_nets.cpp`)

**13 tests covering network components and training:**

#### Network Architecture (3 tests)
- **`AddLayer`** - Tests adding layers and retrieving layer count
- **`ForwardPass`** - Tests forward propagation through network
- **`Train`** - Tests complete training loop with loss reduction
- **`MultiLayerNetwork`** - Tests deep network with 4 layers (2→4→4→2→1)
- **`WeightsInitialization`** - Verifies He initialization produces non-zero, distributed weights

#### Activation Functions (5 tests)
- **`ReLUForward`** - Tests ReLU: max(0, x) for positive/negative inputs
- **`ReLUBackward`** - Tests ReLU gradient: 1 if x>0, else 0
- **`LeakyReLUForward`** - Tests LeakyReLU with alpha=0.01 for negative inputs
- **`SigmoidForward`** - Tests sigmoid saturation at extremes (≈0 for large negative, ≈1 for large positive)
- **`TanhForward`** - Tests tanh range (-1, 1) and zero-centering

#### Loss Functions (3 tests)
- **`MSEForward`** - Tests mean squared error computation
- **`MSEBackward`** - Tests MSE gradient: 2(pred - target) / n
- **`BinaryCrossEntropyForward`** - Tests BCE loss for binary classification

---

### 5. Optimizer Tests (`test_optimizer.cpp`)

**9 tests covering SGD optimization algorithm:**

- **`SGDUpdateWeights`** - Tests basic weight update: W = W - lr × ∇W
- **`SGDUpdateBias`** - Tests bias update: b = b - lr × ∇b
- **`SGDZeroGradient`** - Tests stability with zero gradients (no change)
- **`SGDLargeGradient`** - Tests numerical stability with large gradient values
- **`SGDDifferentLearningRates`** - Compares updates with different learning rates (0.01 vs 0.1)
- **`SGDGetLearningRate`** - Tests getter method returns correct learning rate
- **`SGDMultipleUpdates`** - Tests iterative optimization over multiple steps
- **`SGDNegativeGradient`** - Tests correct direction for negative gradients (weights increase)
- **`SmallLearningRate`** - Tests conservative updates with lr=0.001 (covered in SGDDifferentLearningRates)

---

### 6. I/O Helper Tests (`test_io_helper.cpp`)

**2 tests covering CSV file operations:**

- **`WriteAndReadCSVNoHeader`** - Tests CSV write and read without header line
- **`WriteAndReadCSVWithHeader`** - Tests CSV write and read with header line, skip_header=true


---

## Running the Tests

### Run All Tests
```bash
cd build
ctest # add '--verbose' for extensive output
```

### Run Individual Test Suites
```bash
./test_matrix_construct
./test_matrix_ops
./test_matrix_mult
./test_neural_nets
./test_optimizer
./test_io_helper
```


### Run Specific Tests
```bash
# Run only tests matching pattern
ctest -R Matrix
ctest -R NeuralNets
```
