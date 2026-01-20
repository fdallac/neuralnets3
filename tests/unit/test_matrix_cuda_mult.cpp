// Tests for CUDA matrix multiplication

#include "matrix/matrix.hpp"
#include "matrix/matmul.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>

#ifdef CUDA_AVAILABLE
#include "matrix/matmul_cuda.hpp"
#endif

// Helper function to compare floating point matrices with tolerance
template<typename T>
bool matricesEqual(const Matrix<T>& A, const Matrix<T>& B, T tolerance = static_cast<T>(1e-5)) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        return false;
    }
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            if (std::abs(A(i, j) - B(i, j)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

// Test fixture to check CUDA availability before each test
class CUDAMatrixMultiplication : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef CUDA_AVAILABLE
        if (!cuda_matmul::isCudaAvailable()) {
            GTEST_SKIP() << "CUDA device not available - skipping test";
        }
        std::cout << "Using CUDA device: " << cuda_matmul::getCudaDeviceInfo() << std::endl;
#else
        GTEST_SKIP() << "CUDA not compiled - skipping test";
#endif
    }
};


TEST_F(CUDAMatrixMultiplication, SmallFloatMatrices) {
    std::vector<float> dataA = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    std::vector<float> dataB = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };
    Matrix<float> A(2, 3, dataA);
    Matrix<float> B(3, 2, dataB);
    
    Matrix<float> C_cuda = MatMul<float>::mm(A, B, MatMulMethod::CUDA);
    Matrix<float> C_ref = MatMul<float>::mm(A, B, MatMulMethod::Vanilla);
    
    EXPECT_EQ(C_cuda.rows(), 2);
    EXPECT_EQ(C_cuda.cols(), 2);
    EXPECT_TRUE(matricesEqual(C_cuda, C_ref));
}


TEST_F(CUDAMatrixMultiplication, SmallDoubleMatrices) {
    std::vector<double> dataA = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };
    std::vector<double> dataB = {
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0
    };
    Matrix<double> A(2, 3, dataA);
    Matrix<double> B(3, 2, dataB);
    
    Matrix<double> C_cuda = MatMul<double>::mm(A, B, MatMulMethod::CUDA);
    Matrix<double> C_ref = MatMul<double>::mm(A, B, MatMulMethod::Vanilla);
    
    EXPECT_EQ(C_cuda.rows(), 2);
    EXPECT_EQ(C_cuda.cols(), 2);
    EXPECT_TRUE(matricesEqual(C_cuda, C_ref, 1e-10));
}


TEST_F(CUDAMatrixMultiplication, IncompatibleDimensions) {
    std::vector<float> dataA = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };
    std::vector<float> dataB = {
        5.0f, 6.0f, 7.0f
    };
    Matrix<float> A(2, 2, dataA);
    Matrix<float> B(1, 3, dataB);
    
    EXPECT_THROW(MatMul<float>::mm(A, B, MatMulMethod::CUDA), std::invalid_argument);
}


TEST_F(CUDAMatrixMultiplication, SquareMatrixFloat) {
    std::vector<float> dataA = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    std::vector<float> dataB = {
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f,
        29.0f, 30.0f, 31.0f, 32.0f
    };
    Matrix<float> A(4, 4, dataA);
    Matrix<float> B(4, 4, dataB);
    
    Matrix<float> C_cuda = MatMul<float>::mm(A, B, MatMulMethod::CUDA);
    Matrix<float> C_ref = MatMul<float>::mm(A, B, MatMulMethod::Vanilla);
    
    EXPECT_EQ(C_cuda.rows(), 4);
    EXPECT_EQ(C_cuda.cols(), 4);
    EXPECT_TRUE(matricesEqual(C_cuda, C_ref));
}


TEST_F(CUDAMatrixMultiplication, RectangularMatrixFloat) {
    std::vector<float> dataA(24);
    std::vector<float> dataB(24);
    for (std::size_t i = 0; i < 24; ++i) {
        dataA[i] = static_cast<float>(i + 1);
        dataB[i] = static_cast<float>(24 - i);
    }
    Matrix<float> A(6, 4, dataA);
    Matrix<float> B(4, 6, dataB);
    
    Matrix<float> C_cuda = MatMul<float>::mm(A, B, MatMulMethod::CUDA);
    Matrix<float> C_ref = MatMul<float>::mm(A, B, MatMulMethod::Vanilla);
    
    EXPECT_EQ(C_cuda.rows(), 6);
    EXPECT_EQ(C_cuda.cols(), 6);
    EXPECT_TRUE(matricesEqual(C_cuda, C_ref));
}


TEST_F(CUDAMatrixMultiplication, LargeMatrixFloat) {
    const std::size_t M = 64;
    const std::size_t K = 32;
    const std::size_t N = 48;
    
    std::vector<float> dataA(M * K);
    std::vector<float> dataB(K * N);
    
    for (std::size_t i = 0; i < M * K; ++i) {
        dataA[i] = static_cast<float>(i % 100) / 10.0f;
    }
    for (std::size_t i = 0; i < K * N; ++i) {
        dataB[i] = static_cast<float>(i % 100) / 10.0f;
    }
    
    Matrix<float> A(M, K, dataA);
    Matrix<float> B(K, N, dataB);
    
    Matrix<float> C_cuda = MatMul<float>::mm(A, B, MatMulMethod::CUDA);
    Matrix<float> C_ref = MatMul<float>::mm(A, B, MatMulMethod::Vanilla);
    
    EXPECT_EQ(C_cuda.rows(), M);
    EXPECT_EQ(C_cuda.cols(), N);
    EXPECT_TRUE(matricesEqual(C_cuda, C_ref, 1e-3f));
}


TEST_F(CUDAMatrixMultiplication, LargeMatrixDouble) {
    const std::size_t M = 32;
    const std::size_t K = 48;
    const std::size_t N = 40;
    
    std::vector<double> dataA(M * K);
    std::vector<double> dataB(K * N);
    
    for (std::size_t i = 0; i < M * K; ++i) {
        dataA[i] = static_cast<double>(i % 100) / 10.0;
    }
    for (std::size_t i = 0; i < K * N; ++i) {
        dataB[i] = static_cast<double>(i % 100) / 10.0;
    }
    
    Matrix<double> A(M, K, dataA);
    Matrix<double> B(K, N, dataB);
    
    Matrix<double> C_cuda = MatMul<double>::mm(A, B, MatMulMethod::CUDA);
    Matrix<double> C_ref = MatMul<double>::mm(A, B, MatMulMethod::Vanilla);
    
    EXPECT_EQ(C_cuda.rows(), M);
    EXPECT_EQ(C_cuda.cols(), N);
    EXPECT_TRUE(matricesEqual(C_cuda, C_ref, 1e-8));
}


TEST_F(CUDAMatrixMultiplication, IdentityMatrixFloat) {
    std::vector<float> dataA = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    std::vector<float> dataI = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    Matrix<float> A(3, 3, dataA);
    Matrix<float> I(3, 3, dataI);
    
    Matrix<float> C_cuda = MatMul<float>::mm(A, I, MatMulMethod::CUDA);
    
    EXPECT_EQ(C_cuda.rows(), 3);
    EXPECT_EQ(C_cuda.cols(), 3);
    EXPECT_TRUE(matricesEqual(C_cuda, A));
}


TEST_F(CUDAMatrixMultiplication, ZeroMatrixFloat) {
    std::vector<float> dataA = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };
    std::vector<float> dataZero = {
        0.0f, 0.0f,
        0.0f, 0.0f
    };
    Matrix<float> A(2, 2, dataA);
    Matrix<float> Z(2, 2, dataZero);
    
    Matrix<float> C_cuda = MatMul<float>::mm(A, Z, MatMulMethod::CUDA);
    
    EXPECT_EQ(C_cuda.rows(), 2);
    EXPECT_EQ(C_cuda.cols(), 2);
    EXPECT_TRUE(matricesEqual(C_cuda, Z));
}


TEST_F(CUDAMatrixMultiplication, NonSquareTileBoundary) {
    // Test matrix dimensions that don't align perfectly with 16x16 tiles
    const std::size_t M = 17;  // Not divisible by 16
    const std::size_t K = 19;
    const std::size_t N = 21;
    
    std::vector<float> dataA(M * K);
    std::vector<float> dataB(K * N);
    
    for (std::size_t i = 0; i < M * K; ++i) {
        dataA[i] = static_cast<float>(i % 10);
    }
    for (std::size_t i = 0; i < K * N; ++i) {
        dataB[i] = static_cast<float>(i % 10);
    }
    
    Matrix<float> A(M, K, dataA);
    Matrix<float> B(K, N, dataB);
    
    Matrix<float> C_cuda = MatMul<float>::mm(A, B, MatMulMethod::CUDA);
    Matrix<float> C_ref = MatMul<float>::mm(A, B, MatMulMethod::Vanilla);
    
    EXPECT_EQ(C_cuda.rows(), M);
    EXPECT_EQ(C_cuda.cols(), N);
    EXPECT_TRUE(matricesEqual(C_cuda, C_ref));
}


TEST_F(CUDAMatrixMultiplication, VeryLargeMatrixFloat) {
    // Test with larger matrices that would benefit from GPU acceleration
    const std::size_t M = 128;
    const std::size_t K = 256;
    const std::size_t N = 128;
    
    std::vector<float> dataA(M * K);
    std::vector<float> dataB(K * N);
    
    for (std::size_t i = 0; i < M * K; ++i) {
        dataA[i] = static_cast<float>((i * 7) % 100) / 50.0f - 1.0f;
    }
    for (std::size_t i = 0; i < K * N; ++i) {
        dataB[i] = static_cast<float>((i * 13) % 100) / 50.0f - 1.0f;
    }
    
    Matrix<float> A(M, K, dataA);
    Matrix<float> B(K, N, dataB);
    
    Matrix<float> C_cuda = MatMul<float>::mm(A, B, MatMulMethod::CUDA);
    Matrix<float> C_ref = MatMul<float>::mm(A, B, MatMulMethod::SIMD_OpenMP_Tile);
    
    EXPECT_EQ(C_cuda.rows(), M);
    EXPECT_EQ(C_cuda.cols(), N);
    EXPECT_TRUE(matricesEqual(C_cuda, C_ref, 1e-2f));  // Slightly larger tolerance for large matrices
}


// ============================================================================
// CUDA + OpenMP Hybrid Tests
// ============================================================================

TEST_F(CUDAMatrixMultiplication, HybridSmallFloatMatrices) {
    std::vector<float> dataA = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    std::vector<float> dataB = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };
    Matrix<float> A(2, 3, dataA);
    Matrix<float> B(3, 2, dataB);
    
    Matrix<float> C_hybrid = MatMul<float>::mm(A, B, MatMulMethod::CUDA_OpenMP);
    Matrix<float> C_ref = MatMul<float>::mm(A, B, MatMulMethod::Vanilla);
    
    EXPECT_EQ(C_hybrid.rows(), 2);
    EXPECT_EQ(C_hybrid.cols(), 2);
    EXPECT_TRUE(matricesEqual(C_hybrid, C_ref));
}


TEST_F(CUDAMatrixMultiplication, HybridSmallDoubleMatrices) {
    std::vector<double> dataA = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };
    std::vector<double> dataB = {
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0
    };
    Matrix<double> A(2, 3, dataA);
    Matrix<double> B(3, 2, dataB);
    
    Matrix<double> C_hybrid = MatMul<double>::mm(A, B, MatMulMethod::CUDA_OpenMP);
    Matrix<double> C_ref = MatMul<double>::mm(A, B, MatMulMethod::Vanilla);
    
    EXPECT_EQ(C_hybrid.rows(), 2);
    EXPECT_EQ(C_hybrid.cols(), 2);
    EXPECT_TRUE(matricesEqual(C_hybrid, C_ref, 1e-10));
}


TEST_F(CUDAMatrixMultiplication, HybridMediumMatrixFloat) {
    const std::size_t M = 64;
    const std::size_t K = 32;
    const std::size_t N = 48;
    
    std::vector<float> dataA(M * K);
    std::vector<float> dataB(K * N);
    
    for (std::size_t i = 0; i < M * K; ++i) {
        dataA[i] = static_cast<float>(i % 100) / 10.0f;
    }
    for (std::size_t i = 0; i < K * N; ++i) {
        dataB[i] = static_cast<float>(i % 100) / 10.0f;
    }
    
    Matrix<float> A(M, K, dataA);
    Matrix<float> B(K, N, dataB);
    
    Matrix<float> C_hybrid = MatMul<float>::mm(A, B, MatMulMethod::CUDA_OpenMP);
    Matrix<float> C_ref = MatMul<float>::mm(A, B, MatMulMethod::Vanilla);
    
    EXPECT_EQ(C_hybrid.rows(), M);
    EXPECT_EQ(C_hybrid.cols(), N);
    EXPECT_TRUE(matricesEqual(C_hybrid, C_ref, 1e-3f));
}


TEST_F(CUDAMatrixMultiplication, HybridMediumMatrixDouble) {
    const std::size_t M = 32;
    const std::size_t K = 48;
    const std::size_t N = 40;
    
    std::vector<double> dataA(M * K);
    std::vector<double> dataB(K * N);
    
    for (std::size_t i = 0; i < M * K; ++i) {
        dataA[i] = static_cast<double>(i % 100) / 10.0;
    }
    for (std::size_t i = 0; i < K * N; ++i) {
        dataB[i] = static_cast<double>(i % 100) / 10.0;
    }
    
    Matrix<double> A(M, K, dataA);
    Matrix<double> B(K, N, dataB);
    
    Matrix<double> C_hybrid = MatMul<double>::mm(A, B, MatMulMethod::CUDA_OpenMP);
    Matrix<double> C_ref = MatMul<double>::mm(A, B, MatMulMethod::Vanilla);
    
    EXPECT_EQ(C_hybrid.rows(), M);
    EXPECT_EQ(C_hybrid.cols(), N);
    EXPECT_TRUE(matricesEqual(C_hybrid, C_ref, 1e-8));
}


TEST_F(CUDAMatrixMultiplication, HybridLargeMatrixFloat) {
    // Test with larger matrices that benefit from stream-based parallelism
    const std::size_t M = 512;
    const std::size_t K = 256;
    const std::size_t N = 512;
    
    std::vector<float> dataA(M * K);
    std::vector<float> dataB(K * N);
    
    for (std::size_t i = 0; i < M * K; ++i) {
        dataA[i] = static_cast<float>((i * 7) % 100) / 50.0f - 1.0f;
    }
    for (std::size_t i = 0; i < K * N; ++i) {
        dataB[i] = static_cast<float>((i * 13) % 100) / 50.0f - 1.0f;
    }
    
    Matrix<float> A(M, K, dataA);
    Matrix<float> B(K, N, dataB);
    
    Matrix<float> C_hybrid = MatMul<float>::mm(A, B, MatMulMethod::CUDA_OpenMP);
    Matrix<float> C_ref = MatMul<float>::mm(A, B, MatMulMethod::SIMD_OpenMP_Tile);
    
    EXPECT_EQ(C_hybrid.rows(), M);
    EXPECT_EQ(C_hybrid.cols(), N);
    EXPECT_TRUE(matricesEqual(C_hybrid, C_ref, 1e-2f));
}


TEST_F(CUDAMatrixMultiplication, HybridLargeMatrixDouble) {
    // Test with larger matrices using stream-based parallelism
    const std::size_t M = 256;
    const std::size_t K = 128;
    const std::size_t N = 256;
    
    std::vector<double> dataA(M * K);
    std::vector<double> dataB(K * N);
    
    for (std::size_t i = 0; i < M * K; ++i) {
        dataA[i] = static_cast<double>((i * 11) % 100) / 50.0 - 1.0;
    }
    for (std::size_t i = 0; i < K * N; ++i) {
        dataB[i] = static_cast<double>((i * 17) % 100) / 50.0 - 1.0;
    }
    
    Matrix<double> A(M, K, dataA);
    Matrix<double> B(K, N, dataB);
    
    Matrix<double> C_hybrid = MatMul<double>::mm(A, B, MatMulMethod::CUDA_OpenMP);
    Matrix<double> C_ref = MatMul<double>::mm(A, B, MatMulMethod::SIMD_OpenMP_Tile);
    
    EXPECT_EQ(C_hybrid.rows(), M);
    EXPECT_EQ(C_hybrid.cols(), N);
    EXPECT_TRUE(matricesEqual(C_hybrid, C_ref, 1e-7));
}


TEST_F(CUDAMatrixMultiplication, HybridNonSquareTileBoundary) {
    // Test matrix dimensions that don't align with block boundaries
    const std::size_t M = 517;  // Larger than BLOCK_SIZE (512)
    const std::size_t K = 133;  // Not divisible by tile size
    const std::size_t N = 267;
    
    std::vector<float> dataA(M * K);
    std::vector<float> dataB(K * N);
    
    for (std::size_t i = 0; i < M * K; ++i) {
        dataA[i] = static_cast<float>(i % 10);
    }
    for (std::size_t i = 0; i < K * N; ++i) {
        dataB[i] = static_cast<float>(i % 10);
    }
    
    Matrix<float> A(M, K, dataA);
    Matrix<float> B(K, N, dataB);
    
    Matrix<float> C_hybrid = MatMul<float>::mm(A, B, MatMulMethod::CUDA_OpenMP);
    Matrix<float> C_ref = MatMul<float>::mm(A, B, MatMulMethod::SIMD_OpenMP_Tile);
    
    EXPECT_EQ(C_hybrid.rows(), M);
    EXPECT_EQ(C_hybrid.cols(), N);
    EXPECT_TRUE(matricesEqual(C_hybrid, C_ref, 1e-2f));
}


TEST_F(CUDAMatrixMultiplication, HybridVsPlainCUDA) {
    // Compare hybrid approach with plain CUDA on moderately large matrices
    const std::size_t M = 256;
    const std::size_t K = 256;
    const std::size_t N = 256;
    
    std::vector<float> dataA(M * K);
    std::vector<float> dataB(K * N);
    
    for (std::size_t i = 0; i < M * K; ++i) {
        dataA[i] = static_cast<float>(i % 100) / 10.0f;
    }
    for (std::size_t i = 0; i < K * N; ++i) {
        dataB[i] = static_cast<float>(i % 100) / 10.0f;
    }
    
    Matrix<float> A(M, K, dataA);
    Matrix<float> B(K, N, dataB);
    
    Matrix<float> C_hybrid = MatMul<float>::mm(A, B, MatMulMethod::CUDA_OpenMP);
    Matrix<float> C_cuda = MatMul<float>::mm(A, B, MatMulMethod::CUDA);
    
    EXPECT_EQ(C_hybrid.rows(), M);
    EXPECT_EQ(C_hybrid.cols(), N);
    // Both should produce identical results
    EXPECT_TRUE(matricesEqual(C_hybrid, C_cuda, 1e-4f));
}


TEST_F(CUDAMatrixMultiplication, HybridRectangularMatrix) {
    // Test rectangular matrices with hybrid approach
    const std::size_t M = 384;
    const std::size_t K = 192;
    const std::size_t N = 256;
    
    std::vector<double> dataA(M * K);
    std::vector<double> dataB(K * N);
    
    for (std::size_t i = 0; i < M * K; ++i) {
        dataA[i] = static_cast<double>(i % 50) / 25.0 - 1.0;
    }
    for (std::size_t i = 0; i < K * N; ++i) {
        dataB[i] = static_cast<double>(i % 50) / 25.0 - 1.0;
    }
    
    Matrix<double> A(M, K, dataA);
    Matrix<double> B(K, N, dataB);
    
    Matrix<double> C_hybrid = MatMul<double>::mm(A, B, MatMulMethod::CUDA_OpenMP);
    Matrix<double> C_ref = MatMul<double>::mm(A, B, MatMulMethod::SIMD_OpenMP_Tile);
    
    EXPECT_EQ(C_hybrid.rows(), M);
    EXPECT_EQ(C_hybrid.cols(), N);
    EXPECT_TRUE(matricesEqual(C_hybrid, C_ref, 1e-7));
}


TEST_F(CUDAMatrixMultiplication, HybridIncompatibleDimensions) {
    std::vector<float> dataA = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };
    std::vector<float> dataB = {
        5.0f, 6.0f, 7.0f
    };
    Matrix<float> A(2, 2, dataA);
    Matrix<float> B(1, 3, dataB);
    
    EXPECT_THROW(MatMul<float>::mm(A, B, MatMulMethod::CUDA_OpenMP), std::invalid_argument);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
