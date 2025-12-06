// Code for testing matrix multiplication

#include "matrix.hpp"
#include "matmul.hpp"
#include <gtest/gtest.h>

TEST(MatrixMultiplication, SmallMatricesMultiplication) {
    std::vector<int> dataA = {
        1, 2, 3,
        4, 5, 6
    };
    std::vector<int> dataB = {
        7, 8,
        9, 10,
        11, 12
    };
    Matrix<int> A(2, 3, dataA);
    Matrix<int> B(3, 2, dataB);
    
    Matrix<int> C = A * B;
    
    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_EQ(C(0, 0), 58); // 1*7 + 2*9 + 3*11
    EXPECT_EQ(C(0, 1), 64); // 1*8 + 2*10 + 3*12
    EXPECT_EQ(C(1, 0), 139); // 4*7 + 5*9 + 6*11
    EXPECT_EQ(C(1, 1), 154); // 4*8 + 5*10 + 6*12
}


TEST(MatrixMultiplication, IncompatibleMatricesMultiplication) {
    std::vector<int> dataA = {
        1, 2,
        3, 4
    };
    std::vector<int> dataB = {
        5, 6, 7
    };
    Matrix<int> A(2, 2, dataA);
    Matrix<int> B(1, 3, dataB);
    
    EXPECT_THROW(Matrix<int> C = A * B, std::invalid_argument);
}


TEST(MatrixMultiplication, MultiplicationWithZeroMatrix) {
    std::vector<int> dataA = {
        1, 2,
        3, 4
    };
    std::vector<int> dataB = {
        0, 0,
        0, 0
    };
    Matrix<int> A(2, 2, dataA);
    Matrix<int> B(2, 2, dataB);
    
    Matrix<int> C = A * B;
    
    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    for (std::size_t i = 0; i < C.rows(); ++i) {
        for (std::size_t j = 0; j < C.cols(); ++j) {
            EXPECT_EQ(C(i, j), 0);
        }
    }
}


TEST(MatrixMultiplication, IdentityMatrixMultiplication) {
    std::vector<int> dataA = {
        1, 2,
        3, 4
    };
    std::vector<int> dataB = {
        1, 0,
        0, 1
    };
    Matrix<int> A(2, 2, dataA);
    Matrix<int> B(2, 2, dataB);
    
    Matrix<int> C = A * B;
    
    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_TRUE(C == A);
}


TEST(MatrixMultiplication, UnrolledMultiplication) {
    std::vector<int> dataA = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    std::vector<int> dataB = {
        17, 18, 19, 20,
        21, 22, 23, 24,
        25, 26, 27, 28,
        29, 30, 31, 32
    };
    Matrix<int> A(4, 4, dataA);
    Matrix<int> B(4, 4, dataB);
    
    Matrix<int> C = MatMul<int>::mm_unrolled4(A, B);
    Matrix<int> D = MatMul<int>::mm_unrolled<4>(A, B); // Using default multiplication for verification
    Matrix<int> E = MatMul<int>::mm(A, B); // Using default multiplication for verification
    
    EXPECT_EQ(C.rows(), 4);
    EXPECT_EQ(C.cols(), 4);
    EXPECT_TRUE(C == D);
    EXPECT_TRUE(C == E);
}


TEST(MatrixMultiplication, Unrolled8Multiplication) {
    std::vector<int> dataA = {
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64
    };
    std::vector<int> dataB = {
        65, 66, 67, 68, 69, 70, 71, 72,
        73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88,
        89, 90, 91, 92, 93, 94, 95, 96,
        97, 98, 99,100,101,102,103,104,
       105,106,107,108,109,110,111,112,
       113,114,115,116,117,118,119,120,
       121,122,123,124,125,126,127,128
    };
    Matrix<int> A(8, 8, dataA);
    Matrix<int> B(8, 8, dataB);
    
    Matrix<int> C = MatMul<int>::mm_unrolled8(A, B);
    Matrix<int> D = MatMul<int>::mm_unrolled<8>(A, B); // Using default multiplication for verification
    Matrix<int> E = MatMul<int>::mm(A, B); // Using default multiplication for verification
    
    EXPECT_EQ(C.rows(), 8);
    EXPECT_EQ(C.cols(), 8);
    EXPECT_TRUE(C == D);
    EXPECT_TRUE(C == E);
}


TEST(MatrixMultiplication, TiledMultiplication) {
    std::vector<int> dataA = {
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64
    };
    std::vector<int> dataB = {
        65, 66, 67, 68, 69, 70, 71, 72,
        73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88,
        89, 90, 91, 92, 93, 94, 95, 96,
        97, 98, 99,100,101,102,103,104,
       105,106,107,108,109,110,111,112,
       113,114,115,116,117,118,119,120,
       121,122,123,124,125,126,127,128
    };
    Matrix<int> A(8, 8, dataA);
    Matrix<int> B(8, 8, dataB);
    
    Matrix<int> C = MatMul<int>::mm_tiled<16>(A, B);
    Matrix<int> D = MatMul<int>::mm(A, B); // Using default multiplication for verification
    
    EXPECT_EQ(C.rows(), 8);
    EXPECT_EQ(C.cols(), 8);
    EXPECT_TRUE(C == D);
}


TEST(MatrixMultiplication, OpenMP_Multiplication) {
    std::vector<int> dataA = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    std::vector<int> dataB = {
        17, 18, 19, 20,
        21, 22, 23, 24,
        25, 26, 27, 28,
        29, 30, 31, 32
    };
    Matrix<int> A(4, 4, dataA);
    Matrix<int> B(4, 4, dataB);
    
    Matrix<int> C = MatMul<int>::mm_openmp(A, B);
    Matrix<int> D = MatMul<int>::mm(A, B); // Using default multiplication for verification
    
    EXPECT_EQ(C.rows(), 4);
    EXPECT_EQ(C.cols(), 4);
    EXPECT_TRUE(C == D);
}



TEST(MatrixMultiplication, SIMD_Multiplication) {
    std::vector<float> dataA(32);
    std::vector<float> dataB(32);
    for (std::size_t i = 0; i < 32; ++i) {
        dataA[i] = static_cast<float>(i + 1);
        dataB[i] = static_cast<float>(i + 1);
    }
    Matrix<float> A(4, 8, dataA);
    Matrix<float> B(8, 4, dataB);
    
    Matrix<float> C = MatMul<float>::mm_avx512(A, B);
    Matrix<float> D = MatMul<float>::mm(A, B); // Using default multiplication for verification
    
    EXPECT_EQ(C.rows(), 4);
    EXPECT_EQ(C.cols(), 4);
    EXPECT_TRUE(C == D);
}


TEST(MatrixMultiplication, Optimized_Multiplication) {
    std::vector<float> dataA(32);
    std::vector<float> dataB(32);
    for (std::size_t i = 0; i < 32; ++i) {
        dataA[i] = static_cast<float>(i + 1);
        dataB[i] = static_cast<float>(i + 1);
    }
    Matrix<float> A(4, 8, dataA);
    Matrix<float> B(8, 4, dataB);
    
    Matrix<float> C = MatMul<float>::mm_optimized(A, B);
    Matrix<float> D = MatMul<float>::mm(A, B); // Using default multiplication for verification
    
    EXPECT_EQ(C.rows(), 4);
    EXPECT_EQ(C.cols(), 4);
    EXPECT_TRUE(C == D);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
