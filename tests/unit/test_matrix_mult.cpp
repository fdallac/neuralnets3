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
    Matrix<int> D = MatMul<int>::mm(A, B); // Using default multiplication for verification
    
    EXPECT_EQ(C.rows(), 4);
    EXPECT_EQ(C.cols(), 4);
    EXPECT_TRUE(C == D);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}