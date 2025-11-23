// Code for testing matrix multiplication

#include "matrix.hpp"
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