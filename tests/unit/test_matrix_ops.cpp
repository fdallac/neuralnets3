// Code for testing matrix sum and equality operators

#include "matrix.hpp"
#include <gtest/gtest.h>

TEST(MatrixOperations, MatrixAddition) {
    std::vector<int> dataA = {
        1, 2,
        3, 4
    };
    std::vector<int> dataB = {
        5, 6,
        7, 8
    };
    Matrix<int> A(2, 2, dataA);
    Matrix<int> B(2, 2, dataB);
    
    Matrix<int> C = A + B;
    
    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_EQ(C(0, 0), 6); // 1 + 5
    EXPECT_EQ(C(0, 1), 8); // 2 + 6
    EXPECT_EQ(C(1, 0), 10); // 3 + 7
    EXPECT_EQ(C(1, 1), 12); // 4 + 8
}


TEST(MatrixOperations, MatrixEquality) {
    std::vector<int> dataA = {
        1, 2,
        3, 4
    };
    std::vector<int> dataB = {
        1, 2,
        3, 4
    };
    std::vector<int> dataC = {
        5, 6,
        7, 8
    };
    Matrix<int> A(2, 2, dataA);
    Matrix<int> B(2, 2, dataB);
    Matrix<int> C(2, 2, dataC);
    
    EXPECT_TRUE(A == B);
    EXPECT_FALSE(A == C);
    EXPECT_TRUE(A != C);
    EXPECT_FALSE(A != B);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}