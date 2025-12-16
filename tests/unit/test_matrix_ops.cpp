// Code for testing matrix sum and equality operators

#include "matrix/matrix.hpp"
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


TEST(MatrixOperations, MatrixTranspose) {
    std::vector<int> dataA = {
        1, 2, 3,
        4, 5, 6
    };
    Matrix<int> A(2, 3, dataA);
    
    Matrix<int> At = A.transpose();
    
    EXPECT_EQ(At.rows(), 3);
    EXPECT_EQ(At.cols(), 2);
    EXPECT_EQ(At(0, 0), 1);
    EXPECT_EQ(At(0, 1), 4);
    EXPECT_EQ(At(1, 0), 2);
    EXPECT_EQ(At(1, 1), 5);
    EXPECT_EQ(At(2, 0), 3);
    EXPECT_EQ(At(2, 1), 6);
}

TEST(MatrixOperations, MatrixAdditionIncompatibleDimensions) {
    Matrix<int> A(2, 2, {1, 2, 3, 4});
    Matrix<int> B(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    EXPECT_THROW(Matrix<int> C = A + B, std::invalid_argument);
}

TEST(MatrixOperations, MatrixInPlaceAddition) {
    Matrix<int> A(2, 2, {1, 2, 3, 4});
    Matrix<int> B(2, 2, {5, 6, 7, 8});
    A += B;
    EXPECT_EQ(A(0, 0), 6);
    EXPECT_EQ(A(0, 1), 8);
    EXPECT_EQ(A(1, 0), 10);
    EXPECT_EQ(A(1, 1), 12);
}

TEST(MatrixOperations, ElementwiseMultiply) {
    Matrix<int> A(2, 2, {1, 2, 3, 4});
    Matrix<int> B(2, 2, {2, 3, 4, 5});
    Matrix<int> C = A.elementwise_multiply(B);
    EXPECT_EQ(C(0, 0), 2);  // 1*2
    EXPECT_EQ(C(0, 1), 6);  // 2*3
    EXPECT_EQ(C(1, 0), 12); // 3*4
    EXPECT_EQ(C(1, 1), 20); // 4*5
}

TEST(MatrixOperations, ElementwiseMultiplyInplace) {
    Matrix<int> A(2, 2, {1, 2, 3, 4});
    Matrix<int> B(2, 2, {2, 3, 4, 5});
    A.elementwise_multiply_inplace(B);
    EXPECT_EQ(A(0, 0), 2);
    EXPECT_EQ(A(0, 1), 6);
    EXPECT_EQ(A(1, 0), 12);
    EXPECT_EQ(A(1, 1), 20);
}

TEST(MatrixOperations, HorizontalSum) {
    Matrix<int> A(2, 3, {1, 2, 3,
                         4, 5, 6});
    Matrix<int> sum = A.horizontal_sum();
    EXPECT_EQ(sum.rows(), 1);
    EXPECT_EQ(sum.cols(), 3);
    EXPECT_EQ(sum(0, 0), 5);  // 1+4
    EXPECT_EQ(sum(0, 1), 7);  // 2+5
    EXPECT_EQ(sum(0, 2), 9);  // 3+6
}

TEST(MatrixOperations, VerticalSum) {
    Matrix<int> A(2, 3, {1, 2, 3,
                         4, 5, 6});
    Matrix<int> sum = A.vertical_sum();
    EXPECT_EQ(sum.rows(), 2);
    EXPECT_EQ(sum.cols(), 1);
    EXPECT_EQ(sum(0, 0), 6);  // 1+2+3
    EXPECT_EQ(sum(1, 0), 15); // 4+5+6
}

TEST(MatrixOperations, BroadcastHorizontalSum) {
    Matrix<int> A(2, 3, {1, 2, 3,
                         4, 5, 6});
    Matrix<int> row_vec(1, 3, {10, 20, 30});
    Matrix<int> result = A.broadcast_horizontal_sum(row_vec);
    EXPECT_EQ(result(0, 0), 11);  // 1+10
    EXPECT_EQ(result(0, 1), 22);  // 2+20
    EXPECT_EQ(result(0, 2), 33);  // 3+30
    EXPECT_EQ(result(1, 0), 14);  // 4+10
    EXPECT_EQ(result(1, 1), 25);  // 5+20
    EXPECT_EQ(result(1, 2), 36);  // 6+30
}

TEST(MatrixOperations, BroadcastVerticalSum) {
    Matrix<int> A(2, 3, {1, 2, 3,
                         4, 5, 6});
    Matrix<int> col_vec(2, 1, {10, 20});
    Matrix<int> result = A.broadcast_vertical_sum(col_vec);
    EXPECT_EQ(result(0, 0), 11);  // 1+10
    EXPECT_EQ(result(0, 1), 12);  // 2+10
    EXPECT_EQ(result(0, 2), 13);  // 3+10
    EXPECT_EQ(result(1, 0), 24);  // 4+20
    EXPECT_EQ(result(1, 1), 25);  // 5+20
    EXPECT_EQ(result(1, 2), 26);  // 6+20
}

TEST(MatrixOperations, BroadcastInvalidDimensions) {
    Matrix<int> A(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix<int> wrong_row_vec(1, 2, {10, 20});
    Matrix<int> wrong_col_vec(3, 1, {10, 20, 30});
    EXPECT_THROW(A.broadcast_horizontal_sum(wrong_row_vec), std::invalid_argument);
    EXPECT_THROW(A.broadcast_vertical_sum(wrong_col_vec), std::invalid_argument);
}

TEST(MatrixOperations, DoubleTranspose) {
    Matrix<int> A(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix<int> At = A.transpose();
    Matrix<int> Att = At.transpose();
    EXPECT_TRUE(A == Att);
}

TEST(MatrixOperations, EqualityWithDifferentDimensions) {
    Matrix<int> A(2, 2, {1, 2, 3, 4});
    Matrix<int> B(2, 3, {1, 2, 3, 4, 5, 6});
    EXPECT_FALSE(A == B);
    EXPECT_TRUE(A != B);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}