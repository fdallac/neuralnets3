// Code for testing matrix constructors

#include "matrix/matrix.hpp"
#include <gtest/gtest.h>

TEST(MatrixConstructors, DefaultConstructor) {
    Matrix<int> mat;
    EXPECT_EQ(mat.rows(), 0);
    EXPECT_EQ(mat.cols(), 0);
}

TEST(MatrixConstructors, SizeConstructor) {
    Matrix<double> mat(3, 4);
    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 4);
    for (std::size_t i = 0; i < mat.rows(); ++i) {
        for (std::size_t j = 0; j < mat.cols(); ++j) {
            EXPECT_EQ(mat(i, j), 0.0);
        }
    }
}

TEST(MatrixConstructors, DataConstructor) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Matrix<float> mat(2, 3, data);
    EXPECT_EQ(mat.rows(), 2);
    EXPECT_EQ(mat.cols(), 3);
    for (std::size_t i = 0; i < mat.rows(); ++i) {
        for (std::size_t j = 0; j < mat.cols(); ++j) {
            EXPECT_EQ(mat(i, j), data[i * mat.cols() + j]);
        }
    }
}

TEST(MatrixConstructors, DataConstructorInvalidSize) {
    std::vector<int> data = {1, 2, 3};
    EXPECT_THROW(Matrix<int> mat(2, 2, data), std::invalid_argument);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}