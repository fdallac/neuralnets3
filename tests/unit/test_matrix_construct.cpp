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

TEST(MatrixConstructors, LargeMatrix) {
    Matrix<double> mat(100, 100);
    EXPECT_EQ(mat.rows(), 100);
    EXPECT_EQ(mat.cols(), 100);
    // Check all elements are zero
    for (std::size_t i = 0; i < 10; ++i) {
        for (std::size_t j = 0; j < 10; ++j) {
            EXPECT_EQ(mat(i, j), 0.0);
        }
    }
}

TEST(MatrixConstructors, SingleElementMatrix) {
    Matrix<int> mat(1, 1);
    EXPECT_EQ(mat.rows(), 1);
    EXPECT_EQ(mat.cols(), 1);
    mat(0, 0) = 42;
    EXPECT_EQ(mat(0, 0), 42);
}

TEST(MatrixConstructors, RowVector) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    Matrix<double> mat(1, 4, data);
    EXPECT_EQ(mat.rows(), 1);
    EXPECT_EQ(mat.cols(), 4);
    for (std::size_t j = 0; j < 4; ++j) {
        EXPECT_EQ(mat(0, j), data[j]);
    }
}

TEST(MatrixConstructors, ColumnVector) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    Matrix<double> mat(4, 1, data);
    EXPECT_EQ(mat.rows(), 4);
    EXPECT_EQ(mat.cols(), 1);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(mat(i, 0), data[i]);
    }
}

TEST(MatrixAccess, AtMethodBoundsChecking) {
    Matrix<int> mat(2, 2, {1, 2, 3, 4});
    EXPECT_NO_THROW(mat.at(0, 0));
    EXPECT_NO_THROW(mat.at(1, 1));
    EXPECT_THROW(mat.at(2, 0), std::out_of_range);
    EXPECT_THROW(mat.at(0, 2), std::out_of_range);
    EXPECT_THROW(mat.at(2, 2), std::out_of_range);
}

TEST(MatrixAccess, ConstAccess) {
    const Matrix<int> mat(2, 2, {1, 2, 3, 4});
    EXPECT_EQ(mat(0, 0), 1);
    EXPECT_EQ(mat(1, 1), 4);
    EXPECT_EQ(mat.at(0, 1), 2);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}