#include "matrix.hpp"
#include "utils/iohelper.hpp"
#include <gtest/gtest.h>


TEST(IOHelper, WriteAndReadCSVNoHeader) {
    Matrix<double> M(2, 3, {1.1, 2.2, 3.3,
                            4.4, 5.5, 6.6});
    std::string filename = "../tests/sample_data/out/test_matrix.csv";

    // Write to CSV
    IOHelper<double>::write_csv(M, filename, ',', false);

    // Read back from CSV
    Matrix<double> M_read = IOHelper<double>::read_csv(filename, ',', false);

    // Check dimensions
    EXPECT_EQ(M_read.rows(), M.rows());
    EXPECT_EQ(M_read.cols(), M.cols());

    // Check values
    for (std::size_t i = 0; i < M.rows(); ++i) {
        for (std::size_t j = 0; j < M.cols(); ++j) {
            EXPECT_DOUBLE_EQ(M_read(i, j), M(i, j));
        }
    }
}


TEST(IOHelper, WriteAndReadCSVWithHeader) {
    Matrix<double> M(2, 3, {7.7, 8.8, 9.9,
                            10.1, 11.2, 12.3});
    std::string filename = "../tests/sample_data/out/test_matrix_with_header.csv";
    std::string header = "Col1, Col2, Col3";

    // Write to CSV with header
    IOHelper<double>::write_csv(M, filename, ',', true, header);

    // Read back from CSV, skipping header
    Matrix<double> M_read = IOHelper<double>::read_csv(filename, ',', true);

    // Check dimensions
    EXPECT_EQ(M_read.rows(), M.rows());
    EXPECT_EQ(M_read.cols(), M.cols());

    // Check values
    for (std::size_t i = 0; i < M.rows(); ++i) {
        for (std::size_t j = 0; j < M.cols(); ++j) {
            EXPECT_DOUBLE_EQ(M_read(i, j), M(i, j));
        }
    }
}