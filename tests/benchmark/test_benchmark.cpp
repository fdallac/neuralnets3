#include "bench.hpp"
#include "matrix.hpp"
#include "matmul.hpp"
#include <iostream>

int main() {
    const std::size_t N = 512; // Size of the square matrices

    // Initialize matrices A and B with some values
    Matrix<double> A(N, N);
    Matrix<double> B(N, N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            A(i, j) = static_cast<double>(i + j);
            B(i, j) = static_cast<double>(i - j);
        }
    }

    // Benchmark vanilla matrix multiplication
    double time_vanilla = Benchmark::measure([&]() {
        Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::Vanilla);
    });

    // Benchmark loop unrolled by 4 matrix multiplication
    double time_unrolled4 = Benchmark::measure([&]() {
        Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::LoopUnrolled4);
    });

    std::cout << "Average time for Vanilla MM: " << time_vanilla << " ms\n";
    std::cout << "Average time for Loop Unrolled by 4 MM: " << time_unrolled4 << " ms\n";

    return 0;
}