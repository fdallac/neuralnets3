#include "bench.hpp"
#include "matrix.hpp"
#include "matmul.hpp"
#include <iostream>

extern "C" {
    #include <cblas.h>
}

int main() {
    const std::size_t N = 250; // Size of the square matrices

    // Initialize matrices A and B with some values
    Matrix<double> A(N, N);
    Matrix<double> B(N, N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            A(i, j) = static_cast<double>(i + j);
            B(i, j) = static_cast<double>(i - j);
        }
    }

    // // Benchmark OpenBLAS matrix multiplication
    // double time_openblas = Benchmark::measure([&]() {
    //     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //         N, N, N, 1.0, A.data(), N, B.data(), N, 0.0, C2.data(), N);
    // });

    // Create result buffers for OpenBLAS


    // Benchmark OpenBLAS matrix multiplication
    double time_openblas = Benchmark::measure([&]() {
        Matrix<double> C2(N, N);
        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            N, N, N,
            1.0,
            A.data(), N,
            B.data(), N,
            0.0,
            C2.data(), N
        );
    });

    // Benchmark vanilla matrix multiplication
    double time_vanilla = Benchmark::measure([&]() {
        Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::Vanilla);
    });

    // Benchmark loop unrolled by 4 matrix multiplication
    double time_unrolled4 = Benchmark::measure([&]() {
        Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::LoopUnrolled4);
    });

    // Benchmark loop unrolled by 4 matrix multiplication
    double time_unrolled8 = Benchmark::measure([&]() {
        Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::LoopUnrolled8);
    });

    // Benchmark loop unrolled by n (e.g., 4) matrix multiplication
    double time_unrolled_n4 = Benchmark::measure([&]() {
        Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::LoopUnrolledN, 4);
    });

    // Benchmark loop unrolled by n (e.g., 8) matrix multiplication
    double time_unrolled_n8 = Benchmark::measure([&]() {
        Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::LoopUnrolledN, 8);
    });



    // Output results
    std::cout << "Average time for OpenBLAS MM: " << time_openblas << " ms\n";
    std::cout << "Average time for Vanilla MM: " << time_vanilla << " ms\n";
    std::cout << "Average time for Loop Unrolled by 4 MM: " << time_unrolled4 << " ms\n";
    std::cout << "Average time for Loop Unrolled by 8 MM: " << time_unrolled8 << " ms\n";
    std::cout << "Average time for Loop Unrolled by n (e.g., 4) MM: " << time_unrolled_n4 << " ms\n";
    std::cout << "Average time for Loop Unrolled by n (e.g., 8) MM: " << time_unrolled_n8 << " ms\n";

    return 0;
}