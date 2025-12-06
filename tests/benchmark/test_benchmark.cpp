#include "bench.hpp"
#include "matrix.hpp"
#include "matmul.hpp"
#include <iostream>

extern "C" {
    #include <cblas.h>
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>\n";
        return 1;
    }

    std::size_t N = std::strtoul(argv[1], nullptr, 10);
    if (N <= 0) {
        std::cerr << "Error: matrix_size must be a positive integer.\n";
        return 1;
    }

    std::cout << "Running benchmark for N = " << N << "...\n";


    // Initialize matrices A and B with some values
    Matrix<double> A(N, N);
    Matrix<double> B(N, N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            A(i, j) = static_cast<double>(i + j);
            B(i, j) = static_cast<double>(i - j);
        }
    }

 
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


    // Benchmark loop unrolled by 8 matrix multiplication
    double time_unrolled8 = Benchmark::measure([&]() {
        Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::LoopUnrolled8);
    });


    double time_unrolled8_1 = Benchmark::measure([&]() {
        Matrix<double> C = MatMul<double>::mm_unrolled<8>(A, B);
    });


    // Benchmark tiling matrix multiplication
    double time_tiled = Benchmark::measure([&]() {
        Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::Tiled);
    });


    // Benchmark OpenMP matrix multiplication
    double time_openmp = Benchmark::measure([&]() {
        Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::OpenMP);
    });


    // Benchmark SIMD (AVX-512) matrix multiplication
    double time_avx512 = Benchmark::measure([&]() {
        Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::SIMD_AVX512);
    });


    // Benchmark Optimized matrix multiplication
    double time_optimized = Benchmark::measure([&]() {
        Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::Optimized);
    });




    // Output results
    std::cout << "Average time for OpenBLAS MM: " << time_openblas << " ms\n";
    std::cout << "Average time for Vanilla MM: " << time_vanilla << " ms\n";
    // std::cout << "Average time for Loop Unrolled by 4 MM: " << time_unrolled4 << " ms\n";
    std::cout << "Average time for Loop Unrolled by 8 MM: " << time_unrolled8 << " ms\n";
    std::cout << "Average time for Loop Unrolled by n (n = 8) MM: " << time_unrolled8_1 << " ms\n";
    // std::cout << "Average time for Loop Unrolled by n (e.g., 8) MM: " << time_unrolled_n8 << " ms\n";
    std::cout << "Average time for Tiled MM: " << time_tiled << " ms\n";
    std::cout << "Average time for OpenMP MM: " << time_openmp << " ms\n";
    std::cout << "Average time for SIMD (AVX-512) MM: " << time_avx512 << " ms\n";
    std::cout << "Average time for Optimized MM: " << time_optimized << " ms\n";

    return 0;
}