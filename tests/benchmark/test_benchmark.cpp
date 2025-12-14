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

    std::cout << "Running benchmark for N = " << N << " ...\n";

    // Report filename
    const std::string report_filename = "../output/benchmark/matrix_mult_benchmark_logs.csv";

    // Initialize matrices A and B with some values
    Matrix<double> A(N, N);
    Matrix<double> B(N, N);
    A.fill_uniform_noise(-10.0, 10.0);
    B.fill_uniform_noise(-10.0, 10.0);

 
    // Benchmark OpenBLAS matrix multiplication
    double time_openblas = Benchmark::measure_and_report_matmul(
        std::to_string(N),
        "OpenBLAS",
        [&]() {
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
        },
        report_filename
    );

    // Benchmark vanilla matrix multiplication
    double time_vanilla = Benchmark::measure_and_report_matmul(
        std::to_string(N),
        "Vanilla",
        [&]() {Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::Vanilla);},
        report_filename
    );


    // Benchmark loop unrolled by 8 matrix multiplication
    double time_unrolled8 = Benchmark::measure_and_report_matmul(
        std::to_string(N),
        "LoopUnrolled_8",
        [&]() {Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::LoopUnrolled8);},
        report_filename
    );


    double time_unrolled8_1 = Benchmark::measure_and_report_matmul(
        std::to_string(N),
        "LoopUnrolled_custom_8",
        [&]() {Matrix<double> C = MatMul<double>::mm_unrolled<8>(A, B);},
        report_filename
    );


    // Benchmark tiling matrix multiplication
    double time_tiled = Benchmark::measure_and_report_matmul(
        std::to_string(N),
        "Tiled",
        [&]() {Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::Tiled);},
        report_filename
    );


    // Benchmark OpenMP matrix multiplication
    double time_openmp = Benchmark::measure_and_report_matmul(
        std::to_string(N),
        "OpenMP",
        [&]() {Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::OpenMP);},
        report_filename
    );


    // Benchmark SIMD (AVX-512) matrix multiplication
    double time_avx512 = Benchmark::measure_and_report_matmul(
        std::to_string(N),
        "SIMD_AVX512",
        [&]() {Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::SIMD_AVX512);},
        report_filename
    );


    // Benchmark Optimized matrix multiplication
    double time_optimized = Benchmark::measure_and_report_matmul(
        std::to_string(N),
        "Optimized",
        [&]() {Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::Optimized);},
        report_filename
    );




    // Output results
    std::cout << "Average time for OpenBLAS MM: " << time_openblas << " ms\n";
    std::cout << "Average time for Vanilla MM: " << time_vanilla << " ms\n";
    std::cout << "Average time for Loop Unrolled by 8 MM: " << time_unrolled8 << " ms\n";
    std::cout << "Average time for Loop Unrolled by n (n = 8) MM: " << time_unrolled8_1 << " ms\n";
    std::cout << "Average time for Tiled MM: " << time_tiled << " ms\n";
    std::cout << "Average time for OpenMP MM: " << time_openmp << " ms\n";
    std::cout << "Average time for SIMD (AVX-512) MM: " << time_avx512 << " ms\n";
    std::cout << "Average time for Optimized MM: " << time_optimized << " ms\n";

    return 0;
}