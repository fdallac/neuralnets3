#include "utils/bench.hpp"
#include "matrix/matrix.hpp"
#include "matrix/matmul.hpp"
#include <iostream>

extern "C" {
    #include <cblas.h>
}

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

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

    std::cout << "Average time for OpenBLAS MM: " << time_openblas << " ms\n";

/*

    // Benchmark vanilla matrix multiplication
    double time_vanilla = Benchmark::measure_and_report_matmul(
        std::to_string(N),
        "Vanilla",
        [&]() {Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::Vanilla);},
        report_filename
    );

    std::cout << "Average time for Vanilla MM: " << time_vanilla << " ms\n";

    

    // Benchmark loop unrolled by 8 matrix multiplication
    double time_unrolled8 = Benchmark::measure_and_report_matmul(
        std::to_string(N),
        "LoopUnrolled_8",
        [&]() {Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::LoopUnrolled8);},
        report_filename
    );

    std::cout << "Average time for Loop Unrolled by 8 MM: " << time_unrolled8 << " ms\n";


    double time_unrolled8_1 = Benchmark::measure_and_report_matmul(
        std::to_string(N),
        "LoopUnrolled_custom_8",
        [&]() {Matrix<double> C = MatMul<double>::mm_unrolled<8>(A, B);},
        report_filename
    );

    std::cout << "Average time for Loop Unrolled (template 8) MM: " << time_unrolled8_1 << " ms\n";


    // Benchmark tiling matrix multiplication
    double time_tiled = Benchmark::measure_and_report_matmul(
        std::to_string(N),
        "Tiled",
        [&]() {Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::Tiled);},
        report_filename
    );

    std::cout << "Average time for Tiled MM: " << time_tiled << " ms\n";


    // Benchmark OpenMP matrix multiplication
    double time_openmp = Benchmark::measure_and_report_matmul(
        std::to_string(N),
        "OpenMP",
        [&]() {Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::OpenMP);},
        report_filename
    );

    std::cout << "Average time for OpenMP MM: " << time_openmp << " ms\n";


    // Benchmark SIMD (AVX-512) matrix multiplication
    double time_avx512 = Benchmark::measure_and_report_matmul(
        std::to_string(N),
        "SIMD_AVX512",
        [&]() {Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::SIMD_AVX512);},
        report_filename
    );

    std::cout << "Average time for SIMD (AVX-512) MM: " << time_avx512 << " ms\n";


    // Benchmark SIMD+OpenMP+Tile matrix multiplication
    double time_simd_openmp_tile = Benchmark::measure_and_report_matmul(
        std::to_string(N),
        "SIMD_OpenMP_Tile",
        [&]() {Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::SIMD_OpenMP_Tile);},
        report_filename
    );

    std::cout << "Average time for SIMD+OpenMP+Tile MM: " << time_simd_openmp_tile << " ms\n";
 
*/   


    // Benchmark CUDA matrix multiplication if available
#ifdef CUDA_AVAILABLE


    // Benchmark cuBLAS matrix multiplication for reference
    try {
        // Initialize cuBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Allocate device memory
        double *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, N * N * sizeof(double));
        cudaMalloc(&d_B, N * N * sizeof(double));
        cudaMalloc(&d_C, N * N * sizeof(double));

        // Copy data to device
        cudaMemcpy(d_A, A.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);

        const double alpha = 1.0;
        const double beta = 0.0;

        // // Warm-up run
        // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        //             N, N, N,
        //             &alpha,
        //             d_B, N,  // Note: cuBLAS uses column-major, so we swap A and B
        //             d_A, N,
        //             &beta,
        //             d_C, N);
        // cudaDeviceSynchronize();

        double time_cublas = Benchmark::measure_and_report_matmul(
            std::to_string(N),
            "cuBLAS",
            [&]() {
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, N, N,
                            &alpha,
                            d_B, N,
                            d_A, N,
                            &beta,
                            d_C, N);
                cudaDeviceSynchronize();
            },
            report_filename
        );

        std::cout << "Average time for cuBLAS MM: " << time_cublas << " ms\n";

        // Cleanup
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
    } catch (const std::exception& e) {
        std::cerr << "cuBLAS benchmark skipped: " << e.what() << "\n";
    }



    // Benchmark CUDA matrix multiplication
    try {
        double time_cuda = Benchmark::measure_and_report_matmul(
            std::to_string(N),
            "CUDA",
            [&]() {Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::CUDA);},
            report_filename
        );

        std::cout << "Average time for CUDA MM: " << time_cuda << " ms\n";
    } catch (const std::exception& e) {
        std::cerr << "CUDA benchmark skipped: " << e.what() << "\n";
    }

    // Benchmark CUDA+OpenMP hybrid matrix multiplication
    try {
        double time_cuda_openmp = Benchmark::measure_and_report_matmul(
            std::to_string(N),
            "CUDA_OpenMP",
            [&]() {Matrix<double> C = MatMul<double>::mm(A, B, MatMulMethod::CUDA_OpenMP);},
            report_filename
        );

        std::cout << "Average time for CUDA+OpenMP MM: " << time_cuda_openmp << " ms\n";
    } catch (const std::exception& e) {
        std::cerr << "CUDA+OpenMP benchmark skipped: " << e.what() << "\n";
    }
#endif

    std::cout << "Benchmark completed. Results saved to " << report_filename << "\n";

    return 0;
}