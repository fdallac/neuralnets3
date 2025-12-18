/**
 * @file bench.hpp
 * @brief Benchmarking utilities for performance measurement
 * 
 * Provides timing and reporting functionality for benchmarking
 * matrix operations and other performance-critical code.
 */

#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

/**
 * @brief Benchmark timing and reporting utilities
 * 
 * All methods are static - no instantiation needed.
 * Uses std::chrono::steady_clock for high-resolution timing.
 */
class Benchmark {
    public:
        /**
         * @brief Measure average execution time of a function
         * @param func Function to benchmark (takes no arguments, returns void)
         * @param trials Number of timed executions (default: 5)
         * @param warmup Number of untimed warm-up runs (default: 1)
         * @return Average execution time in milliseconds
         * 
         * Warm-up runs help stabilize CPU caches and branch predictors.
         * Returns mean time across all trials.
         */
        static double measure(
            const std::function<void()>& func,
            int trials = 5,
            int warmup = 1
        ) {
            // Warm-up runs (not timed)
            for (int i = 0; i < warmup; i++) {
                func();
            }

            double total_ms = 0.0;

            for (int i = 0; i < trials; i++) {
                auto start = std::chrono::steady_clock::now();

                func();

                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;

                total_ms += elapsed.count();
            }

            return total_ms / trials;
        }


        /**
         * @brief Measure and log matrix multiplication performance
         * @param matrix_size String describing matrix dimensions (e.g., "512x512")
         * @param matmul_method Name of multiplication method being tested
         * @param func Function to benchmark
         * @param report_filename CSV file to append results to
         * @param trials Number of timed executions (default: 5)
         * @param warmup Number of warm-up runs (default: 1)
         * @return Average execution time in milliseconds
         * @throws std::runtime_error if file cannot be opened
         * 
         * Appends row to CSV: timestamp,matrix_size,method,time_ms
         * Creates file with header if it doesn't exist.
         */
        static double measure_and_report_matmul(
            const std::string& matrix_size,
            const std::string& matmul_method,
            const std::function<void()>& func,
            const std::string& report_filename,
            int trials = 5,
            int warmup = 1
        ) {
            double avg_time_ms = measure(func, trials, warmup);
            // Save results to CSV
            std::ofstream file(report_filename, std::ios_base::app);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file for writing: " + report_filename);
            }
            std::time_t now = std::time(nullptr);
            char buf[100];
            std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
            std::string actual_timestamp(buf);

            file << actual_timestamp << "," << matrix_size << "," << matmul_method << "," << avg_time_ms << "\n";
            file.close();

            return avg_time_ms;
        }
};

