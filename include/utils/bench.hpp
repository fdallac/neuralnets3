#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

class Benchmark {
    public:
        /**
         * Runs the provided function multiple times and returns the average elapsed time in milliseconds.
         * 
         * @param func      Function to benchmark
         * @param trials    Number of measurement runs
         * @param warmup    Number of warm-up runs
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

