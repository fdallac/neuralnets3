#pragma once

#include <chrono>
#include <functional>
#include <iostream>
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
};

