#!/bin/bash

# Run benchmarks for different matrix sizes
MATRIX_SIZES=(8 10 16 32 50 64 100 128 200 256 300 400 500 512 800 1000 1024 2000 2048 3000 4000 4096) 
for SIZE in "${MATRIX_SIZES[@]}"; do
    echo "Running benchmark for matrix size: $SIZE"
    ./test_benchmark "$SIZE"
done