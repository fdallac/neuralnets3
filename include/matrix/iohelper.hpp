/**
 * @file iohelper.hpp
 * @brief CSV input/output utilities for matrices
 * 
 * Provides static methods to read and write matrices in CSV format.
 */

#pragma once

#include "matrix/matrix.hpp"
#include <vector>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>


/**
 * @brief CSV I/O helper for Matrix class
 * @tparam T Numeric type (float, double, int, etc.)
 * 
 * All methods are static - no instantiation needed.
 * Supports custom delimiters and optional headers.
 */
template<typename T>
class IOHelper {
    private:
        /**
         * @brief Trim leading/trailing whitespace from string
         * @param str Input string
         * @return Trimmed string
         */
        static std::string trim(const std::string& str) {
            size_t first = str.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) return "";
            size_t last = str.find_last_not_of(" \t\r\n");
            return str.substr(first, (last - first + 1));
        }

    public:
        /**
         * @brief Write matrix to CSV file
         * @param M Matrix to write
         * @param filename Output file path
         * @param sep Delimiter character (default: ',')
         * @param add_header Whether to add header line
         * @param header Header string (if add_header is true)
         * @throws std::runtime_error if file cannot be opened
         */
        static void write_csv(const Matrix<T>& M, const std::string& filename, char sep=',', bool add_header=false, const std::string& header="") {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file for writing: " + filename);
            }
            if (add_header && !header.empty()) {
                file << header << "\n";
            }
            for (std::size_t i = 0; i < M.rows(); ++i) {
                for (std::size_t j = 0; j < M.cols(); ++j) {
                    file << M(i, j);
                    if (j < M.cols() - 1) {
                        file << sep;
                    }
                }
                file << "\n";
            }
            file.close();
        }

        /**
         * @brief Read matrix from CSV file
         * @param filename Input file path
         * @param sep Delimiter character (default: ',')
         * @param skip_header Whether to skip first line as header
         * @return Matrix constructed from CSV data
         * @throws std::runtime_error if file cannot be opened or data is malformed
         * 
         * Automatically trims whitespace from values.
         * All rows must have the same number of columns.
         */
        static Matrix<T> read_csv(const std::string& filename, char sep=',', bool skip_header=false) {
            std::ifstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file for reading: " + filename);
            }
            std::vector<T> data;
            std::vector<T> data_buffer;
            std::size_t row_count = 0; 
            std::size_t col_count = 0;
            std::size_t buffer_col_count;

            std::string line;
            while (std::getline(file, line)) {
                // Check header
                if (skip_header) {
                    // Count columns in the header
                    std::stringstream ss_header(line);
                    std::string value_header;
                    col_count = 0;
                    while (std::getline(ss_header, value_header, sep)) col_count++;
                    skip_header = false;  // Reset flag BEFORE continue
                    continue; // Skip header
                }

                // Parse data line
                std::stringstream ss(line);
                std::string value;

                buffer_col_count = 0;
                while (std::getline(ss, value, sep)) {
                    buffer_col_count++;
                    std::string trimmed_value = trim(value);
                    data_buffer.push_back(static_cast<T>(std::stod(trimmed_value)));
                }

                if (col_count == 0) {
                    // First data row sets column count
                    col_count = buffer_col_count;
                } else if (col_count != buffer_col_count) {
                    throw std::runtime_error("Inconsistent number of columns in CSV file at line " + std::to_string(row_count + 1) + " in file: " + filename);
                }

                data.insert(data.end(), data_buffer.begin(), data_buffer.end());
                data_buffer.clear();

                row_count++;
            }   

            file.close();

            if (data.empty()) return Matrix<T>(0, 0);
            return Matrix<T>(row_count, col_count, data);
        }
};