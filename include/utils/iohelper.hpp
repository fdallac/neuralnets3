#include "matrix.hpp"
#include <vector>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>


template<typename T>
class IOHelper {
    private:
        // Helper function to trim whitespace from string
        static std::string trim(const std::string& str) {
            size_t first = str.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) return "";
            size_t last = str.find_last_not_of(" \t\r\n");
            return str.substr(first, (last - first + 1));
        }

    public:
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