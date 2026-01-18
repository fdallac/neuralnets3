/**
 * @file numpy_matrix_helper.hpp
 * @brief Zero-copy conversion utilities between NumPy arrays and C++ Matrix
 * 
 * This header provides internal utilities for converting between NumPy arrays
 * and the C++ Matrix class.
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matrix/matrix.hpp"

namespace py = pybind11;

namespace pynn3 {
namespace internal {

/**
 * @brief Create a C++ Matrix from NumPy array data
 * 
 * Copies data from the NumPy buffer into a new Matrix.
 * 
 * @tparam T Numeric type (float, double)
 * @param array Input NumPy array (must be 2D, C-contiguous)
 * @return Matrix<T> with copied data
 * @throws std::runtime_error if array is not 2D
 */
template<typename T>
Matrix<T> numpy_to_matrix(py::array_t<T, py::array::c_style | py::array::forcecast> array) {
    py::buffer_info buf = array.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D array");
    }
    
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    T* ptr = static_cast<T*>(buf.ptr);
    
    // Copy data into vector for Matrix constructor
    std::vector<T> vec_data(ptr, ptr + (rows * cols));
    return Matrix<T>(rows, cols, vec_data);
}


/**
 * @brief Create a NumPy array from C++ Matrix (with copy)
 * 
 * Creates a new NumPy array and copies data from the Matrix.
 * 
 * @tparam T Numeric type (float, double)
 * @param matrix Input C++ Matrix
 * @return NumPy array with copied data
 */
template<typename T>
py::array_t<T> matrix_to_numpy_copy(const Matrix<T>& matrix) {
    // Create output array with proper shape
    py::array_t<T> result({static_cast<py::ssize_t>(matrix.rows()), 
                           static_cast<py::ssize_t>(matrix.cols())});
    
    py::buffer_info buf = result.request();
    T* ptr = static_cast<T*>(buf.ptr);
    
    // Fast memory copy from Matrix to NumPy buffer
    std::memcpy(ptr, matrix.data(), sizeof(T) * matrix.rows() * matrix.cols());
    
    return result;
}


/**
 * @brief Create a NumPy array that owns Matrix memory (zero-copy)
 * 
 * Moves the Matrix into a capsule so NumPy owns the memory.
 * The Matrix is consumed - do not use it after this call.
 * 
 * This is the preferred method for returning large results to Python
 * as it avoids unnecessary data copying.
 * 
 * @tparam T Numeric type (float, double)
 * @param matrix Matrix to wrap (will be moved/consumed)
 * @return NumPy array that owns the Matrix memory
 */
template<typename T>
py::array_t<T> matrix_to_numpy_zerocopy(Matrix<T>&& matrix) {
    // Move matrix to heap so it outlives this function
    auto* matrix_ptr = new Matrix<T>(std::move(matrix));
    
    // Create capsule that will delete the Matrix when NumPy array is destroyed
    py::capsule free_when_done(matrix_ptr, [](void* p) {
        delete reinterpret_cast<Matrix<T>*>(p);
    });
    
    return py::array_t<T>(
        {static_cast<py::ssize_t>(matrix_ptr->rows()), 
         static_cast<py::ssize_t>(matrix_ptr->cols())}, // shape
        {static_cast<py::ssize_t>(matrix_ptr->cols() * sizeof(T)), 
         static_cast<py::ssize_t>(sizeof(T))},          // strides (row-major)
        matrix_ptr->data(),                             // data pointer
        free_when_done                                  // prevent deallocation
    );
}

} // namespace internal
} // namespace pynn3
