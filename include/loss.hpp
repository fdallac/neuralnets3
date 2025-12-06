#pragma once

#include "matrix.hpp"

template<typename T>
class Loss {
    public:
        virtual T forward(const Matrix<T>& predictions, const Matrix<T>& targets) = 0;
        virtual Matrix<T> backward(const Matrix<T>& predictions, const Matrix<T>& targets) = 0;
        virtual ~Loss() = default;
};

template<typename T>
class MSELoss : public Loss<T> {
    public:
        T forward(const Matrix<T>& predictions, const Matrix<T>& targets) override;
        Matrix<T> backward(const Matrix<T>& predictions, const Matrix<T>& targets) override;
};

template<typename T>
class CrossEntropyLoss : public Loss<T> {
    public:
        T forward(const Matrix<T>& predictions, const Matrix<T>& targets) override;
        Matrix<T> backward(const Matrix<T>& predictions, const Matrix<T>& targets) override;
};
    