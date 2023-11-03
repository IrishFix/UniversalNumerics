#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <initializer_list>
#include <stdexcept>

template <typename T>
class Tensor {
private:
    std::vector<std::vector<T>> data;
    size_t rows, cols;

public:
    Tensor() : rows(0), cols(0) {}

    Tensor(size_t rows, size_t cols, const T& initial_value = T())
            : rows(rows), cols(cols), data(rows, std::vector<T>(cols, initial_value)) {}

    Tensor(std::initializer_list<std::initializer_list<T>> list)
            : rows(list.size()), cols(0), data(rows) {
        size_t i = 0;
        for (const auto& row : list) {
            data[i] = std::vector<T>(row);
            if (cols == 0) {
                cols = row.size();
            } else if (cols != row.size()) {
                throw std::invalid_argument("All rows must have the same number of columns");
            }
            ++i;
        }
    }

    Tensor(const Tensor& other)
            : rows(other.rows), cols(other.cols), data(other.data) {}

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            data = other.data;
        }
        return *this;
    }

    std::vector<T>& operator[](size_t index) {
        if (index >= rows) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    const std::vector<T>& operator[](size_t index) const {
        if (index >= rows) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    void fill(const T& value) {
        for (auto& row : data) {
            std::fill(row.begin(), row.end(), value);
        }
    }
};

#endif