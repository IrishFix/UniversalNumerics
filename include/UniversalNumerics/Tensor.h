#ifndef TENSOR_H
#define TENSOR_H

#include <utility>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <numeric>

template <typename T>
class Tensor;

template <typename T>
class TensorSlice {
private:
    Tensor<T>& tensor;
    std::vector<size_t> indices;

public:
    TensorSlice(Tensor<T>& tensor, std::vector<size_t> initialIndices) : tensor(tensor) {
        indices = std::move(initialIndices);
    }

    TensorSlice<T>& operator[](size_t index) {
        const auto& shape = tensor.get_shape();
        if (indices.size() + 1 > shape.size()) {
            throw std::out_of_range("Too many indices for tensor dimensions.");
        }

        if (index >= shape[indices.size()]) {
            throw std::out_of_range("Index out of bounds for dimension " + std::to_string(indices.size()));
        }

        indices.push_back(index);
        return *this;
    }

    T& at(size_t index) const {
        const auto& shape = tensor.get_shape();
        std::vector<size_t> newIndices = indices;
        newIndices.push_back(index);

        if (newIndices.size() > shape.size()) {
            throw std::out_of_range("Too many indices for tensor dimensions.");
        }

        if (index >= shape[newIndices.size() - 1]) {
            throw std::out_of_range("Index out of bounds for dimension " + std::to_string(newIndices.size() - 1));
        }

        return tensor(newIndices);
    }

    const T& get() const {
        const auto& shape = tensor.get_shape();
        if (indices.size() != shape.size()) {
            throw std::out_of_range("Incorrect number of indices for tensor dimensions.");
        }

        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds for dimension " + std::to_string(i));
            }
        }

        return tensor(indices);
    }
};

template <typename T>
class Tensor {
private:
    std::vector<T> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    void compute_strides() {
        strides.resize(shape.size());
        size_t stride = 1;
        for (size_t i = shape.size(); i > 0; --i) {
            strides[i - 1] = stride;
            stride *= shape[i - 1];
        }
    }

public:
    Tensor() = default;

    Tensor(std::initializer_list<size_t> dims, const T& initial_value = T()) {
        shape.assign(dims);
        compute_strides();
        data.assign(std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>()), initial_value);
    }

    T& operator()(const std::vector<size_t>& indices) {
        if (indices.size() > get_shape().size()) {
            throw std::out_of_range("Too many indices for tensor dimensions.");
        }

        size_t index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds.");
            }
            index += indices[i] * strides[i];
        }

        return data[index];
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        if (indices.size() > get_shape().size()) {
            throw std::out_of_range("Too many indices for tensor dimensions.");
        }

        size_t index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds.");
            }
            index += indices[i] * strides[i];
        }

        return data[index];
    }

    TensorSlice<T> operator[](size_t index) {
        if (index > get_shape()[0]) {
            throw std::out_of_range("Index out of bounds.");
        }

        return TensorSlice<T>(*this, {index});
    }

    TensorSlice<T> operator[](size_t index) const {
        if (index > get_shape()[0]) {
            throw std::out_of_range("Index out of bounds.");
        }

        return TensorSlice<T>(*this, {index});
    }

    [[nodiscard]] const std::vector<size_t>& get_shape() const { return shape; }
    [[nodiscard]] const std::vector<size_t>& get_strides() const { return strides; }

    [[nodiscard]] size_t size() const { return data.size(); }

    void fill(const T& value) {
        std::fill(data.begin(), data.end(), value);
    }
};

#endif