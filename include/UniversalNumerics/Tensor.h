#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <numeric>

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
        data.assign(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()), initial_value);
    }

    // Access elements.
    T& operator()(const std::vector<size_t>& indices) {
        size_t index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            index += indices[i] * strides[i];
        }
        return data[index];
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        size_t index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            index += indices[i] * strides[i];
        }
        return data[index];
    }

    const std::vector<size_t>& get_shape() const { return shape; }
    const std::vector<size_t>& get_strides() const { return strides; }

    void fill(const T& value) {
        std::fill(data.begin(), data.end(), value);
    }

    size_t size() const {
        return data.size();
    }
};

#endif