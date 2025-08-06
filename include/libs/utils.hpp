#pragma once
#include "traits.hpp"

namespace numcpp {
    template <typename V>
    array<V> arange(const range_t<V>& range) {
        return array(std::vector<V>(range.begin(), range.end()), {1, range.size()});
    }

    template <typename V>
    array<V> fill(const shape_t& shape, const V& value) {
        const size_t size = shape.rows * shape.cols;
        buffer_t<V> buf(size);
        std::fill_n(buf.data(), size, value);
        return array<V>(buf, shape, false);
    }

    template <typename V = double>
    array<V> ones(const shape_t& shape) {
        return fill(shape, V(1));
    }

    template <typename V>
    array<V> empty(const shape_t& shape) {
        if (shape.rows * shape.cols) {
            buffer_t<V> data(shape.rows * shape.cols);
            return array<V>(data, shape, false);
        }
        return array<V>();
    }

    template <typename V = double>
    array<V> zeros(const shape_t& shape) {
        return fill(shape, V());
    }
} // namespace numcpp
