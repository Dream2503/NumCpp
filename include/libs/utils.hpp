#pragma once
#include "traits.hpp"

namespace numcpp {
    template <typename V>
    array<V> arange(const range_t<V>& range) {
        return array(std::vector<V>(range.begin(), range.end()), {1, range.size()});
    }

    template <typename V>
    array<V> fill(const shape_t& shape, const V& value) {
        std::vector<V> res(shape.rows * shape.cols, value);
        return array(std::move(res), shape);
    }
    template <typename V = double>
    array<V> ones(const shape_t& shape) { return fill(shape, V() + 1); }

    template <typename V>
    array<V> empty(const shape_t& shape) {
        buffer_t<V> data(shape.rows * shape.cols);
        return array<V>(std::move(data), shape, false);
    }
    template <typename V = double>
    array<V> zeros(const shape_t& shape) { return fill(shape, V()); }
} // namespace numcpp
