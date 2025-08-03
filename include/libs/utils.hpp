#pragma once
#include "traits.hpp"

namespace numcpp {
    template <typename V>
    array<V> arange(const range<V>&);
    inline array<double> ones(const shape_t&);
    template <typename V>
    array<V> fill(const shape_t&, const V&);
    inline array<double> zeros(const shape_t&);
} // namespace numcpp

namespace numcpp {
    template <typename V>
    array<V> arange(const range<V>& range) {
        return array(std::vector<V>(range.begin(), range.end()), 1, range.size());
    }

    array<double> ones(const shape_t& shape) { return fill(shape, 1.0); }

    template <typename V>
    array<V> fill(const shape_t& shape, const V& value) {}

    array<double> zeros(const shape_t& shape) { return fill(shape, 0.0); }
} // namespace numcpp
