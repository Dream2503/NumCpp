#pragma once

namespace numcpp {
    template <typename V>
    array<V> arange(const range<V>& range) {
        return array(std::vector<V>(range.begin(), range.end()), 1, range.size());
    }

    template <typename V>
    array<V> ones(const size_t row, const size_t column) {
        return fill(row, column, V(1));
    }

    template <typename V>
    array<V> fill(const size_t row, const size_t column, const V& value) {
        return array(std::vector(row * column, value), row, column);
    }

    template <typename V>
    array<V> zeros(const size_t row, const size_t column) {
        return fill(row, column, V(0));
    }
} // namespace numcpp
