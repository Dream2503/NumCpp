#pragma once

namespace numcpp::math {
    template <typename V, typename dtype = V>
    auto abs(const V value) {
        if constexpr (is_complex_v<V>) {
            return static_cast<dtype>(value.abs());
        } else {
            return static_cast<dtype>(std::abs(value));
        }
    }
} // namespace numcpp::math
