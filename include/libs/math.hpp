#pragma once

namespace numcpp::math {
    template <typename T, typename dtype = T>
    requires(is_numeric_v<T>)
    auto abs(const T value) {
        if constexpr (is_complex_v<T>) {
            return static_cast<real_t<dtype>>(value.abs());
        } else {
            return static_cast<dtype>(std::abs(value));
        }
    }

    template <typename T, typename dtype = T>
    requires(is_numeric_v<T> && !is_complex_v<T>)
    auto floor(const T value) {
        return static_cast<dtype>(std::floor(value));
    }
} // namespace numcpp::math
