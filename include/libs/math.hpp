#pragma once
#include "traits.hpp"

namespace numcpp::math {
    template <typename T, typename dtype = T>
    requires(is_numeric_v<T>)
    dtype absolute(const T& x) {
        if constexpr (is_complex_v<T>) {
            return static_cast<dtype>(x.abs());
        } else {
            return static_cast<dtype>(std::abs(x));
        }
    }

    template <typename T, typename dtype = bool>
    requires(is_numeric_v<T>)
    dtype all(const array<T>& a, const where_t& where) {
        auto [row, col] = a.shape();

        for (ll_t i = 0; i < row; i++) {
            for (ll_t j = 0; j < col; j++) {
                if ((!where || where->mask_at(i, j)) && !static_cast<bool>(a[{i, j}])) {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename T, typename U>
    requires(is_numeric_v<T> && is_numeric_v<U>)
    bool allclose(const T& a, const U& b, const float64_t rtol, const float64_t atol, const bool equal_nan) {
        if (std::isnan(a) && std::isnan(b)) {
            return equal_nan;
        }
        return absolute(a - b) <= atol + rtol * absolute(b);
    }

    template <typename T, typename U, typename dtype = promote_t<T, U>>
    requires(is_real_v<T> && is_real_v<U>)
    dtype arctan2(const T& y, const U& x) {
        return std::atan2(y, x);
    }

    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    dtype rad2deg(const T& x) {
        return 180 * x / pi;
    }

    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    dtype floor(const T& x) {
        return static_cast<dtype>(std::floor(x));
    }
} // namespace numcpp::math
