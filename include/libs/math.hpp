#pragma once

namespace numcpp::math {
    template <typename T, typename dtype = T>
    requires(is_numeric_v<T>)
    dtype abs(const T& value) {
        if constexpr (is_complex_v<T>) {
            return static_cast<dtype>(value.abs());
        } else {
            return static_cast<dtype>(std::abs(value));
        }
    }

    template <typename T, typename dtype = bool>
    requires(is_numeric_v<T>)
    dtype all(const array<T>& arr, const where_t& where) {
        auto [row, col] = arr.shape();

        for (ll_t i = 0; i < row; i++) {
            for (ll_t j = 0; j < col; j++) {
                if ((!where || where->mask_at(i, j)) && !static_cast<bool>(static_cast<T>(arr[{i, j}]))) {
                    return 0;
                }
            }
        }
        return 1;
    }

    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    dtype floor(const T& value) {
        return static_cast<dtype>(std::floor(value));
    }
} // namespace numcpp::math
