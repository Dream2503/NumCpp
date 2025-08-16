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
                if ((!where || (*where)[{i, j}]) && !static_cast<bool>(a[{i, j}])) {
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

    template <typename T, typename dtype = bool>
    requires(is_numeric_v<T>)
    dtype any(const array<T>& a, const where_t& where) {
        auto [row, col] = a.shape();

        for (ll_t i = 0; i < row; i++) {
            for (ll_t j = 0; j < col; j++) {
                if ((!where || (*where)[{i, j}]) && static_cast<bool>(a[{i, j}])) {
                    return true;
                }
            }
        }
        return false;
    }

    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    dtype arccos(const T& x) {
        return std::acos(x);
    }

    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    dtype arccosh(const T& x) {
        return std::acosh(x);
    }

    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    dtype arcsin(const T& x) {
        return std::asin(x);
    }

    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    dtype arcsinh(const T& x) {
        return std::asinh(x);
    }

    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    dtype arctan(const T& x) {
        return std::atan(x);
    }

    template <typename T, typename U, typename dtype = promote_t<T, U>>
    requires(is_real_v<T> && is_real_v<U>)
    dtype arctan2(const T& y, const U& x) {
        return std::atan2(y, x);
    }

    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    dtype arctanh(const T& x) {
        return std::atanh(x);
    }

    template <typename T>
    size_t argmax(const array<T>& a) {
        return std::max_element(a.begin(), a.end()) - a.begin();
    }

    template <typename T>
    size_t argmin(const array<T>& a) {
        return std::min_element(a.begin(), a.end()) - a.begin();
    }

    template <typename T>
    void argpartition(array<size_t> index_arr, const array<T>& arr, const size_t kth) {
        std::iota(index_arr.begin(), index_arr.end(), 0);
        std::nth_element(
            index_arr.begin(), index_arr.begin() + kth, index_arr.end(),
            [&arr](const size_t i, const size_t j) { return static_cast<T>(arr[i]) < static_cast<T>(arr[j]); });
    }

    template <typename T>
    void argsort(array<size_t> index_arr, const array<T>& arr, const std::string& kind = "quicksort",
                 const bool stable = false) {
        std::iota(index_arr.begin(), index_arr.end(), 0);

        auto comp = [&arr](const size_t i, const size_t j) { return static_cast<T>(arr[i]) < static_cast<T>(arr[j]); };

        if (stable || kind == "stable" || kind == "mergesort") {
            std::stable_sort(index_arr.begin(), index_arr.end(), comp);
        } else if (kind == "heapsort") {
            std::make_heap(index_arr.begin(), index_arr.end(), comp);
            std::sort_heap(index_arr.begin(), index_arr.end(), comp);
        } else if (kind == "quicksort") {
            std::sort(index_arr.begin(), index_arr.end(), comp);
        } else {
            throw std::invalid_argument("Unsupported kind");
        }
    }

    template <typename T, typename U>
    bool equal(const array<T>& x, const array<U>& y, const bool equal_nan) {
        auto itr1 = x.begin(), itr2 = y.begin(), end = x.end();

        while (itr1 != end) {
            if (*itr1 != *itr2 || (std::isnan(*itr1) && std::isnan(*itr2) && !equal_nan)) {
                return false;
            }
            ++itr1;
            ++itr2;
        }
        return true;
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
