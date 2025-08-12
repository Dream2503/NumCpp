#pragma once
#include "types.hpp"

namespace numcpp::detail {
    template <typename V>
    std::string format(const V& val) {
        std::string res = to_string(val);

        if constexpr (std::is_floating_point_v<V>) {
            const size_t dot = res.find('.');

            if (dot != std::string::npos) {
                const size_t last_non_zero = res.find_last_not_of('0');

                if (last_non_zero == dot) {
                    return res.substr(0, dot + 1);
                }
                return res.substr(0, last_non_zero + 1);
            }
        }
        return res;
    }

    template <typename V>
    std::string to_string(const V& value) {
        if constexpr (numcpp::is_complex_v<V>) {
            std::string res;

            if (!value.imag) {
                res.append(detail::format(value.real));
            } else if (!value.real) {
                res.append(detail::format(value.imag));
                res.push_back('j');
            } else {
                res.append(detail::format(value.real));
                res.push_back(value.imag < 0 ? '-' : '+');
                res.append(detail::format(std::abs(value.imag)));
                res.push_back('j');
            }
            return res;
        } else if constexpr (std::is_same_v<V, dtypes::bitref_t> || std::is_same_v<V, bool>) {
            return value ? "true" : "false";
        } else if constexpr (std::is_same_v<V, dtypes::str>) {
            return value;
        } else {
            return std::to_string(value);
        }
    }

    template <typename T>
    T division_by_zero_warning(T left, const char error[]) {
        std::cerr << "RuntimeWarning: divide by zero encountered in " << error << std::endl;

        if constexpr (is_floating_point_v<T>) {
            if (left == T()) {
                return nan;
            }
            return std::copysign(std::numeric_limits<T>::infinity(), left);
        } else if constexpr (is_complex_v<T>) {
            using V = typename T::value_type;

            if (left == T()) {
                return T(nan, nan);
            }
            return T(std::copysign(inf, left.real()), std::copysign(inf, left.imag()));
        } else if constexpr (is_integral_v<T>) {
            return T();
        }
        return left;
    }

    constexpr auto divides() noexcept {
        return [](auto left, auto right) {
            if (right == decltype(left)()) {
                return division_by_zero_warning(left, __PRETTY_FUNCTION__);
            }
            return left / right;
        };
    }

    constexpr auto modulus() noexcept {
        return [](auto left, auto right) {
            if (right == decltype(left)()) {
                return division_by_zero_warning(left, __PRETTY_FUNCTION__);
            }
            return left % right;
        };
    }
} // namespace numcpp::detail
