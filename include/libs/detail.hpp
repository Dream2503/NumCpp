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
        } else if constexpr (std::is_same_v<V, dtype::bitref_t> || std::is_same_v<V, bool>) {
            return value ? "true" : "false";
        } else if constexpr (std::is_same_v<V, dtype::str>) {
            return value;
        } else {
            return std::to_string(value);
        }
    }

    constexpr auto divides() noexcept {
        return [](auto left, auto right) {
            if (right == decltype(right)()) {
                throw std::invalid_argument("Division by zero");
            }
            return left / right;
        };
    }

    constexpr auto modulus() noexcept {
        return [](auto left, auto right) {
            if (right == decltype(right)()) {
                throw std::invalid_argument("Division by zero");
            }
            return left % right;
        };
    }
} // namespace numcpp::detail
