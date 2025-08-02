#pragma once
#include <algorithm>
#include <complex>
#include <iomanip>
#include <iostream>
#include <memory>
#include <type_traits>
#include <variant>
#include "libs/traits.hpp"

using ll_t = long long;

template <typename T>
std::string format(const T&);

template <typename V>
std::string to_string(const V& value) {
    if constexpr (numcpp::is_complex_v<V>) {
        using T = typename V::value_type;
        const T real = value.real(), imag = value.imag();
        std::ostringstream oss;

        if (imag == T(0)) {
            oss << format(real);
        } else if (real == T(0)) {
            oss << format(imag) << "j";
        } else {
            oss << format(real) << (imag < 0 ? "-" : "+") << format(std::abs(imag)) << "j";
        }
        return oss.str();
    } else {
        return std::to_string(value);
    }
}

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
        return res;
    } else {
        return res;
    }
}

namespace numcpp {
    template <typename V>
    class array;

    template <typename V>
    struct range;
    class slice;
    struct index;
} // namespace numcpp


#include "core/array.hpp"
#include "core/indexing.hpp"
#include "core/operators.hpp"
#include "libs/broadcasting.hpp"
#include "libs/utils.hpp"
