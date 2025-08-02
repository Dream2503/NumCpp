#pragma once
#include "../libs/broadcasting.hpp"

namespace numcpp {
    template <typename L, typename R>
    array<promote_t<L, R>> operator+(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::plus());
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator+(const array<L>& lhs, const R& value) {
        auto [row, col] = lhs.shape();
        using V = promote_t<L, R>;
        std::vector<V> res;
        res.reserve(lhs.size());

        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                res.push_back(lhs.at(i, j) + value);
            }
        }

        return array<V>(std::move(res), row, col);
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator+(const L& value, const array<R>& rhs) {
        return rhs + value;
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator-(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::minus());
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator-(const array<L>& lhs, const R& value) {
        auto [row, col] = lhs.shape();
        using V = promote_t<L, R>;
        std::vector<V> res;
        res.reserve(lhs.size());

        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                res.push_back(lhs.at(i, j) - value);
            }
        }

        return array<V>(std::move(res), row, col);
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator-(const L& value, const array<R>& rhs) {
        return -rhs + value;
    }

    template <typename V>
    array<V> operator-(const array<V>& arr) {
        auto [row, col] = arr.shape();
        std::vector<V> res;
        res.reserve(arr.size());

        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                res.push_back(-arr.at(i, j));
            }
        }
        return array<V>(std::move(res), row, col);
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator*(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::multiplies());
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator*(const array<L>& lhs, const R& value) {
        auto [row, col] = lhs.shape();
        using V = promote_t<L, R>;
        std::vector<V> res;
        res.reserve(lhs.size());

        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                res.push_back(lhs.at(i, j) * value);
            }
        }
        return array<V>(std::move(res), row, col);
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator*(const L& value, const array<R>& rhs) {
        return rhs * value;
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator/(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, [](const L& x, const R& y) {
            if (y == decltype(y){}) {
                throw std::invalid_argument("Division by zero");
            }
            return x / y;
        });
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator/(const array<L>& lhs, const R& value) {
        using V = promote_t<L, R>;

        if (value == R{}) {
            throw std::invalid_argument("Division by zero in array / value");
        }
        auto [row, col] = lhs.shape();
        std::vector<V> res;
        res.reserve(lhs.size());

        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                res.push_back(static_cast<V>(lhs.at(i, j)) / value);
            }
        }
        return array<V>(std::move(res), row, col);
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator/(const L& value, const array<R>& rhs) {
        auto [row, col] = rhs.shape();
        using V = promote_t<L, R>;
        std::vector<V> res;
        res.reserve(rhs.size());

        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                if (rhs.at(i, j) == V{}) {
                    throw std::invalid_argument("Division by zero in value / array");
                }
                res.push_back(static_cast<V>(value) / rhs.at(i, j));
            }
        }
        return array<V>(std::move(res), row, col);
    }

} // namespace numcpp
