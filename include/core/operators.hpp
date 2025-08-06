#pragma once
#include "../libs/broadcasting.hpp"

namespace numcpp {
    template <typename L, typename R>
    array<promote_t<L, R>> operator+(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::plus());
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator+(const array<L>& lhs, const R& value) {
        using V = promote_t<L, R>;
        auto [row, col] = lhs.shape();
        buffer_t<V> res(lhs.size());

        for (ll_t i = 0; i < row; i++) {
            for (ll_t j = 0; j < col; j++) {
                res[i * col + j] = static_cast<V>(lhs[{i, j}]) + value;
            }
        }
        return array<V>(std::move(res), {row, col});
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
        using V = promote_t<L, R>;
        auto [row, col] = lhs.shape();
        buffer_t<V> res(lhs.size());

        for (ll_t i = 0; i < row; i++) {
            for (ll_t j = 0; j < col; j++) {
                res[i * col + j] = static_cast<V>(lhs[{i, j}]) - value;
            }
        }
        return array<V>(std::move(res), {row, col});
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator-(const L& value, const array<R>& rhs) {
        return -rhs + value;
    }

    template <typename V>
    array<V> operator-(const array<V>& arr) {
        auto [row, col] = arr.shape();
        buffer_t<V> res(arr.size());

        for (ll_t i = 0; i < row; i++) {
            for (ll_t j = 0; j < col; j++) {
                res[i * col + j] = -static_cast<V>(arr[{i, j}]);
            }
        }
        return array<V>(std::move(res), {row, col});
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator*(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::multiplies());
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator*(const array<L>& lhs, const R& value) {
        using V = promote_t<L, R>;
        auto [row, col] = lhs.shape();
        buffer_t<V> res(lhs.size());

        for (ll_t i = 0; i < row; i++) {
            for (ll_t j = 0; j < col; j++) {
                res[i * col + j] = static_cast<V>(lhs[{i, j}]) * value;
            }
        }
        return array<V>(std::move(res), {row, col});
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
        buffer_t<V> res(lhs.size());

        for (ll_t i = 0; i < row; i++) {
            for (ll_t j = 0; j < col; j++) {
                res[i * col + j] = static_cast<V>(lhs[{i, j}]) / value;
            }
        }
        return array<V>(std::move(res), {row, col});
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator/(const L& value, const array<R>& rhs) {
        using V = promote_t<L, R>;
        auto [row, col] = rhs.shape();
        buffer_t<V> res(rhs.size());

        for (ll_t i = 0; i < row; i++) {
            for (ll_t j = 0; j < col; j++) {
                if (rhs[{i, j}] == V()) {
                    throw std::invalid_argument("Division by zero in value / array");
                }
                res[i * col + j] = static_cast<V>(value) / static_cast<V>(rhs[{i, j}]);
            }
        }
        return array<V>(std::move(res), {row, col});
    }
} // namespace numcpp
