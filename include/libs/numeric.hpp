#pragma once
#include "math.hpp"
#include "utils.hpp"

namespace numcpp {
    template <typename V>
    template <typename dtype>
    array<real_t<dtype>> array<V>::abs(out_t<real_t<dtype>> out, const where_t& where) const {
        array<real_t<dtype>> result = out ? *out.ptr : empty<real_t<dtype>>({row, col});

        for (ll_t i = 0; i < row; i++) {
            for (ll_t j = 0; j < col; j++) {
                result[{i, j}] = !where || (*where)[{i, j}] ? math::abs<V, dtype>((*this)[{i, j}]) : real_t<dtype>(0);
            }
        }
        if (out) {
            return *out.ptr;
        }
        return result;
    }

    template <typename V>
    auto array<V>::real(out_t<real_t<V>> out, const where_t& where) {
        using T = real_t<V>;

        if (out || where) {
            array<T> result = out ? *out.ptr : empty<T>({row, col});

            for (ll_t i = 0; i < row; i++) {
                for (ll_t j = 0; j < col; j++) {
                    bool take = !where || (*where)[{i, j}];
                    if constexpr (is_complex_v<V>) {
                        result[{i, j}] = take ? static_cast<V>((*this)[{i, j}]).real : T(0);
                    } else {
                        result[{i, j}] = take ? (*this)[{i, j}] : T(0);
                    }
                }
            }
            if (out) {
                return *out.ptr;
            }
            return result;
        }
        if constexpr (is_complex_v<V>) {
            return array<T>(buffer_t<T>(reinterpret_cast<T*>(buffer.data()), size() * 2, nullptr), {row, col},
                            offset * 2, row_stride * 2, col_stride * 2, this, is_matrix, is_scalar, true);
        } else {
            array res = *this;
            res.is_assignable = true;
            return res;
        }
    }

    template <typename V>
    auto array<V>::imag(out_t<real_t<V>> out, const where_t& where) {
        using T = real_t<V>;

        if (out || where) {
            array<T> result = out ? *out.ptr : empty<T>({row, col});

            for (ll_t i = 0; i < row; i++) {
                for (ll_t j = 0; j < col; j++) {
                    bool take = !where || (*where)[{i, j}];
                    if constexpr (is_complex_v<V>) {
                        result[{i, j}] = take ? static_cast<V>((*this)[{i, j}]).imag : T(0);
                    } else {
                        result[{i, j}] = take ? (*this)[{i, j}] : T(0);
                    }
                }
            }
            if (out) {
                return *out.ptr;
            }
            return result;
        }
        if constexpr (is_complex_v<V>) {
            return array<T>(buffer_t<T>(reinterpret_cast<T*>(buffer.data()), size() * 2, nullptr), {row, col},
                            offset * 2 + 1, row_stride * 2, col_stride * 2, this, is_matrix, is_scalar, true);
        } else {
            array res = *this;
            res.is_assignable = true;
            return res;
        }
    }
} // namespace numcpp
