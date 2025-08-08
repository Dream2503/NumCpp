#pragma once
#include "math.hpp"
#include "utils.hpp"

namespace numcpp {
    template <typename T>
    template <typename dtype>
    requires(is_numeric_v<T>)
    array<real_t<dtype>> array<T>::abs(out_t<real_t<dtype>> out, const where_t& where) const noexcept {
        using V = real_t<dtype>;
        array<V> result = out ? *out.ptr : empty<V>({row, col});

        for (ll_t i = 0; i < row; i++) {
            for (ll_t j = 0; j < col; j++) {
                result[{i, j}] = !where || (*where)[{i, j}] ? math::abs<T, dtype>((*this)[{i, j}]) : V(0);
            }
        }
        if (out) {
            return *out.ptr;
        }
        return result;
    }

    template <typename T>
    template <typename dtype>
    requires(is_numeric_v<T>)
    array<real_t<dtype>> array<T>::abs(const where_t& where) const noexcept {
        return abs<real_t<dtype>>(none::out, where);
    }

    template <typename T>
    template <typename dtype>
    requires(is_numeric_v<T> && !is_complex_v<T>)
    array<dtype> array<T>::floor(out_t<dtype> out, const where_t& where) const noexcept {
        array<dtype> result = out ? *out.ptr : empty<dtype>({row, col});

        for (ll_t i = 0; i < row; i++) {
            for (ll_t j = 0; j < col; j++) {
                result[{i, j}] = !where || (*where)[{i, j}] ? math::floor<dtype, dtype>((*this)[{i, j}]) : dtype(0);
            }
        }
        if (out) {
            return *out.ptr;
        }
        return result;
    }

    template <typename T>
    template <typename dtype>
    requires(is_numeric_v<T> && !is_complex_v<T>)
    array<dtype> array<T>::floor(const where_t& where) const noexcept {
        return floor<dtype>(none::out, where);
    }

    template <typename T>
    auto array<T>::real(out_t<real_t<T>> out, const where_t& where) noexcept requires(is_numeric_v<T>)
    {
        using V = real_t<T>;

        if (out || where) {
            array<V> result = out ? *out.ptr : empty<V>({row, col});

            for (ll_t i = 0; i < row; i++) {
                for (ll_t j = 0; j < col; j++) {
                    bool condition = !where || (*where)[{i, j}];

                    if constexpr (is_complex_v<T>) {
                        result[{i, j}] = condition ? static_cast<V>((*this)[{i, j}]).real : V(0);
                    } else {
                        result[{i, j}] = condition ? (*this)[{i, j}] : V(0);
                    }
                }
            }
            if (out) {
                return *out.ptr;
            }
            return result;
        }
        if constexpr (is_complex_v<T>) {
            return array<V>(buffer_t<V>(reinterpret_cast<V*>(buffer.data()), size() * 2, nullptr), {row, col},
                            offset * 2, row_stride * 2, col_stride * 2, this, is_matrix, is_scalar, true);
        } else {
            array res = *this;
            res.is_assignable = true;
            return res;
        }
    }

    template <typename T>
    auto array<T>::real(const where_t& where) const noexcept requires(is_numeric_v<T>)
    {
        return real(none::out, where);
    }

    template <typename T>
    auto array<T>::imag(out_t<real_t<T>> out, const where_t& where) noexcept requires(is_numeric_v<T>)
    {
        using V = real_t<T>;

        if (out || where) {
            array<V> result = out ? *out.ptr : empty<V>({row, col});

            for (ll_t i = 0; i < row; i++) {
                for (ll_t j = 0; j < col; j++) {
                    bool condition = !where || (*where)[{i, j}];

                    if constexpr (is_complex_v<T>) {
                        result[{i, j}] = condition ? static_cast<V>((*this)[{i, j}]).imag : V(0);
                    } else {
                        result[{i, j}] = condition ? (*this)[{i, j}] : V(0);
                    }
                }
            }
            if (out) {
                return *out.ptr;
            }
            return result;
        }
        if constexpr (is_complex_v<T>) {
            return array<V>(buffer_t<V>(reinterpret_cast<V*>(buffer.data()), size() * 2, nullptr), {row, col},
                            offset * 2 + 1, row_stride * 2, col_stride * 2, this, is_matrix, is_scalar, true);
        } else {
            array res = *this;
            res.is_assignable = true;
            return res;
        }
    }

    template <typename T>
    auto array<T>::imag(const where_t& where) const noexcept requires(is_numeric_v<T>)
    {
        return imag(none::out, where);
    }
} // namespace numcpp
