#pragma once
#include "ufunc.hpp"

namespace numcpp {
    template <typename T, typename dtype = T>
    requires(is_numeric_v<T>)
    array<dtype> absolute(const array<T>& x, out_t<dtype> out = none::out<dtype>, const where_t& where = none::where) {
        return ufunc_unary(x, out, where, &math::absolute<T, dtype>);
    }
    template <typename dtype, typename T>
    requires(is_numeric_v<T>)
    array<dtype> absolute(const array<T>& x, const where_t& where = none::where) {
        return absolute(x, none::out<dtype>, where);
    }
    template <typename T, typename dtype = T>
    requires(is_numeric_v<T>)
    array<dtype> absolute(const array<T>& x, const where_t& where) {
        return absolute(x, none::out<dtype>, where);
    }

    template <typename T, typename dtype = T>
    requires(is_numeric_v<T>)
    array<dtype> abs(const array<T>& x, out_t<dtype> out = none::out<dtype>, const where_t& where = none::where) {
        return absolute(x, out, where);
    }
    template <typename dtype, typename T>
    requires(is_numeric_v<T>)
    array<dtype> abs(const array<T>& x, const where_t& where = none::where) {
        return absolute(x, where);
    }
    template <typename T, typename dtype = T>
    requires(is_numeric_v<T>)
    array<dtype> abs(const array<T>& x, const where_t& where) {
        return absolute(x, where);
    }

    template <typename T, typename U, typename dtype = promote_t<T, U>>
    array<dtype> add(const array<T>& x, const array<U>& y, out_t<dtype> out = none::out<dtype>,
                     const where_t& where = none::where) {
        return ufunc_binary(x, y, out, where, std::plus());
    }
    template <typename dtype, typename T, typename U>
    array<dtype> add(const array<T>& x, const array<U>& y, const where_t& where = none::where) {
        return add(x, y, none::out<dtype>, where);
    }
    template <typename T, typename U, typename dtype = promote_t<T, U>>
    array<dtype> add(const array<T>& x, const array<U>& y, const where_t& where) {
        return add(x, y, none::out<dtype>, where);
    }

    template <typename T, typename dtype = bool_t>
    std::conditional_t<is_bool_t<dtype>, MaskedArray, array<dtype>>
    all(const array<T>& a, const int8_t axis = none::axis, out_t<dtype> out = none::out<dtype>,
        const bool keepdims = false, const where_t& where = none::where) {
        return ufunc_axes(a, axis, out, keepdims, where, &math::all<T, dtype>);
    }
    template <typename dtype, typename T>
    std::conditional_t<is_bool_t<dtype>, MaskedArray, array<dtype>>
    all(const array<T>& a, const int8_t axis = none::axis, const bool keepdims = false,
        const where_t& where = none::where) {
        return all(a, axis, none::out<dtype>, keepdims, where);
    }

    template <typename T, typename U>
    requires(is_numeric_v<T> && is_numeric_v<U>)
    bool allclose(const array<T>& a, const array<U>& b, const float64_t rtol = 1e-5, const float64_t atol = 1e-8,
                  const bool equal_nan = false) {
        const size_t size = a.size();

        if (a.shape() != b.shape()) {
            throw std::invalid_argument("shape of both array should be same (broadcasting is not supported)");
        }
        buffer_t<bool_t> temp(size);
        size_t i = 0;
        const buffer_t<T> buf1 = buffer(a);
        const buffer_t<U> buf2 = buffer(b);

        while (i < size) {
            temp[i++] = math::allclose(buf1[i], buf2[i], rtol, atol, equal_nan);
        }
        return all(MaskedArray(std::move(temp), size));
    }
    template <typename T, typename U>
    requires(is_numeric_v<T> && is_numeric_v<U>)
    bool allclose(const array<T>& a, const array<U>& b, const bool equal_nan) {
        return allclose(a, b, 1e-5, 1e-8, equal_nan);
    }

    template <typename T>
    array<T> amax(const array<T>& a, const int8_t axis = none::axis, out_t<T> out = none::out<T>,
                  const bool keepdims = false, const T initial = none::initial<T>, const where_t& where = none::where) {
        // implementation of max()
        return array<T>();
    }

    template <typename T>
    array<T> amin(const array<T>& a, const int8_t axis = none::axis, out_t<T> out = none::out<T>,
                  const bool keepdims = false, const T initial = none::initial<T>, const where_t& where = none::where) {
        // implementation of min()
        return array<T>();
    }

    template <typename T>
    requires(is_numeric_v<T>)
    array<real_t<T>> angle(array<T>& z, const bool deg = false) {
        array<real_t<T>> res;

        if constexpr (is_complex_v<T>) {
            res = arctan2(z.imag(), z.real());
        } else {
            res = arctan2(array<real_t<T>>(0), z);
        }
        return deg ? rad2deg(res, out_t(res)) : res;
    }

    template <typename T, typename U, typename dtype = promote_t<T, U>>
    requires(is_real_v<T> && is_real_v<U>)
    array<dtype> arctan2(const array<T>& y, const array<U>& x, out_t<dtype> out = none::out<dtype>,
                         const where_t& where = none::where) {
        return ufunc_binary(y, x, out, where, &math::arctan2<T, U, dtype>);
    }
    template <typename dtype, typename T, typename U>
    requires(is_real_v<T> && is_real_v<U>)
    array<dtype> arctan2(const array<T>& y, const array<U>& x, const where_t& where = none::where) {
        return arctan2(y, x, none::out<dtype>, where);
    }
    template <typename T, typename U, typename dtype = promote_t<T, U>>
    requires(is_real_v<T> && is_real_v<U>)
    array<dtype> arctan2(const array<T>& y, const array<U>& x, const where_t& where) {
        return arctan2(y, x, none::out<dtype>, where);
    }

    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    array<dtype> rad2deg(const array<T>& x, out_t<dtype> out = none::out<dtype>, const where_t& where = none::where) {
        return ufunc_unary(x, out, where, &math::rad2deg<T, dtype>);
    }
    template <typename dtype, typename T>
    requires(is_real_v<T>)
    array<dtype> rad2deg(const array<T>& x, const where_t& where = none::where) {
        return rad2deg(x, none::out<dtype>, where);
    }
    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    array<dtype> rad2deg(const array<T>& x, const where_t& where) {
        return rad2deg(x, none::out<dtype>, where);
    }


    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    array<dtype> floor(const array<T>& arr, out_t<dtype> out = none::out<dtype>, const where_t& where = none::where) {
        return ufunc(arr, out, where, &math::floor<T, dtype>);
    }
    template <typename dtype, typename T>
    requires(is_real_v<T>)
    array<dtype> floor(const array<T>& arr, const where_t& where = none::where) {
        return floor(arr, none::out<dtype>, where);
    }
    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    array<dtype> floor(const array<T>& arr, const where_t& where) {
        return floor<dtype>(arr, none::out<dtype>, where);
    }

    template <typename T, typename dtype = real_t<T>>
    requires(is_numeric_v<T>)
    array<dtype> real(array<T>& arr, out_t<dtype> out = none::out<dtype>, const where_t& where = none::where) {
        if constexpr (is_complex_v<T>) {
            return ufunc(arr.real(), out, where, none::func<real_t<T>, dtype>);
        } else {
            return ufunc(arr.real(), out ? out : arr, where, none::func<real_t<T>, dtype>);
        }
    }
    template <typename dtype, typename T>
    requires(is_numeric_v<T>)
    array<dtype> real(array<T>& arr, const where_t& where = none::where) {
        return real(arr, none::out<dtype>, where);
    }
    template <typename T, typename dtype = real_t<T>>
    requires(is_numeric_v<T>)
    array<dtype> real(array<T>& arr, const where_t& where) {
        return real(arr, none::out<dtype>, where);
    }

    template <typename T, typename dtype = real_t<T>>
    requires(is_numeric_v<T>)
    array<dtype> imag(array<T>& arr, out_t<dtype> out = none::out<dtype>, const where_t& where = none::where) {
        if constexpr (is_complex_v<T>) {
            return ufunc(arr.imag(), out, where, none::func<real_t<T>, dtype>);
        } else {
            return ufunc(arr.imag(), out ? out : arr, where, none::func<real_t<T>, dtype>);
        }
    }
    template <typename dtype, typename T>
    requires(is_numeric_v<T>)
    array<dtype> imag(array<T>& arr, const where_t& where = none::where) {
        return imag(arr, none::out<dtype>, where);
    }
    template <typename T, typename dtype = T>
    requires(is_numeric_v<T>)
    array<dtype> imag(array<T>& arr, const where_t& where) {
        return imag(arr, none::out<dtype>, where);
    }
} // namespace numcpp
