#pragma once
#include "ufunc.hpp"

namespace numcpp {
    template <typename T, typename dtype = T>
    requires(is_numeric_v<T>)
    array<dtype> absolute(const array<T>& arr, out_t<dtype> out = none::out<dtype>,
                          const where_t& where = none::where) {
        return ufunc_unary(arr, out, where, &math::abs<T, dtype>);
    }

    template <typename dtype, typename T>
    requires(is_numeric_v<T>)
    array<dtype> absolute(const array<T>& arr, const where_t& where = none::where) {
        return absolute(arr, none::out<dtype>, where);
    }

    template <typename T, typename dtype = T>
    requires(is_numeric_v<T>)
    array<dtype> absolute(const array<T>& arr, const where_t& where) {
        return absolute(arr, none::out<dtype>, where);
    }

    template <typename T, typename dtype = T>
    requires(is_numeric_v<T>)
    array<dtype> abs(const array<T>& arr, out_t<dtype> out = none::out<dtype>, const where_t& where = none::where) {
        return absolute(arr, out, where);
    }

    template <typename dtype, typename T>
    requires(is_numeric_v<T>)
    array<dtype> abs(const array<T>& arr, const where_t& where = none::where) {
        return absolute(arr, where);
    }

    template <typename T, typename dtype = T>
    requires(is_numeric_v<T>)
    array<dtype> abs(const array<T>& arr, const where_t& where) {
        return absolute(arr, where);
    }

    template <typename T, typename U, typename dtype = promote_t<T, U>>
    requires(is_numeric_v<T>)
    array<dtype> add(const array<T>& arr1, const array<U>& arr2, out_t<dtype> out = none::out<dtype>,
                     const where_t& where = none::where) {
        return ufunc_binary(arr1, arr2, out, where, std::plus());
    }

    template <typename dtype, typename T, typename U>
    requires(is_numeric_v<T>)
    array<dtype> add(const array<T>& arr1, const array<U>& arr2, const where_t& where = none::where) {
        return add(arr1, arr2, none::out<dtype>, where);
    }

    template <typename T, typename U, typename dtype = promote_t<T, U>>
    requires(is_numeric_v<T>)
    array<dtype> add(const array<T>& arr1, const array<U>& arr2, const where_t& where) {
        return add(arr1, arr2, none::out<dtype>, where);
    }

    template <typename T, typename dtype = dtypes::bool_t>
    std::conditional_t<std::is_same_v<dtype, dtypes::bool_t>, MaskedArray, array<dtype>>
    all(const array<T>& arr, const int axis = none::axis, out_t<dtype> out = none::out<dtype>,
        const bool keepdims = false, const where_t& where = none::where) {
        return ufunc_axes(arr, axis, out, keepdims, where, &math::all<T, dtype>);
    }

    template <typename dtype, typename T>
    std::conditional_t<std::is_same_v<dtype, dtypes::bool_t>, MaskedArray, array<dtype>>
    all(const array<T>& arr, const int axis = none::axis, const bool keepdims = false,
        const where_t& where = none::where) {
        return all(arr, axis, none::out<dtype>, keepdims, where);
    }


    // template <typename T, typename dtype = T>
    // requires(is_real_v<T>)
    // array<dtype> floor(const array<T>& arr, out_t<dtype> out = none::out<dtype>,
    //                    const where_t& where = none::where)  {
    //     return ufunc(arr, out, where, &math::floor<T, dtype>);
    // }
    //
    // template <typename T, typename dtype = T>
    // requires(is_real_v<T>)
    // array<dtype> floor(const array<T>& arr, const where_t& where)  {
    //     return floor<dtype>(arr, none::out<dtype>, where);
    // }
    //
    // template <typename T, typename dtype = real_t<T>>
    // requires(is_numeric_v<T>)
    // array<dtype> real(array<T>& arr, out_t<dtype> out = none::out<dtype>, const where_t& where = none::where)
    //  {
    //     if constexpr (is_complex_v<T>) {
    //         return ufunc(arr.real(), out, where, none::func<real_t<T>, dtype>);
    //     } else {
    //         return ufunc(arr.real(), out ? out : arr, where, none::func<real_t<T>, dtype>);
    //     }
    // }
    //
    // template <typename T, typename dtype = real_t<T>>
    // requires(is_numeric_v<T>)
    // array<dtype> real(array<T>& arr, const where_t& where)  {
    //     return real(arr, none::out<dtype>, where);
    // }
    //
    // template <typename T, typename dtype = real_t<T>>
    // requires(is_numeric_v<T>)
    // array<dtype> imag(array<T>& arr, out_t<dtype> out = none::out<dtype>, const where_t& where = none::where)
    //  {
    //     if constexpr (is_complex_v<T>) {
    //         return ufunc(arr.imag(), out, where, none::func<real_t<T>, dtype>);
    //     } else {
    //         return ufunc(arr.imag(), out ? out : arr, where, none::func<real_t<T>, dtype>);
    //     }
    // }
    //
    // template <typename T, typename dtype = T>
    // requires(is_numeric_v<T>)
    // array<dtype> imag(array<T>& arr, const where_t& where)  {
    //     return imag(arr, none::out<dtype>, where);
    // }
} // namespace numcpp
