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

    template <typename T, typename dtype = bool>
    array<dtype> all(const array<T>& a, const int8_t axis = none::axis, out_t<dtype> out = none::out<dtype>,
                     const bool keepdims = false, const where_t& where = none::where) {
        return ufunc_axes_unary(a, axis, out, keepdims, &math::all<T, dtype>, where);
    }
    template <typename dtype, typename T>
    array<dtype> all(const array<T>& a, const int8_t axis = none::axis, const bool keepdims = false,
                     const where_t& where = none::where) {
        return all(a, axis, none::out<dtype>, keepdims, where);
    }

    template <typename T, typename U>
    requires(is_numeric_v<T> && is_numeric_v<U>)
    bool allclose(const array<T>& a, const array<U>& b, const float64_t rtol = 1e-5, const float64_t atol = 1e-8,
                  const bool equal_nan = false) {
        return ufunc_axes_binary(a, b, none::axis, none::out<bool>, false, &math::allclose<T, U>, rtol, atol,
                                 equal_nan);
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
        array<real_t<T>> res = arctan2(z.imag(), z.real());
        return deg ? rad2deg(res, out_t(res)) : res;
    }

    template <typename T, typename dtype = bool>
    array<dtype> any(const array<T>& a, const int8_t axis = none::axis, out_t<dtype> out = none::out<dtype>,
                     const bool keepdims = false, const where_t& where = none::where) {
        return ufunc_axes_unary(a, axis, out, keepdims, &math::any<T, dtype>, where);
    }
    template <typename dtype, typename T>
    array<dtype> any(const array<T>& a, const int8_t axis = none::axis, const bool keepdims = false,
                     const where_t& where = none::where) {
        return any(a, axis, none::out<dtype>, keepdims, where);
    }

    template <typename T, typename U, typename dtype = promote_t<T, U>>
    array<dtype> append(const array<T>& arr, const array<U>& values, const int8_t axis = none::axis) {
        const shape_t arr_shape = arr.shape(), values_shape = values.shape();
        shape_t res_shape;

        if (axis == none::axis) {
            res_shape = shape_t(arr_shape.size() + values_shape.size());
        } else if (axis == 0 || axis == -2) {
            if (values_shape.cols != arr_shape.cols) {
                throw std::invalid_argument("dimension of values mis-match with arr");
            }
            res_shape = shape_t(arr_shape.rows + values_shape.rows, arr_shape.cols);
        } else if (axis == 1 || axis == -1) {
            if (values_shape.rows != arr_shape.rows) {
                throw std::invalid_argument("dimension of values mis-match with arr");
            }
            res_shape = shape_t(arr_shape.rows, arr_shape.cols + values_shape.cols);
        } else {
            throw std::invalid_argument("other axes are not suppoerted");
        }
        buffer_t<dtype> buf(res_shape.size());
        std::copy(arr.begin(), arr.end(), buf.data());
        std::copy(values.begin(), values.end(), buf.data() + arr_shape.size());
        return array(std::move(buf), res_shape);
    }

    template <typename T>
    array<T> arange(const range_t<T>& range) {
        return array(range.evaluate(), range.size());
    }

    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    array<dtype> arccos(const array<T>& x, out_t<dtype> out = none::out<dtype>, const where_t& where = none::where) {
        return ufunc_unary(x, out, where, &math::arccos<T, dtype>);
    }
    template <typename dtype, typename T>
    requires(is_real_v<T>)
    array<dtype> arccos(const array<T>& x, const where_t& where = none::where) {
        return arccos(x, none::out<dtype>, where);
    }
    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    array<dtype> arccos(const array<T>& x, const where_t& where) {
        return arccos(x, none::out<dtype>, where);
    }

    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    array<dtype> arccosh(const array<T>& x, out_t<dtype> out = none::out<dtype>, const where_t& where = none::where) {
        return ufunc_unary(x, out, where, &math::arccosh<T, dtype>);
    }
    template <typename dtype, typename T>
    requires(is_real_v<T>)
    array<dtype> arccosh(const array<T>& x, const where_t& where = none::where) {
        return arccosh(x, none::out<dtype>, where);
    }
    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    array<dtype> arccosh(const array<T>& x, const where_t& where) {
        return arccosh(x, none::out<dtype>, where);
    }

    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    array<dtype> arcsin(const array<T>& x, out_t<dtype> out = none::out<dtype>, const where_t& where = none::where) {
        return ufunc_unary(x, out, where, &math::arcsin<T, dtype>);
    }
    template <typename dtype, typename T>
    requires(is_real_v<T>)
    array<dtype> arcsin(const array<T>& x, const where_t& where = none::where) {
        return arcsin(x, none::out<dtype>, where);
    }
    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    array<dtype> arcsin(const array<T>& x, const where_t& where) {
        return arcsin(x, none::out<dtype>, where);
    }

    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    array<dtype> arcsinh(const array<T>& x, out_t<dtype> out = none::out<dtype>, const where_t& where = none::where) {
        return ufunc_unary(x, out, where, &math::arcsinh<T, dtype>);
    }
    template <typename dtype, typename T>
    requires(is_real_v<T>)
    array<dtype> arcsinh(const array<T>& x, const where_t& where = none::where) {
        return arcsinh(x, none::out<dtype>, where);
    }
    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    array<dtype> arcsinh(const array<T>& x, const where_t& where) {
        return arcsinh(x, none::out<dtype>, where);
    }

    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    array<dtype> arctan(const array<T>& x, out_t<dtype> out = none::out<dtype>, const where_t& where = none::where) {
        return ufunc_unary(x, out, where, &math::arctan<T, dtype>);
    }
    template <typename dtype, typename T>
    requires(is_real_v<T>)
    array<dtype> arctan(const array<T>& x, const where_t& where = none::where) {
        return arctan(x, none::out<dtype>, where);
    }
    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    array<dtype> arctan(const array<T>& x, const where_t& where) {
        return arctan(x, none::out<dtype>, where);
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
    array<dtype> arctanh(const array<T>& x, out_t<dtype> out = none::out<dtype>, const where_t& where = none::where) {
        return ufunc_unary(x, out, where, &math::arctanh<T, dtype>);
    }
    template <typename dtype, typename T>
    requires(is_real_v<T>)
    array<dtype> arctanh(const array<T>& x, const where_t& where = none::where) {
        return arctanh(x, none::out<dtype>, where);
    }
    template <typename T, typename dtype = T>
    requires(is_real_v<T>)
    array<dtype> arctanh(const array<T>& x, const where_t& where) {
        return arctanh(x, none::out<dtype>, where);
    }

    template <typename T>
    array<size_t> argmax(const array<T>& a, const int8_t axis = none::axis, out_t<size_t> out = none::out<size_t>,
                         const bool keepdims = false) {
        if (a.size() == 0) {
            throw std::invalid_argument("attempt to get argmax of an empty sequence");
        }
        return ufunc_axes_unary(a, axis, out, keepdims, &math::argmax<T>);
    }
    template <typename T>
    array<size_t> argmax(const array<T>& a, const int8_t axis, const bool keepdims) {
        return argmax(a, axis, none::out<size_t>, keepdims);
    }

    template <typename T>
    array<size_t> argmin(const array<T>& a, const int8_t axis = none::axis, out_t<size_t> out = none::out<size_t>,
                         const bool keepdims = false) {
        if (a.size() == 0) {
            throw std::invalid_argument("attempt to get argmin of an empty sequence");
        }
        return ufunc_axes_unary(a, axis, out, keepdims, &math::argmin<T>);
    }
    template <typename T>
    array<size_t> argmin(const array<T>& a, const int8_t axis, const bool keepdims) {
        return argmin(a, axis, none::out<size_t>, keepdims);
    }

    template <typename T>
    array<size_t> argpartition(const array<T>& a, const size_t kth, const int8_t axis = 1) {
        auto [row, col] = a.shape();
        array res(buffer_t<size_t>(row * col), {row, col});

        if (axis == none::axis) {
            if (kth >= row * col) {
                throw std::invalid_argument("out of bounce");
            }
            res = res.reshape({1, row * col});
            math::argpartition(res, a.reshape({1, row * col}), kth);
        } else if (axis == 0 || axis == -2) {
            if (kth >= row) {
                throw std::invalid_argument("out of bounce");
            }
            for (ll_t i = 0; i < col; i++) {
                math::argpartition(res[{slice_t(), i}], a[{slice_t(), i}], kth);
            }
        } else if (axis == 1 || axis == -1) {
            if (kth >= col) {
                throw std::invalid_argument("out of bounce");
            }
            for (ll_t i = 0; i < row; i++) {
                math::argpartition(res[i], a[i], kth);
            }
        } else {
            throw std::invalid_argument("other axes are not suppoerted");
        }
        return res;
    }

    template <typename T>
    array<size_t> argsort(const array<T>& a, const int8_t axis = 1, const std::string& kind = "quicksort",
                          const bool stable = false) {
        auto [row, col] = a.shape();
        array res(buffer_t<size_t>(row * col), {row, col});

        if (axis == none::axis) {
            res = res.reshape({1, row * col});
            math::argsort(res, a.reshape({1, row * col}), kind, stable);
        } else if (axis == 0 || axis == -2) {
            for (ll_t i = 0; i < col; i++) {
                math::argsort(res[{slice_t(), i}], a[{slice_t(), i}], kind, stable);
            }
        } else if (axis == 1 || axis == -1) {
            for (ll_t i = 0; i < row; i++) {
                math::argsort(res[i], a[i], kind, stable);
            }
        } else {
            throw std::invalid_argument("other axes are not suppoerted");
        }
        return res;
    }

    template <typename T>
    array<size_t> argwhere(const array<T>& a) {
        auto [row, col] = a.shape();
        const size_t size = a.size() - std::count(a.begin(), a.end(), T());
        size_t k = 0;
        buffer_t<size_t> res;

        if (is_matrix(a)) {
            res = buffer_t<size_t>(size * 2);

            for (ll_t i = 0; i < row; i++) {
                for (ll_t j = 0; j < col; j++) {
                    if (a[{i, j}]) {
                        res[k++] = i;
                        res[k++] = j;
                    }
                }
            }
            return array(std::move(res), {size, 2});
        }
        res = buffer_t<size_t>(size);

        for (ll_t i = 0; i < col; i++) {
            if (a[i]) {
                res[k++] = i;
            }
        }
        return array(std::move(res), size);
    }

    template <typename T, typename dtype = T>
    array<dtype> around(const array<T>& a, const int8_t decimals = 0, out_t<dtype> = none::out<dtype>) {
        // implementation of round()
        return array<dtype>();
    }

    template <typename T>
    std::string array2string(const array<T>& a, const size_t max_line_width = 75, const size_t precision = 8,
                             const bool suppress_smail = false, const std::string& seperator = " ",
                             const std::string& prefix = "", const size_t threshold = 1000, const size_t edgeitems = 3,
                             const int8_t sign = '-', const std::string& floatmode = "maxprec",
                             const std::string suffix = "") {
        const printoptions temp = _format_options;
        _format_options = {suppress_smail, sign,  edgeitems, max_line_width, precision,
                           threshold,      "inf", floatmode, "nan",          seperator};
        std::stringstream ss;
        ss << prefix << a << suffix;
        _format_options = temp;
        return ss.str();
    }

    template <typename T, typename U>
    bool array_equal(const array<T>& a1, const array<U>& a2, bool equal_nan = false) {
        if (a1.shape() != a2.shape()) {
            return false;
        }
        return ufunc_axes_binary(a1, a2, none::axis, none::out<bool>, false, &math::equal<T, U>, equal_nan);
    }

    template <typename T, typename U>
    bool array_equiv(const array<T>& a1, const array<U>& a2) {
        if (a1.shape() == a2.shape()) {
            return array_equal(a1, a2);
        }
        return all(ufunc_binary(a1, a2, none::out<bool>, none::where, std::equal_to()));
    }

    template <typename T>
    std::string array_repr(const array<T>& arr, const size_t max_line_width = 75, const size_t precision = 8,
                           const bool suppress_smail = false) {
        return array2string(arr, max_line_width, precision, suppress_smail);
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
