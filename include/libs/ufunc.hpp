#pragma once

namespace numcpp {
    template <typename T, typename dtype, typename Func, typename... Args>
    array<dtype> ufunc_unary(const array<T>& arr, out_t<dtype> out, const where_t& where, Func func, Args&&... args) {
        const shape_t arr_shape = arr.shape(), where_shape = where ? where->shape() : none::shape;

        if (out && out->shape() != arr_shape) {
            throw std::invalid_argument("Shape mis-match with out and expected out-put");
        }
        if (where && arr_shape != broadcast_shape(arr_shape, where_shape)) {
            throw std::invalid_argument("Cannot broadcast where shape to array shape");
        }
        buffer_t<dtype> result = out ? buffer(*out.ptr) : buffer_t<dtype>(arr_shape.size());
        size_t idx = 0;

        for (ll_t i = 0; i < arr_shape.rows; i++) {
            for (ll_t j = 0; j < arr_shape.cols; j++) {
                result[idx++] = !where || (*where)[broadcast_index({i, j}, where_shape)]
                    ? func(static_cast<T>(arr[{i, j}]), std::forward<Args>(args)...)
                    : dtype(0);
            }
        }
        return out ? *out.ptr : array<dtype>(std::move(result), arr_shape);
    }

    template <typename L, typename R, typename dtype, typename Func, typename... Args>
    array<dtype> ufunc_binary(const array<L>& lhs, const array<R>& rhs, out_t<dtype> out, const where_t& where, Func func, Args&&... args) {
        const shape_t lhs_shape = lhs.shape(), rhs_shape = rhs.shape(), where_shape = where ? where->shape() : none::shape;
        shape_t res_shape = broadcast_shape(lhs_shape, rhs_shape);
        size_t idx = 0;

        if (out && out->shape() != res_shape) {
            throw std::invalid_argument("Shape mis-match with out and expected out-put");
        }
        if (where && res_shape != broadcast_shape(res_shape, where_shape)) {
            throw std::invalid_argument("Cannot broadcast where shape to array shape");
        }
        res_shape = where ? broadcast_shape(res_shape, where_shape) : res_shape;

        if (res_shape.size() == 0) {
            return array<dtype>();
        }
        buffer_t<dtype> result = out ? buffer(*out.ptr) : buffer_t<dtype>(res_shape.size());

        for (ll_t i = 0; i < res_shape.rows; i++) {
            for (ll_t j = 0; j < res_shape.cols; j++) {
                result[idx++] = !where || (*where)[broadcast_index({i, j}, where_shape)]
                    ? func(static_cast<L>(lhs[broadcast_index({i, j}, lhs_shape)]), static_cast<R>(rhs[broadcast_index({i, j}, rhs_shape)]),
                           std::forward<Args>(args)...)
                    : dtype(0);
            }
        }
        return out ? *out.ptr : array<dtype>(std::move(result), res_shape);
    }

    template <typename T, typename dtype, typename Func, typename... Args>
    array<dtype> ufunc_axes_unary(const array<T>& arr, const int8_t axis, out_t<dtype> out, const bool keepdims, Func func, Args&&... args) {
        auto [row, col] = arr.shape();
        buffer_t<dtype> buf;
        array<dtype> res;

        if (axis == none::axis) {
            if (out && out->shape() != shape_t(1)) {
                throw std::invalid_argument("Shape mis-match with out and expected out-put");
            }
            buf = out ? buffer(*out.ptr) : buffer_t<dtype>(1);
            buf[0] = func(arr, std::forward<Args>(args)...);
            res = array<dtype>(std::move(buf), 1);
        } else if (axis == 0 || axis == -2) {
            if (out && out->shape() != shape_t(col)) {
                throw std::invalid_argument("Shape mis-match with out and expected out-put");
            }
            buf = out ? buffer(*out.ptr) : buffer_t<dtype>(col);

            for (ll_t i = 0; i < col; i++) {
                buf[i] = func(arr[{slice_t(), i}], std::forward<Args>(args)...);
            }
            res = array<dtype>(std::move(buf), col);
        } else if (axis == 1 || axis == -1) {
            if (out && out->shape() != shape_t(row, 1)) {
                throw std::invalid_argument("Shape mis-match with out and expected out-put");
            }
            buf = out ? buffer(*out.ptr) : buffer_t<dtype>(row);

            for (size_t i = 0; i < row; i++) {
                buf[i] = func(arr[i], std::forward<Args>(args)...);
            }
            res = array<dtype>(std::move(buf), {row, 1});

        } else {
            throw std::invalid_argument("other axes are not suppoerted");
        }
        return keepdims ? res[slice_t()] : res;
    }

    template <typename L, typename R, typename dtype, typename Func, typename... Args>
    array<dtype> ufunc_axes_binary(const array<L>& lhs, const array<R>& rhs, const int8_t axis, out_t<dtype> out, const bool keepdims, Func func,
                                   Args&&... args) {
        if (lhs.shape() != rhs.shape()) {
            throw std::invalid_argument("currently no broadcasting allowed");
        }
        auto [row, col] = lhs.shape();
        buffer_t<dtype> buf;
        array<dtype> res;

        if (axis == none::axis) {
            if (out && out->shape() != shape_t(1)) {
                throw std::invalid_argument("Shape mis-match with out and expected out-put");
            }
            buf = out ? buffer(*out.ptr) : buffer_t<dtype>(1);
            buf[0] = func(lhs, rhs, std::forward<Args>(args)...);
            res = array<dtype>(std::move(buf), 1);
        } else if (axis == 0 || axis == -2) {
            if (out && out->shape() != shape_t(col)) {
                throw std::invalid_argument("Shape mis-match with out and expected out-put");
            }
            buf = out ? buffer(*out.ptr) : buffer_t<dtype>(col);

            for (ll_t i = 0; i < col; i++) {
                buf[i] = func(lhs[{slice_t(), i}], rhs[{slice_t(), i}], std::forward<Args>(args)...);
            }
            res = array<dtype>(std::move(buf), col);
        } else if (axis == 1 || axis == -1) {
            if (out && out->shape() != shape_t(row, 1)) {
                throw std::invalid_argument("Shape mis-match with out and expected out-put");
            }
            buf = out ? buffer(*out.ptr) : buffer_t<dtype>(row);

            for (size_t i = 0; i < row; i++) {
                buf[i] = func(lhs[i], rhs[i], std::forward<Args>(args)...);
            }
            res = array<dtype>(std::move(buf), {row, 1});
        } else {
            throw std::invalid_argument("other axes are not suppoerted");
        }
        return keepdims ? res[slice_t()] : res;
    }
} // namespace numcpp
