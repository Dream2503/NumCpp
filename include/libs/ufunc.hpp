#pragma once

namespace numcpp {
    template <typename T, typename dtype, typename Func>
    array<dtype> ufunc_unary(const array<T>& arr, out_t<dtype> out, const where_t& where, Func func) {
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
                result[idx++] = !where || where->mask_at(i, j) ? func(static_cast<T>(arr[{i, j}])) : dtype(0);
            }
        }
        return out ? *out.ptr : array<dtype>(std::move(result), arr_shape);
    }

    template <typename L, typename R, typename dtype, typename Func>
    array<dtype> ufunc_binary(const array<L>& lhs, const array<R>& rhs, out_t<dtype> out, const where_t& where,
                              Func func) {
        const shape_t lhs_shape = lhs.shape(), rhs_shape = rhs.shape();
        const shape_t res_shape = broadcast_shape(lhs_shape, rhs_shape);
        const shape_t where_shape = where ? where->shape() : none::shape;
        auto [rows, cols] = where ? broadcast_shape(res_shape, where_shape) : res_shape;
        size_t idx = 0;

        if (out && out->shape() != res_shape) {
            throw std::invalid_argument("Shape mis-match with out and expected out-put");
        }
        if (where && res_shape != broadcast_shape(res_shape, where_shape)) {
            throw std::invalid_argument("Cannot broadcast where shape to array shape");
        }
        if (rows * cols == 0) {
            return array<dtype>();
        }
        buffer_t<dtype> result = out ? buffer(*out.ptr) : buffer_t<dtype>(rows * cols);

        for (ll_t i = 0; i < rows; i++) {
            const ll_t ai = broadcast_index(i, lhs_shape.rows), bi = broadcast_index(i, rhs_shape.rows);

            for (ll_t j = 0; j < cols; j++) {
                const ll_t aj = broadcast_index(j, lhs_shape.cols), bj = broadcast_index(j, rhs_shape.cols);

                result[idx++] = !where || where->mask_at(i, j)
                    ? func(static_cast<L>(lhs[{ai, aj}]), static_cast<R>(rhs[{bi, bj}]))
                    : dtype(0);
            }
        }
        return out ? *out.ptr : array<dtype>(std::move(result), {rows, cols});
    }

    template <typename T, typename dtype, typename Func>
    array<dtype> ufunc_axes(const array<T>& arr, const int axis, out_t<dtype> out, const bool keepdims,
                            const where_t& where, Func func) {
        auto [row, col] = arr.shape();
        buffer_t<dtype> buf;

        if (axis == none::axis) {
            if (out && out->shape() != shape_t(1)) {
                throw std::invalid_argument("Shape mis-match with out and expected out-put");
            }
            buf = out ? buffer(*out.ptr) : buffer_t<dtype>(1);
            buf[0] = func(arr, where);
            array<dtype> res(std::move(buf), 1);
            return keepdims ? res[slice_t()] : res[{0, 0}];
        }
        if (axis == 0) {
            if (out && out->shape() != shape_t(col)) {
                throw std::invalid_argument("Shape mis-match with out and expected out-put");
            }
            buf = out ? buffer(*out.ptr) : buffer_t<dtype>(col);

            for (ll_t i = 0; i < col; i++) {
                buf[i] = func(arr[{slice_t(), i}], where);
            }
            array<dtype> res(std::move(buf), col);
            return keepdims ? res[slice_t()] : res;
        }
        if (axis == 1) {
            if (out && out->shape() != shape_t(row, 1)) {
                throw std::invalid_argument("Shape mis-match with out and expected out-put");
            }
            buf = out ? buffer(*out.ptr) : buffer_t<dtype>(row);

            for (size_t i = 0; i < row; i++) {
                buf[i] = func(arr[i], where);
            }
            array<dtype> res(std::move(buf), {row, 1});
            return keepdims ? res[slice_t()] : res;
        }
        throw std::invalid_argument("other axes are not suppoerted");
    }
} // namespace numcpp
