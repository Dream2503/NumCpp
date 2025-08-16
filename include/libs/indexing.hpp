#pragma once

namespace numcpp {
    constexpr bool can_broadcast_shape(const shape_t& shape1, const shape_t& shape2) {
        return (shape1.rows == shape2.rows || shape1.rows == 1 || shape2.rows == 1) &&
            (shape1.cols == shape2.cols || shape1.cols == 1 || shape2.cols == 1);
    }

    inline shape_t broadcast_shape(const shape_t& shape1, const shape_t& shape2) {
        if (!can_broadcast_shape(shape1, shape2)) {
            throw std::invalid_argument("Cannot broadcast shapes");
        }
        return {std::max(shape1.rows, shape2.rows), std::max(shape1.cols, shape2.cols)};
    }

    constexpr ll_t broadcast_index(const ll_t idx, const size_t dim) noexcept { return dim == 1 ? 0 : idx; }

    inline index_t broadcast_index(const index_t& index, const shape_t& shape) {
        if (!index.is_scalar()) {
            throw std::invalid_argument("broadcast_index: expected scalar indies");
        }
        return {(broadcast_index(index.get_scalar_row(), shape.rows)), (broadcast_index(index.get_scalar_col(), shape.cols))};
    }

    template <typename L, typename R, typename Op, typename Operation = none_t<>>
    array<promote_t<L, R, Operation>> binary_opr_broadcast(const array<L>& lhs, const array<R>& rhs, Op opr, Operation = none_t()) {
        using T = promote_t<L, R, Operation>;
        const shape_t lhs_shape = lhs.shape(), rhs_shape = rhs.shape();
        const shape_t res_shape = broadcast_shape(lhs_shape, rhs_shape);
        buffer_t<T> result;
        size_t idx = 0;

        if constexpr (std::is_same_v<Operation, operations::in_place_t>) {
            result = buffer(lhs);
            idx = offset(lhs);
        } else {
            result = buffer_t<T>(res_shape.size());
        }
        for (ll_t i = 0; i < res_shape.rows; i++) {
            for (ll_t j = 0; j < res_shape.cols; j++) {
                result[idx++] = opr(static_cast<T>(lhs[broadcast_index({i, j}, lhs_shape)]), static_cast<T>(rhs[broadcast_index({i, j}, rhs_shape)]));
            }
        }
        return array<T>(std::move(result), res_shape);
    }

    template <typename L, typename R, typename Op, typename Operation = none_t<>>
    array<promote_t<L, R, Operation>> binary_opr_element_wise(const array<L>& lhs, const R& value, Op opr, Operation = none_t()) {
        using T = promote_t<L, R, Operation>;
        const shape_t shape = lhs.shape();
        buffer_t<T> result;
        size_t idx = 0;

        if constexpr (std::is_same_v<Operation, operations::in_place_t>) {
            result = buffer(lhs);
            idx = offset(lhs);
        } else {
            result = buffer_t<T>(lhs.size());
        }
        for (ll_t i = 0; i < shape.rows; i++) {
            for (ll_t j = 0; j < shape.cols; j++) {
                if constexpr (std::is_same_v<Operation, operations::swap_t>) {
                    result[idx] = opr(value, static_cast<T>(lhs[{i, j}]));
                } else {
                    result[idx] = opr(static_cast<T>(lhs[{i, j}]), value);
                }
                idx++;
            }
        }
        return array<T>(std::move(result), shape);
    }

    template <typename T, typename Op>
    array<T> unary_opr_element_wise(const array<T>& lhs, Op opr) {
        const shape_t shape = lhs.shape();
        buffer_t<T> result;
        size_t idx = 0;
        result = buffer_t<T>(lhs.size());

        for (ll_t i = 0; i < shape.rows; i++) {
            for (ll_t j = 0; j < shape.cols; j++) {
                result[idx++] = opr(static_cast<T>(lhs[{i, j}]));
            }
        }
        return array<T>(std::move(result), shape);
    }

    template <typename T>
    array<T> array<T>::operator[](const index_t& index) const {
        if (index.is_scalar()) {
            auto [i, j] = index.get_scalars();

            if (i < 0) {
                i += row;
            }
            if (j < 0) {
                j += col;
            }
            if (i < 0 || i >= row || j < 0 || j >= col) {
                throw std::out_of_range("Index out of bounds");
            }
            return array(buffer, {1, 1}, offset + i * row_stride + j * col_stride, row_stride, col_stride, (base ? base : this), false, true, true);
        }
        if (index.is_scalar_row() && index.is_slice_col()) {
            ll_t i = index.get_scalar_row();
            slice_t cols = index.get_slice_col().resolve(col);

            if (!is_matrix) {
                return operator[]({0, i});
            }
            if (i < 0) {
                i += row;
            }
            if (i < 0 || i >= row) {
                throw std::out_of_range("Row index out of bounds");
            }
            return array(buffer, {1, cols.size(col)}, offset + i * row_stride + cols.start * col_stride, row_stride, col_stride * cols.step,
                         base ? base : this, false, false, true);
        }
        if (index.is_scalar_row() && index.is_array_col()) {
            ll_t x = index.get_scalar_row();
            const array<ll_t> y(*index.get_array_col());
            size_t idx = 0;
            auto [rows, cols] = y.shape();
            buffer_t<T> buf(y.size());

            for (ll_t i = 0; i < rows; i++) {
                for (ll_t j = 0; j < cols; j++) {
                    buf[idx++] = (*this)[{x, static_cast<ll_t>(y[{i, j}])}];
                }
            }
            return array(std::move(buf), {rows, cols});
        }
        if (index.is_slice_row() && index.is_scalar_col()) {
            slice_t rows = index.get_slice_row().resolve(row);
            ll_t j = index.get_scalar_col();

            if (j < 0) {
                j += col;
            }
            if (j < 0 || j >= col) {
                throw std::out_of_range("Column index out of bounds");
            }
            return array(buffer, {rows.size(row), 1}, offset + rows.start * row_stride + j * col_stride, row_stride * rows.step, col_stride,
                         base ? base : this, true, false, true);
        }
        if (index.is_slice()) {
            auto [rows, cols] = index.get_slices();
            rows.resolve(row);
            cols.resolve(col);
            return array(buffer, {rows.size(row), cols.size(col)}, offset + rows.start * row_stride + cols.start * col_stride, row_stride * rows.step,
                         col_stride * cols.step, base ? base : this, true, false, true);
        }
        if (index.is_slice_row() && index.is_array_col()) {
            slice_t x = index.get_slice_row().resolve(row);
            const array<ll_t> y(*index.get_array_col());
            auto [rows, cols] = y.shape();

            if (rows > 1) {
                throw std::invalid_argument("Higher dimension than 2D are not supported");
            }
            size_t idx = 0, size = x.size();
            buffer_t<T> buf(size * y.size());

            for (ll_t k = x.start; k < x.stop; k += x.step) {
                for (ll_t i = 0; i < rows; i++) {
                    for (ll_t j = 0; j < cols; j++) {
                        buf[idx++] = (*this)[{k, static_cast<ll_t>(y[{i, j}])}];
                    }
                }
            }
            return array(std::move(buf), {size, cols});
        }
        if (index.is_array_row() && index.is_scalar_col()) {
            const array<ll_t> x(*index.get_array_row());
            ll_t y = index.get_scalar_col();
            size_t idx = 0;
            auto [rows, cols] = x.shape();
            buffer_t<T> buf(x.size());

            for (ll_t i = 0; i < rows; i++) {
                for (ll_t j = 0; j < cols; j++) {
                    buf[idx++] = (*this)[{static_cast<ll_t>(x[{i, j}]), y}];
                }
            }
            return array(std::move(buf), {rows, cols});
        }
        if (index.is_array_row() && index.is_slice_col()) {
            const array<ll_t> x(*index.get_array_row());
            slice_t y = index.get_slice_col().resolve(row);
            auto [rows, cols] = x.shape();

            if (rows > 1) {
                throw std::invalid_argument("Higher dimension than 2D are not supported");
            }
            size_t idx = 0, size = y.size();
            buffer_t<T> buf(size * x.size());

            for (ll_t i = 0; i < rows; i++) {
                for (ll_t j = 0; j < cols; j++) {
                    const ll_t row = x[{i, j}];

                    for (ll_t k = y.start; k < y.stop; k += y.step) {
                        buf[idx++] = (*this)[{row, k}];
                    }
                }
            }
            return array(std::move(buf), {cols, size});
        }
        if (index.is_array_row() && index.is_array_col()) {
            const array<ll_t> row_array(*index.get_array_row()), col_array(*index.get_array_col());
            const shape_t row_shape = row_array.shape(), col_shape = col_array.shape();
            const shape_t res = broadcast_shape(row_shape, col_shape);
            size_t idx = 0;
            buffer_t<T> result(res.size());

            for (ll_t i = 0; i < res.rows; i++) {
                for (ll_t j = 0; j < res.cols; j++) {
                    result[idx++] = (*this)[{static_cast<ll_t>(row_array[broadcast_index({i, j}, row_shape)]),
                                             static_cast<ll_t>(col_array[broadcast_index({i, j}, col_shape)])}];
                }
            }
            return array(std::move(result), res);
        }
        throw std::invalid_argument("unexpected error");
    }
} // namespace numcpp
