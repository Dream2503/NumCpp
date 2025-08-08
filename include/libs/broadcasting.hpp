#pragma once
#include "types.hpp"

namespace numcpp {
    inline shape_t broadcast_shape(const shape_t& shape1, const shape_t& shape2) {
        if ((shape1.rows != shape2.rows && shape1.rows != 1 && shape2.rows != 1) ||
            (shape1.cols != shape2.cols && shape1.cols != 1 && shape2.cols != 1)) {
            throw std::invalid_argument("Cannot broadcast shapes");
        }
        return {std::max(shape1.rows, shape2.rows), std::max(shape1.cols, shape2.cols)};
    }

    inline size_t broadcast_index(const size_t idx, const size_t dim) noexcept { return dim == 1 ? 0 : idx; }

    inline index_t broadcast_index(const index_t& index, const shape_t& shape) {
        if (!index.is_scalar()) {
            throw std::invalid_argument("broadcast_index: expected a scalar index");
        }
        return {static_cast<ll_t>(broadcast_index(index.get_scalar_row(), shape.rows)),
                static_cast<ll_t>(broadcast_index(index.get_scalar_col(), shape.cols))};
    }

    template <typename L, typename R, typename Op, typename Custom = none_t<>>
    array<promote_t<L, R, Custom>> binary_opr_broadcast(const array<L>& lhs, const array<R>& rhs, Op opr,
                                                        Custom = none_t()) {
        using V = promote_t<L, R, Custom>;
        const shape_t lhs_shape = lhs.shape(), rhs_shape = rhs.shape();
        auto [res_row, res_col] = broadcast_shape(lhs_shape, rhs_shape);
        buffer_t<V> result;
        size_t idx = 0;

        if (res_row * res_col == 0) {
            return array<V>();
        }
        if constexpr (std::is_same_v<Custom, in_place_t>) {
            result = buffer(lhs);
            idx = offset(lhs);
        } else {
            if constexpr (std::is_same_v<V, dtype::bool_t>) {
                result = buffer_t<V>((res_row * res_col + 7) / 8);
            } else {
                result = buffer_t<V>(res_row * res_col);
            }
        }
        for (ll_t i = 0; i < res_row; i++) {
            const ll_t ai = broadcast_index(i, lhs_shape.rows), bi = broadcast_index(i, rhs_shape.rows);

            for (ll_t j = 0; j < res_col; j++) {
                const ll_t aj = broadcast_index(j, lhs_shape.cols), bj = broadcast_index(j, rhs_shape.cols);

                if constexpr (std::is_same_v<Custom, comparison_t>) {
                    dtype::bitref_t(result[idx / 8].value, idx % 8) =
                        opr(static_cast<L>(lhs[{ai, aj}]), static_cast<R>(rhs[{bi, bj}]));
                } else if constexpr (std::is_same_v<V, dtype::bool_t>) {
                    dtype::bitref_t(result[idx / 8].value, idx % 8) =
                        opr(dtype::bitref_t(lhs[{ai, aj}]), dtype::bitref_t(rhs[{bi, bj}]));
                } else {
                    result[idx] = opr(static_cast<V>(lhs[{ai, aj}]), static_cast<V>(rhs[{bi, bj}]));
                }
                idx++;
            }
        }
        return array<V>(std::move(result), {res_row, res_col});
    }

    template <typename L, typename R, typename Op, typename Custom = none_t<>>
    array<promote_t<L, R, Custom>> binary_opr_element_wise(const array<L>& lhs, const R& value, Op opr,
                                                           Custom = none_t()) {
        using V = promote_t<L, R, Custom>;
        auto [res_row, res_col] = lhs.shape();
        buffer_t<V> result;
        size_t idx = 0;

        if (res_row * res_col == 0) {
            return array<V>();
        }
        if constexpr (std::is_same_v<Custom, in_place_t>) {
            result = buffer(lhs);
            idx = offset(lhs);
        } else {
            if constexpr (std::is_same_v<V, dtype::bool_t>) {
                result = buffer_t<V>((lhs.size() + 7) / 8);
            } else {
                result = buffer_t<V>(lhs.size());
            }
        }
        for (ll_t i = 0; i < res_row; i++) {
            for (ll_t j = 0; j < res_col; j++) {
                if constexpr (std::is_same_v<Custom, comparison_t>) {
                    if constexpr (std::is_same_v<Custom, swap_t>) {
                        dtype::bitref_t(result[idx / 8].value, idx % 8) = opr(value, static_cast<L>(lhs[{i, j}]));
                    } else {
                        dtype::bitref_t(result[idx / 8].value, idx % 8) = opr(static_cast<L>(lhs[{i, j}]), value);
                    }
                } else if constexpr (std::is_same_v<V, dtype::bool_t>) {
                    if constexpr (std::is_same_v<Custom, swap_t>) {
                        dtype::bitref_t(result[idx / 8].value, idx % 8) = opr(value, dtype::bitref_t(lhs[{i, j}]));
                    } else {
                        dtype::bitref_t(result[idx / 8].value, idx % 8) = opr(dtype::bitref_t(lhs[{i, j}]), value);
                    }
                } else if constexpr (std::is_same_v<L, dtype::bool_t>) {
                    if constexpr (std::is_same_v<Custom, swap_t>) {
                        result[idx] = opr(value, static_cast<bool>(dtype::bitref_t(lhs[{i, j}])));
                    } else {
                        result[idx] = opr(static_cast<bool>(dtype::bitref_t(lhs[{i, j}])), value);
                    }
                } else {
                    if constexpr (std::is_same_v<Custom, swap_t>) {
                        result[idx] = opr(value, static_cast<V>(lhs[{i, j}]));
                    } else {
                        result[idx] = opr(static_cast<V>(lhs[{i, j}]), value);
                    }
                }
                idx++;
            }
        }
        return array<V>(std::move(result), {res_row, res_col});
    }

    template <typename V, typename Op>
    array<V> unary_opr_element_wise(const array<V>& lhs, Op opr) {
        auto [res_row, res_col] = lhs.shape();
        buffer_t<V> result;

        if (res_row * res_col == 0) {
            return array<V>();
        }
        if constexpr (std::is_same_v<V, dtype::bool_t>) {
            result = buffer_t<dtype::bool_t>((lhs.size() + 7) / 8);
        } else {
            result = buffer_t<V>(lhs.size());
        }
        for (ll_t i = 0; i < res_row; i++) {
            for (ll_t j = 0; j < res_col; j++) {
                const int idx = i * res_col + j;

                if constexpr (std::is_same_v<V, dtype::bool_t>) {
                    dtype::bitref_t(result[idx / 8].value, idx % 8) = opr(dtype::bitref_t(lhs[{i, j}]));
                } else {
                    result[idx] = opr(static_cast<V>(lhs[{i, j}]));
                }
            }
        }
        return array<V>(std::move(result), {res_row, res_col});
    }

    template <typename V>
    array<V> array<V>::operator[](const index_t& index) const {
        if (index.is_scalar()) {
            auto [i, j] = index.get_scalar();

            if (i < 0) {
                i += row;
            }
            if (j < 0) {
                j += col;
            }
            if (i < 0 || i >= row || j < 0 || j >= col) {
                throw std::out_of_range("Index out of bounds");
            }
            return array(buffer, {1, 1}, offset + i * row_stride + j * col_stride, row_stride, col_stride,
                         (base ? base : this), false, true, true);
        }
        if (index.is_scalar_row() && index.is_slice_col()) {
            ll_t i = index.get_scalar_row();
            slice_t cols = index.get_slice_col();

            if (i < 0) {
                i += row;
            }
            if (i < 0 || i >= row) {
                throw std::out_of_range("Row index out of bounds");
            }
            cols.resolve(col);
            return array(buffer, {1, cols.size(col)}, offset + i * row_stride + cols.start * col_stride, row_stride,
                         col_stride * cols.step, base ? base : this, false, false, true);
        }
        if (index.is_slice_row() && index.is_scalar_col()) {
            slice_t rows = index.get_slice_row();
            ll_t j = index.get_scalar_col();

            if (j < 0) {
                j += col;
            }
            if (j < 0 || j >= col) {
                throw std::out_of_range("Column index out of bounds");
            }
            rows.resolve(row);
            return array(buffer, {rows.size(row), 1}, offset + rows.start * row_stride + j * col_stride,
                         row_stride * rows.step, col_stride, base ? base : this, false, false, true);
        }
        {
            auto [rows, cols] = index.get_slice();
            rows.resolve(row);
            cols.resolve(col);
            return array(buffer, {rows.size(row), cols.size(col)},
                         offset + rows.start * row_stride + cols.start * col_stride, row_stride * rows.step,
                         col_stride * cols.step, base ? base : this, true, false, true);
        }
    }
} // namespace numcpp
