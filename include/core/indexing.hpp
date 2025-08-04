#pragma once
#include "../libs/traits.hpp"

namespace numcpp {
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
            return array(data, {1, 1}, offset + i * row_stride + j * col_stride, row_stride, col_stride,
                         _base ? _base : this, false, true);
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
            return array(data, {1, cols.size(col)}, offset + i * row_stride + cols.start * col_stride, row_stride,
                         col_stride * cols.step, _base ? _base : this, false, false);
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
            return array(data, {rows.size(row), 1}, offset + rows.start * row_stride + j * col_stride,
                         row_stride * rows.step, col_stride, _base ? _base : this, false, false);
        }
        auto [rows, cols] = index.get_slice();
        rows.resolve(row);
        cols.resolve(col);
        return array(data, {rows.size(row), cols.size(col)}, offset + rows.start * row_stride + cols.start * col_stride,
                     row_stride * rows.step, col_stride * cols.step, _base ? _base : this, true, false);
    }
} // namespace numcpp
