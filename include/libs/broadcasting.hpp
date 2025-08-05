#pragma once
#include "traits.hpp"

namespace numcpp {
    inline shape_t broadcast_shape(const shape_t& shape1, const shape_t& shape2) {
        if ((shape1.rows != shape2.rows && shape1.rows != 1 && shape2.rows != 1) ||
            (shape1.cols != shape2.cols && shape1.cols != 1 && shape2.cols != 1)) {
            throw std::invalid_argument("Cannot broadcast shapes");
        }
        return {std::max(shape1.rows, shape2.rows), std::max(shape1.cols, shape2.cols)};
    }

    inline size_t broadcast_index(const size_t idx, const size_t dim) { return dim == 1 ? 0 : idx; }

    inline index_t broadcast_index(const index_t& index, const shape_t& shape) {
        if (!index.is_scalar()) {
            throw std::invalid_argument("broadcast_index: expected a scalar index");
        }
        return {static_cast<ll_t>(broadcast_index(index.get_scalar_row(), shape.rows)),
                static_cast<ll_t>(broadcast_index(index.get_scalar_col(), shape.cols))};
    }

    template <typename L, typename R, typename Op>
    array<promote_t<L, R>> binary_opr_broadcast(const array<L>& lhs, const array<R>& rhs, Op opr) {
        const shape_t lhs_shape = lhs.shape(), rhs_shape = rhs.shape();
        auto [res_row, res_col] = broadcast_shape(lhs_shape, rhs_shape);
        using V = promote_t<L, R>;
        buffer_t<V> result(res_row * res_col);

        for (size_t i = 0; i < res_row; i++) {
            const size_t ai = broadcast_index(i, lhs_shape.rows), bi = broadcast_index(i, rhs_shape.rows);

            for (size_t j = 0; j < res_col; j++) {
                const size_t aj = broadcast_index(j, lhs_shape.cols), bj = broadcast_index(j, rhs_shape.cols);
                result[i * res_col + j] = opr(lhs.at(ai, aj), rhs.at(bi, bj));
            }
        }
        return array<V>(std::move(result), {res_row, res_col});
    }
} // namespace numcpp
