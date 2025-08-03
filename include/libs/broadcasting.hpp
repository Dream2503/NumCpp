#pragma once
#include "traits.hpp"

namespace numcpp {
    inline std::pair<size_t, size_t> broadcast_shape(const std::pair<size_t, size_t>& shape1,
                                                     const std::pair<size_t, size_t>& shape2) {
        if ((shape1.first != shape2.first && shape1.first != 1 && shape2.first != 1) ||
            (shape1.second != shape2.second && shape1.second != 1 && shape2.second != 1)) {
            throw std::invalid_argument("Cannot broadcast shapes");
        }
        return {std::max(shape1.first, shape2.first), std::max(shape1.second, shape2.second)};
    }

    inline size_t broadcast_index(const size_t i, const size_t dim) { return dim == 1 ? 0 : i; }

    inline std::pair<size_t, size_t> broadcast_mask_at(const size_t i, const size_t j,
                                                       const std::pair<const size_t, const size_t>& shape) {
        return {broadcast_index(i, shape.first), broadcast_index(j, shape.second)};
    }

    template <typename L, typename R, typename Op>
    array<promote_t<L, R>> binary_opr_broadcast(const array<L>& lhs, const array<R>& rhs, Op opr) {
        auto [lhs_row, lhs_col] = lhs.shape();
        auto [rhs_row, rhs_col] = rhs.shape();
        auto [res_row, res_col] = broadcast_shape(lhs.shape(), rhs.shape());
        using V = promote_t<L, R>;
        std::vector<V> result;
        result.reserve(res_row * res_col);

        for (size_t i = 0; i < res_row; i++) {
            size_t ai = broadcast_index(i, lhs_row), bi = broadcast_index(i, rhs_row);

            for (size_t j = 0; j < res_col; j++) {
                size_t aj = broadcast_index(j, lhs_col), bj = broadcast_index(j, rhs_col);
                result.push_back(opr(lhs.at(ai, aj), rhs.at(bi, bj)));
            }
        }
        return array<V>(std::move(result), res_row, res_col);
    }
} // namespace numcpp
