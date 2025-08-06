#pragma once
#include <algorithm>
#include <complex>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <type_traits>
#include <variant>
#include "libs/detail.hpp"

#include "core/array.hpp"
#include "core/indexing.hpp"
#include "core/masked_array.hpp"
#include "core/operators.hpp"
#include "libs/broadcasting.hpp"
#include "libs/math.hpp"
#include "libs/none.hpp"
#include "libs/numeric.hpp"
#include "libs/utils.hpp"

/*temp
template <typename ReturnType>
    ReturnType _all_templated(const axis_t axis, const out_t out, const bool keep_dims, const where_t where) const {
        auto truthy = [&](size_t i, size_t j) { return static_cast<bool>((*this)[{i, j}]); };

        auto mask_at = [&](size_t i, size_t j) {
            if (!where.value)
                return true;
            const auto& w = where.value->get();
            return w[{broadcast_index(i, w.shape().first), broadcast_index(j, w.shape().second)}];
        };

        auto reduce_all = [&]() -> ReturnType {
            bool result = true;
            for (size_t i = 0; i < row && result; ++i) {
                for (size_t j = 0; j < col && result; ++j) {
                    if (mask_at(i, j) && !truthy(i, j))
                        result = false;
                }
            }

            if constexpr (std::is_same_v<ReturnType, bool>) {
                return result;
            } else if constexpr (std::is_same_v<ReturnType, array<bool>>) {
                return array<bool>(std::vector{result}, 1, 1);
            } else if constexpr (std::is_same_v<ReturnType, array<bool>&>) {
                if (!out.value)
                    throw std::invalid_argument("Missing output array reference");
                out.value->get()[{0, 0}] = result;
                return out.value->get();
            }
        };

        auto reduce_axis0 = [&]() -> ReturnType {
            auto reduced = std::shared_ptr<bool[]>(new bool[col]);
            for (size_t j = 0; j < col; ++j) {
                bool all_true = true;
                for (size_t i = 0; i < row; ++i) {
                    if (mask_at(i, j) && !truthy(i, j)) {
                        all_true = false;
                        break;
                    }
                }
                reduced[j] = all_true;
            }
            array<bool> result(std::move(reduced), keep_dims ? 1 : col, keep_dims ? col : 1);
            result.is_matrix = keep_dims;

            if constexpr (std::is_same_v<ReturnType, array<bool>>) {
                return result;
            } else if constexpr (std::is_same_v<ReturnType, array<bool>&>) {
                if (!out.value)
                    throw std::invalid_argument("Missing output array reference");
                out.value->get() = result;
                return out.value->get();
            } else {
                throw std::invalid_argument("reduce_axis0 only returns array");
            }
        };

        auto reduce_axis1 = [&]() -> ReturnType {
            auto reduced = std::shared_ptr<bool[]>(new bool[row]);
            for (size_t i = 0; i < row; ++i) {
                bool all_true = true;
                for (size_t j = 0; j < col; ++j) {
                    if (mask_at(i, j) && !truthy(i, j)) {
                        all_true = false;
                        break;
                    }
                }
                reduced[i] = all_true;
            }
            array<bool> result(std::move(reduced), keep_dims ? row : 1, keep_dims ? 1 : row);
            result.is_matrix = keep_dims;

            if constexpr (std::is_same_v<ReturnType, array<bool>>) {
                return result;
            } else if constexpr (std::is_same_v<ReturnType, array<bool>&>) {
                if (!out.value)
                    throw std::invalid_argument("Missing output array reference");
                out.value->get() = result;
                return out.value->get();
            } else {
                throw std::invalid_argument("reduce_axis1 only returns array");
            }
        };

        // Dispatch based on axis
        if (std::holds_alternative<none_t>(axis.value)) {
            return reduce_all();
        } else if (std::holds_alternative<int>(axis.value)) {
            const int ax = std::get<int>(axis.value);
            if (ax == 0)
                return reduce_axis0();
            if (ax == 1)
                return reduce_axis1();
            throw std::invalid_argument("Invalid axis value: must be 0 or 1");
        } else if (std::holds_alternative<std::pair<int, int>>(axis.value)) {
            throw std::invalid_argument("Tuple axis not yet supported");
        }

        throw std::invalid_argument("Unknown axis variant type");
    }

    // Case 1: all() -> bool
    bool all(const bool keep_dims = false, const where_t where = none::where) const {
        return _all_templated<bool>(none::axis, none::out, keep_dims, where);
    }

    // Case 2: all(axis_t axis) -> array<bool>
    array<bool> all(const axis_t& axis, const bool keep_dims = false, const where_t where = none::where) const {
        return _all_templated<array<bool>>(axis, none::out, keep_dims, where);
    }

    // Case 3: all(out&) -> array<bool>&
    array<bool>& all(const out_t out, const bool keep_dims = false, const where_t where = none::where) const {
        return _all_templated<array<bool>&>(none::axis, std::ref(out), keep_dims, where);
    }

    // Case 4: all(axis, out&) -> array<bool>&
    array<bool>& all(const axis_t& axis, const out_t out, const bool keep_dims = false,
                     const where_t where = none::where) const {
        return _all_templated<array<bool>&>(axis, std::ref(out), keep_dims, where);
    }
*/
