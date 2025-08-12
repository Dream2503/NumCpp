#pragma once
#include "../libs/traits.hpp"
#include "../libs/types.hpp"
#include "array.hpp"

namespace numcpp {
    class MaskedArray : public array<dtypes::bool_t> {
        MaskedArray(auto begin, auto end, shape_t shape) {
            const size_t size = end - begin;

            if (shape.cols == none::size) {
                shape.cols = size;
            }
            if (shape.size() != size) {
                throw std::invalid_argument("Size mismatch in flat constructor");
            }
            buffer_t<dtypes::bool_t> buf(size);
            std::fill_n(buf.data(), buf.size, dtypes::bool_t());

            for (size_t i = 0; i < size; i++) {
                if (begin[i]) {
                    buf[i] = true;
                }
            }
            *this = array(std::move(buf), shape);
        }

    public:
        using array::array;
        using array::operator=;

        MaskedArray(const array& other) : array(other) {}
        MaskedArray(array&& other) noexcept : array(other) {}

        MaskedArray(const std::initializer_list<bool> list, const shape_t& shape = none::shape) :
            MaskedArray(list.begin(), list.end(), shape) {}

        MaskedArray(const std::initializer_list<std::initializer_list<bool>> lists) {
            size_t rows = lists.size();
            size_t cols = lists.begin()->size();
            const size_t size = rows * cols;

            for (const std::initializer_list<bool>& row : lists) {
                if (row.size() != cols) {
                    throw std::invalid_argument("Inconsistent list sizes");
                }
            }

            buffer_t<dtypes::bool_t> buf(size);
            std::fill_n(buf.data(), buf.size, dtypes::bool_t{});

            size_t i = 0;
            for (const std::initializer_list<bool>& row : lists) {
                for (const bool val : row) {
                    if (val) {
                        buf[i++] = true;
                    }
                }
            }
            *this = array(std::move(buf), {rows, cols});
        }
        MaskedArray(const array<bool>& base) :
            MaskedArray(base.buffer.data(), base.buffer.data() + base.size(), base.shape()) {}

        bool mask_at(const ll_t i, const ll_t j) const noexcept {
            return (*this)[{static_cast<ll_t>(broadcast_index(i, shape().rows)),
                            static_cast<ll_t>(broadcast_index(j, shape().cols))}];
        }

        MaskedArray operator[](const index_t& index) const { return MaskedArray(array::operator[](index)); }

        MaskedArray& operator=(const bool& other) {
            if (is_scalar) {
                buffer[offset] = other;
            } else if (is_assignable) {
                for (ll_t i = 0; i < row; i++) {
                    for (ll_t j = 0; j < col; j++) {
                        (*this)[{i, j}] = other;
                    }
                }
            } else {
                throw std::invalid_argument("Illegal assignment of a scalar to a non-scalar array.");
            }
            is_assignable = false;
            return *this;
        }

        operator bool() {
            if (!is_scalar) {
                throw std::invalid_argument("illegal boolean conversion of an MaskedArray");
            }
            return buffer[offset];
        }
    };
} // namespace numcpp
