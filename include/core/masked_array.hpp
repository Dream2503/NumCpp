#pragma once

namespace numcpp {
    class MaskedArray : public array<dtype::bool_t> {
        MaskedArray(auto begin, auto end, shape_t shape) {
            const size_t size = end - begin;

            if (shape.cols == none::size) {
                shape.cols = size;
            }
            if (shape.size() != size) {
                throw std::invalid_argument("Size mismatch in flat constructor");
            }
            buffer_t<dtype::bool_t> buf((size + 7) / 8);
            std::fill_n(buf.data(), buf.size, dtype::bool_t());

            for (size_t i = 0; i < size; i++) {
                if (begin[i]) {
                    buf[i / 8][i % 8] = true;
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

            buffer_t<dtype::bool_t> buf((size + 7) / 8);
            std::fill_n(buf.data(), buf.size, dtype::bool_t{});

            size_t i = 0;
            for (const std::initializer_list<bool>& row : lists) {
                for (const bool val : row) {
                    if (val) {
                        buf[i / 8][i % 8] = true;
                    }
                    i++;
                }
            }
            *this = array(std::move(buf), {rows, cols});
        }
        MaskedArray(const array<bool>& base) :
            MaskedArray(base.buffer.data(), base.buffer.data() + base.size(), base.shape()) {}

        bool mask_at(const size_t idx) const { return buffer[idx / 8][idx % 8]; }

        MaskedArray operator[](const index_t& index) const { return MaskedArray(array::operator[](index)); }

        MaskedArray& operator=(const bool& other) {
            if (is_scalar) {
                dtype::bitref_t(buffer[offset / 8].value, offset % 8) = other;
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

        operator const bool() const {
            if (!is_scalar) {
                throw std::invalid_argument("illegal boolean conversion of an MaskedArray");
            }
            return dtype::bitref_t(*this);
        }
    };
} // namespace numcpp
