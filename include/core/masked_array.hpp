#pragma once

namespace numcpp {
    class MaskedArray : public array<bool_t> {
        MaskedArray(auto begin, auto end, shape_t shape) {
            const size_t size = end - begin;

            if (shape.cols == none::size) {
                shape.cols = size;
            }
            if (shape.size() != size) {
                throw std::invalid_argument("Size mismatch in flat constructor");
            }
            buffer_t<bool_t> buf(size);
            std::fill_n(buf.data(), buf.size, bool_t());

            for (size_t i = 0; i < size; i++) {
                if (begin[i]) {
                    buf[i] = true;
                }
            }
            *this = array(std::move(buf), shape);
        }

    public:
        using array::array;

        MaskedArray(const array& other) : array(other) {}
        MaskedArray(array&& other) noexcept : array(other) {}

        MaskedArray(const std::initializer_list<bool> list, const shape_t& shape = none::shape) :
            MaskedArray(list.begin(), list.end(), shape) {}

        MaskedArray(const std::initializer_list<std::initializer_list<bool>> lists) {
            size_t rows = lists.size();
            size_t cols = lists.begin()->size();

            for (const std::initializer_list<bool>& row : lists) {
                if (row.size() != cols) {
                    throw std::invalid_argument("Inconsistent list sizes");
                }
            }
            buffer_t<bool_t> buf(rows * cols);
            std::fill_n(buf.data(), buf.size, bool_t{});
            size_t i = 0;

            for (const std::initializer_list<bool>& row : lists) {
                for (const bool val : row) {
                    buf[i++] = val;
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
        MaskedArray operator=(const bool& other) { return MaskedArray(array::operator=(other)); }
        MaskedArray operator=(const MaskedArray& other) { return MaskedArray(array::operator=(other)); }

        operator bool() {
            if (!is_scalar) {
                throw std::invalid_argument("illegal boolean conversion of an MaskedArray");
            }
            return buffer[offset];
        }
    };
} // namespace numcpp
