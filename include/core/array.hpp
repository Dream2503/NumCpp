#pragma once
#include "../libs/broadcasting.hpp"
#include "../libs/none.hpp"

template <typename V>
class numcpp::array {
    buffer_t<V> buffer;
    size_t row, col, offset, row_stride, col_stride;
    const void* _base;
    bool is_matrix, is_scalar, is_assignable;

    template <typename>
    friend class array;
    friend class MaskedArray;

    array flat_constructor(auto begin, auto end, shape_t shape) {
        if (shape.cols == none::size) {
            shape.cols = end - begin;
        }
        if (shape.rows * shape.cols != end - begin) {
            throw std::invalid_argument("Size mismatch in flat constructor");
        }
        buffer = buffer_t<V>(shape.rows * shape.cols);
        std::copy(begin, end, buffer.data());
        return array(std::move(buffer), shape, 0, shape.cols, 1, none::base, shape.rows > 1 && shape.cols > 1, false,
                     false);
    }

    array(buffer_t<V> data, const shape_t& shape, const size_t offset, const size_t row_stride, const size_t col_stride,
          const void* base, const bool is_matrix, const bool is_scalar, const bool is_assignable) noexcept :
        buffer(std::move(data)), row(shape.rows), col(shape.cols), offset(offset),
        row_stride(row_stride == none::size ? shape.rows : row_stride), col_stride(col_stride), _base(base),
        is_matrix(is_matrix), is_scalar(is_scalar), is_assignable(is_assignable) {}

    bool is_contiguous() const { return row_stride == col && col_stride == 1; }

public:
    array() = default;

    array(const array& other) :
        array(other.buffer, {other.row, other.col}, other.offset, other.row_stride, other.col_stride,
              other._base ? other._base : &other, other.is_matrix, other.is_scalar, false) {}

    array(array&& other) noexcept = default;

    array(std::initializer_list<V> list, const shape_t& shape = none::shape) {
        *this = flat_constructor(list.begin(), list.end(), shape);
    }

    array(const std::vector<V>& list, const shape_t& shape = none::shape) {
        *this = flat_constructor(list.begin(), list.end(), shape);
    }

    array(std::vector<V>&& list, const shape_t& shape = none::shape) :
        row(shape.rows), col(shape.cols), offset(0), row_stride(col), col_stride(1), _base(none::base),
        is_matrix(row > 1 && col > 1), is_scalar(false), is_assignable(false) {
        if (col == none::size) {
            col = list.size();
        }
        if (row * col != list.size()) {
            throw std::invalid_argument("Size mismatch in flat move constructor");
        }
        buffer = buffer_t<V>(std::move(list));
        std::move(list.begin(), list.end(), buffer.data());
    }

    array(const std::initializer_list<std::initializer_list<V>>& lists) :
        row(lists.size()), col(lists.begin()->size()), offset(0), row_stride(col), col_stride(1), _base(none::base),
        is_matrix(row > 1 && col > 1), is_scalar(false), is_assignable(false) {
        for (const std::initializer_list<V>& list : lists) {
            if (list.size() != col) {
                throw std::invalid_argument("Inconsistent list sizes");
            }
        }
        buffer = buffer_t<V>(row * col);
        size_t index = 0;

        for (const std::initializer_list<V>& list : lists) {
            std::copy(list.begin(), list.end(), buffer.data() + index);
            index += col;
        }
    }

    array(std::vector<std::vector<V>>&& lists) :
        row(lists.size()), col(lists.empty() ? 0 : lists.front().size()), offset(0), row_stride(col), col_stride(1),
        _base(none::base), is_matrix(row > 1 && col > 1), is_scalar(false), is_assignable(false) {
        for (std::vector<V>& list : lists) {
            if (list.size() != col) {
                throw std::invalid_argument("Inconsistent inner vector sizes");
            }
        }
        if (row * col == 0) {
            buffer.reset();
        } else {
            buffer = buffer_t<V>(row * col);
            size_t index = 0;

            for (std::vector<V>& list : lists) {
                std::move(list.begin(), list.end(), buffer.data() + index);
                index += col;
            }
        }
    }

    array(buffer_t<V>& buf, const shape_t& shape, const bool copy = true) :
        row(shape.rows), col(shape.cols), offset(0), row_stride(col), col_stride(1), _base(none::base),
        is_matrix(row > 1 && col > 1), is_scalar(false), is_assignable(false) {
        if (copy) {
            buffer = buffer_t<V>(row * col);
            std::copy_n(buf.data(), row * col, buffer.data());
        } else {
            buffer = buf;
        }
    }

    array(buffer_t<V>&& buf, const shape_t& shape) : array(buf, shape, false) {}

    array(V* list, const shape_t& shape, const bool copy = true) :
        row(shape.rows), col(shape.cols), offset(0), row_stride(col), col_stride(1), _base(none::base),
        is_matrix(row > 1 && col > 1), is_scalar(false), is_assignable(false) {
        if (copy) {
            buffer = buffer_t<V>(row * col);
            std::copy_n(list, row * col, buffer.data());
        } else {
            buffer = buffer_t<V>(list, row * col);
        }
    }

    const array* base() const noexcept { return _base; }

    const buffer_t<V>& data() const { return buffer; }

    buffer_t<V>& data() { return buffer; }

    size_t ndim() const noexcept { return row > 1 && col > 1 ? 2 : 1; }

    size_t size() const noexcept { return row * col; }

    template <typename dtype = V>
    array<real_t<dtype>> abs(out_t<real_t<dtype>> out = none::out, const where_t& where = none::where) const;

    template <typename dtype = V>
    array<real_t<dtype>> abs(const where_t& where) const {
        return abs<real_t<dtype>>(none::out, where);
    }

    auto real(out_t<real_t<V>> = none::out, const where_t& = none::where);

    auto real(const where_t& where) const { return real(none::out, where); }

    auto imag(out_t<real_t<V>> = none::out, const where_t& = none::where);

    auto imag(const where_t& where) const { return imag(none::out, where); }

    shape_t shape() const noexcept { return {row, col}; }

    array operator[](const index_t&) const;

    array& operator=(const array& other) {
        if (other.is_scalar) {
            this->operator=(static_cast<const V&>(other));
        }
        if (is_assignable) {
            is_assignable = false;

            const shape_t lhs_shape = shape(), rhs_shape = other.shape();
            const shape_t res_shape = broadcast_shape(lhs_shape, rhs_shape);

            if (res_shape.rows != row || res_shape.cols != col) {
                throw std::runtime_error("Broadcasted shape doesn't match array shape.");
            }

            for (ll_t i = 0; i < row; i++) {
                for (ll_t j = 0; j < col; j++) {
                    (*this)[{i, j}] = other[broadcast_index({i, j}, rhs_shape)];
                }
            }
        } else if (this != &other) {
            buffer = other.buffer;
            row = other.row;
            col = other.col;
            offset = other.offset;
            row_stride = other.row_stride;
            col_stride = other.col_stride;
            _base = other._base;
            is_scalar = other.is_scalar;
            is_matrix = other.is_matrix;
        }
        return *this;
    }

    array& operator=(const V& other) {
        if (is_assignable) {
            is_assignable = false;

            for (ll_t i = 0; i < row; i++) {
                for (ll_t j = 0; j < col; j++) {
                    (*this)[{i, j}] = other;
                }
            }
        } else if (is_scalar) {
            buffer[offset] = other;
        } else {
            throw std::invalid_argument("Illegal assignment of a scalar to a non-scalar array.");
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& out, const array& other) {
        auto [row, col] = other.shape();
        const bool is_col_vector = (col == 1);
        const size_t width_dim = is_col_vector ? row : col;
        size_t col_width = 0;
        std::vector<size_t> col_width_vec;

        if (other.is_matrix) {
            col_width_vec = std::vector(width_dim, 0ul);
        }
        for (ll_t i = 0; i < row; i++) {
            for (ll_t j = 0; j < col; j++) {
                size_t size;
                if constexpr (std::is_same_v<V, dtype::bool_t>) {
                    size = detail::format(dtype::bitref_t(other[{i, j}])).size();
                } else {
                    size = detail::format(static_cast<V>(other[{i, j}])).size();
                }
                if (other.is_matrix) {
                    const size_t pos = is_col_vector ? i : j;
                    col_width_vec[pos] = std::max(col_width_vec[pos], size);
                } else {
                    col_width = std::max(col_width, size);
                }
            }
        }
        if (!other.is_scalar) {
            out << '[';
        }

        for (ll_t i = 0; i < row; i++) {
            if (other.is_matrix) {
                out << (i == 0 ? "[" : " [");
            }
            for (ll_t j = 0; j < col; j++) {
                if (j > 0 || (!other.is_matrix && i > 0)) {
                    out << ' ';
                }
                if (other.is_matrix) {
                    out << std::setw(col_width_vec[is_col_vector ? i : j]);
                } else {
                    out << std::setw(col_width);
                }
                if constexpr (std::is_same_v<V, dtype::bool_t>) {
                    out << detail::format(dtype::bitref_t(other[{i, j}]));
                } else {
                    out << detail::format(static_cast<V>(other[{i, j}]));
                }
            }
            if (other.is_matrix) {
                out << "]";
            }

            if (other.is_matrix && i < row - 1) {
                out << '\n';
            }
        }
        if (!other.is_scalar) {
            out << ']';
        }
        out << std::flush;
        return out;
    }

    operator const V&() const {
        if (!is_scalar) {
            throw std::invalid_argument("illegal scalar conversion of an array");
        }
        return buffer[offset];
    }

    operator V&() { return const_cast<V&>(static_cast<const array&>(*this).operator const V&()); }

    operator dtype::bitref_t() requires(std::is_same_v<V, dtype::bool_t>)
    {
        if (!is_scalar) {
            throw std::invalid_argument("illegal scalar conversion of an array");
        }
        return dtype::bitref_t(buffer[offset / 8].value, offset % 8);
    }


    operator dtype::bool_t() const requires(std::is_same_v<V, dtype::bool_t>)
    {
        if (!is_scalar) {
            throw std::invalid_argument("illegal scalar conversion of an array");
        }
        return dtype::bool_t((buffer[offset / 8].value >> (offset % 8)) & 1);
    }

    // array T() const {
    //     return array(buffer, {col, row}, offset, col_stride, row_stride, _base ? _base : this, is_matrix);
    // }

    array reshape(const size_t rows, const size_t cols) const {
        if (rows * cols != size()) {
            throw std::invalid_argument("reshape size mismatch");
        }
        if (!is_contiguous()) {
            throw std::runtime_error("cannot reshape a non-contiguous array");
        }
        return array(buffer, {rows, cols}, offset, cols, 1, _base ? _base : this, rows > 1 && cols > 1);
    }

    array copy() const {
        buffer_t<V> buf = buffer_t<V>(row * col);
        std::copy_n(buffer.data(), row * col, buf.data());
        return array(std::move(buf), {row, col}, offset, row_stride, col_stride, nullptr, is_matrix, is_scalar, false);
    }
};
