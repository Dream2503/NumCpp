#pragma once
#include "../libs/broadcasting.hpp"
#include "../libs/none.hpp"

template <typename V>
class numcpp::array {
    buffer_t<V> data;
    size_t row, col, offset, row_stride, col_stride;
    const void* _base;
    bool is_matrix, is_scalar, is_assignable;

    template <typename>
    friend class array;

    void flat_constructor(auto begin, auto end, shape_t shape) {
        if (shape.cols == none::size) {
            shape.cols = end - begin;
        }
        if (shape.rows * shape.cols != end - begin) {
            throw std::invalid_argument("Size mismatch in flat constructor");
        }
        row = shape.rows;
        col = shape.cols;
        offset = 0;
        row_stride = col;
        col_stride = 1;
        _base = none::base;
        is_matrix = row > 1 && col > 1;
        is_scalar = is_assignable = false;
        data = buffer_t<V>(row * col);
        std::copy(begin, end, data.data());
    }

    array(buffer_t<V> data, const shape_t& shape, const size_t offset, const size_t row_stride, const size_t col_stride,
          const void* base, const bool is_matrix, const bool is_scalar, const bool is_assignable) noexcept :
        data(std::move(data)), row(shape.rows), col(shape.cols), offset(offset),
        row_stride(row_stride == none::size ? shape.rows : row_stride), col_stride(col_stride), _base(base),
        is_matrix(is_matrix), is_scalar(is_scalar), is_assignable(is_assignable) {}

    bool is_contiguous() const { return row_stride == col && col_stride == 1; }

public:
    array() = delete;

    array(const array& other) :
        array(other.data, {other.row, other.col}, other.offset, other.row_stride, other.col_stride,
              other._base ? other._base : &other, other.is_matrix, other.is_scalar, false) {}

    array(array&& other) noexcept = default;

    array(std::initializer_list<V> list, const shape_t& shape = none::shape) {
        flat_constructor(list.begin(), list.end(), shape);
    }

    array(const std::vector<V>& list, const shape_t& shape = none::shape) {
        flat_constructor(list.begin(), list.end(), shape);
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
        data = buffer_t<V>(std::move(list));
        std::move(list.begin(), list.end(), data.data());
    }

    array(const std::initializer_list<std::initializer_list<V>>& lists) :
        row(lists.size()), col(lists.begin()->size()), offset(0), row_stride(col), col_stride(1), _base(none::base),
        is_matrix(row > 1 && col > 1), is_scalar(false), is_assignable(false) {
        for (const std::initializer_list<V>& list : lists) {
            if (list.size() != col) {
                throw std::invalid_argument("Inconsistent list sizes");
            }
        }
        data = buffer_t<V>(row * col);
        size_t index = 0;

        for (const std::initializer_list<V>& list : lists) {
            std::copy(list.begin(), list.end(), data.data() + index);
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
            data.reset();
        } else {
            data = buffer_t<V>(row * col);
            size_t index = 0;

            for (std::vector<V>& list : lists) {
                std::move(list.begin(), list.end(), data.data() + index);
                index += col;
            }
        }
    }

    array(buffer_t<V>& buf, const shape_t& shape, const bool copy = true) :
        row(shape.rows), col(shape.cols), offset(0), row_stride(col), col_stride(1), _base(none::base),
        is_matrix(row > 1 && col > 1), is_scalar(false), is_assignable(false) {
        if (copy) {
            data = buffer_t<V>(row * col);
            std::copy(buf.data(), buf.data() + row * col, data.data());
        } else {
            data = buf;
        }
    }

    array(V* list, const shape_t& shape, const bool copy = true) :
        row(shape.rows), col(shape.cols), offset(0), row_stride(col), col_stride(1), _base(none::base),
        is_matrix(row > 1 && col > 1), is_scalar(false), is_assignable(false) {
        if (copy) {
            data = buffer_t<V>(row * col);
            std::copy(list, list + row * col, data.data());
        } else {
            data = buffer_t<V>(list, row * col);
        }
    }

    const array* base() const noexcept { return _base; }
    size_t ndim() const noexcept { return row > 1 && col > 1 ? 2 : 1; }
    size_t size() const noexcept { return row * col; }

    template <class dtype = V>
    array<dtype> abs(out_t<dtype> out = none::out, const where_t& where = none::where) const;
    auto real(out_t<real_t<V>> = none::out, const where_t& = none::where);
    auto imag(out_t<real_t<V>> = none::out, const where_t& = none::where);

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
        } else if (this == &other) {
            data = other.data;
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
            data[offset] = other;
        } else {
            throw std::invalid_argument("Illegal assignment of a scalar to a non-scalar array.");
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& out, const array& other) {
        auto [row, col] = other.shape();
        const bool is_col_vector = (col == 1);
        const size_t width_dim = is_col_vector ? col : row;
        size_t col_width = 0;
        std::vector<size_t> col_width_vec;

        if (other.is_matrix) {
            col_width_vec = std::vector(width_dim, 0ul);
        }
        for (ll_t i = 0; i < row; i++) {
            for (ll_t j = 0; j < col; j++) {
                const array element = other[{i, j}];
                const size_t size = format(element.data[element.offset]).size();

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
                const array element = other[{i, j}];
                out << format(element.data[element.offset]);
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
        return data[offset];
    }

    operator V&() { return const_cast<V&>(static_cast<const array&>(*this).operator const V&()); }

    array T() const { return array(data, {col, row}, offset, col_stride, row_stride, _base ? _base : this, is_matrix); }

    array reshape(const size_t rows, const size_t cols) const {
        if (rows * cols != size()) {
            throw std::invalid_argument("reshape size mismatch");
        }
        if (!is_contiguous()) {
            throw std::runtime_error("cannot reshape a non-contiguous array");
        }
        return array(data, {rows, cols}, offset, cols, 1, _base ? _base : this, rows > 1 && cols > 1);
    }

    array copy() const {
        buffer_t<V> buf = buffer_t<V>(row * col);
        std::copy(data.data(), data.data() + row * col, buf.data());
        return array(std::move(buf), {row, col}, offset, row_stride, col_stride, _base, is_matrix, is_scalar, false);
    }
};
