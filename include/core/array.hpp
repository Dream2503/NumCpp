#pragma once
#include "../libs/types.hpp"

template <typename T>
class numcpp::array {
    buffer_t<T> buffer;
    size_t row, col, offset, row_stride, col_stride;
    const void* base;
    bool is_matrix, is_scalar, is_assignable;

    template <typename>
    friend class array;
    friend class MaskedArray;

    array flat_constructor(auto begin, auto end, shape_t shape) {
        if (shape.cols == none::size) {
            shape.cols = end - begin;
        }
        if (shape.size() != end - begin) {
            throw std::invalid_argument("Size mismatch in flat constructor");
        }
        buffer = buffer_t<T>(shape.size());
        std::copy(begin, end, buffer.data());
        return array(std::move(buffer), shape, 0, shape.cols, 1, none::base, shape.rows > 1 && shape.cols > 1, false,
                     false);
    }

    array nested_constructor(const auto& lists) {
        row = lists.size();
        col = lists.begin()->size();

        if (row * col == 0) {
            return array();
        }
        for (const auto& list : lists) {
            if (list.size() != col) {
                throw std::invalid_argument("Size mismatch in nested constructor");
            }
        }
        buffer = buffer_t<T>(row * col);
        size_t idx = 0;

        for (const auto& list : lists) {
            std::copy(list.begin(), list.end(), buffer.data() + idx);
            idx += col;
        }
        return array(std::move(buffer), {row, col}, 0, col, 1, none::base, row > 1 && col > 1, false, false);
    }

    array(buffer_t<T> data, const shape_t& shape, const size_t offset, const size_t row_stride, const size_t col_stride,
          const void* base, const bool is_matrix, const bool is_scalar, const bool is_assignable) noexcept :
        buffer(std::move(data)), row(shape.rows), col(shape.cols), offset(offset),
        row_stride(row_stride == none::size ? shape.rows : row_stride), col_stride(col_stride), base(base),
        is_matrix(is_matrix), is_scalar(is_scalar), is_assignable(is_assignable) {}

    bool is_contiguous() const { return row_stride == col && col_stride == 1; }

public:
    array() : array(buffer_t<T>(), {0, 0}, 0, 0, 0, nullptr, false, false, false) {}

    array(const array& other) :
        array(other.buffer, {other.row, other.col}, other.offset, other.row_stride, other.col_stride,
              other.base ? other.base : &other, other.is_matrix, other.is_scalar, false) {}

    array(array&& other) noexcept = default;

    array(std::initializer_list<T> list, const shape_t& shape = none::shape) {
        *this = flat_constructor(list.begin(), list.end(), shape);
    }

    array(const std::vector<T>& list, const shape_t& shape = none::shape) {
        *this = flat_constructor(list.begin(), list.end(), shape);
    }

    array(std::vector<T>&& list, const shape_t& shape = none::shape) :
        row(shape.rows), col(shape.cols), offset(0), row_stride(col), col_stride(1), base(none::base),
        is_matrix(row > 1 && col > 1), is_scalar(false), is_assignable(false) {
        if (col == none::size) {
            col = list.size();
        }
        if (row * col != list.size()) {
            throw std::invalid_argument("Size mismatch in flat move constructor");
        }
        buffer = buffer_t<T>(std::move(list));
        std::move(list.begin(), list.end(), buffer.data());
    }

    array(const std::initializer_list<std::initializer_list<T>>& lists) { *this = nested_constructor(lists); }

    array(const std::vector<std::vector<T>>& lists) { *this = nested_constructor(lists); }

    array(std::vector<std::vector<T>>&& lists) :
        row(lists.size()), col(lists.empty() ? 0 : lists.front().size()), offset(0), row_stride(col), col_stride(1),
        base(none::base), is_matrix(row > 1 && col > 1), is_scalar(false), is_assignable(false) {
        for (std::vector<T>& list : lists) {
            if (list.size() != col) {
                throw std::invalid_argument("Inconsistent inner vector sizes");
            }
        }
        if (row * col == 0) {
            buffer.reset();
        } else {
            buffer = buffer_t<T>(row * col);
            size_t index = 0;

            for (std::vector<T>& list : lists) {
                std::move(list.begin(), list.end(), buffer.data() + index);
                index += col;
            }
        }
    }

    array(buffer_t<T>& buf, const shape_t& shape, const bool copy = true) :
        row(shape.rows), col(shape.cols), offset(0), row_stride(col), col_stride(1), base(none::base),
        is_matrix(row > 1 && col > 1), is_scalar(false), is_assignable(false) {
        if (copy) {
            buffer = buffer_t<T>(row * col);
            std::copy_n(buf.data(), row * col, buffer.data());
        } else {
            buffer = buf;
        }
    }

    array(buffer_t<T>&& buf, const shape_t& shape) : array(buf, shape, false) {}

    array(const T* list, const shape_t& shape, const bool copy = true) :
        array(copy ? buffer_t<T>(const_cast<T*>(list), shape.size())
                   : buffer_t<T>(const_cast<T*>(list), shape.size(), nullptr),
              shape) {}

    explicit array(const T& value) : array(&value, {1, 1}) { is_scalar = true; }

    size_t ndim() const noexcept { return row > 1 && col > 1 ? 2 : 1; }

    size_t size() const noexcept { return row * col; }

    template <typename dtype = T>
    requires(is_numeric_v<T>)
    array<real_t<dtype>> abs(out_t<real_t<dtype>> out = none::out, const where_t& where = none::where) const noexcept;
    template <typename dtype = T>
    requires(is_numeric_v<T>)
    array<real_t<dtype>> abs(const where_t& where) const noexcept;

    template <typename dtype = T>
    requires(is_numeric_v<T> && !is_complex_v<T>)
    array<dtype> floor(out_t<dtype> out = none::out, const where_t& where = none::where) const noexcept;
    template <typename dtype = T>
    requires(is_numeric_v<T> && !is_complex_v<T>)
    array<dtype> floor(const where_t& where) const noexcept;

    auto real(out_t<real_t<T>> = none::out, const where_t& = none::where) noexcept requires(is_numeric_v<T>);
    auto real(const where_t& where) const noexcept requires(is_numeric_v<T>);

    auto imag(out_t<real_t<T>> = none::out, const where_t& = none::where) noexcept requires(is_numeric_v<T>);
    auto imag(const where_t& where) const noexcept requires(is_numeric_v<T>);

    shape_t shape() const noexcept { return {row, col}; }

    array operator[](const index_t&) const;

    array& operator=(const array& other) {
        if (other.is_scalar) {
            this->operator=(static_cast<const T&>(other));
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
            base = other.base;
            is_scalar = other.is_scalar;
            is_matrix = other.is_matrix;
        }
        return *this;
    }

    array& operator=(const T& other) {
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
                if constexpr (std::is_same_v<T, dtype::bool_t>) {
                    size = detail::format(dtype::bitref_t(other[{i, j}])).size();
                } else {
                    size = detail::format(static_cast<T>(other[{i, j}])).size();
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
                if constexpr (std::is_same_v<T, dtype::bool_t>) {
                    out << detail::format(dtype::bitref_t(other[{i, j}]));
                } else {
                    out << detail::format(static_cast<T>(other[{i, j}]));
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

    operator const T&() const {
        if (is_scalar || size() == 1) {
            return buffer[offset];
        }
        throw std::invalid_argument("illegal scalar conversion of an array");
    }

    operator T&() { return const_cast<T&>(static_cast<const array&>(*this).operator const T&()); }

    operator dtype::bitref_t() requires(std::is_same_v<T, dtype::bool_t>)
    {
        if (!is_scalar) {
            throw std::invalid_argument("illegal bool& conversion of an array");
        }
        return dtype::bitref_t(buffer[offset / 8].value, offset % 8);
    }

    operator dtype::bitref_t() const requires(std::is_same_v<T, dtype::bool_t>)
    {
        if (!is_scalar) {
            throw std::invalid_argument("illegal const bool& conversion of an array");
        }
        return dtype::bitref_t(const_cast<uint8_t&>(buffer[offset / 8].value), offset % 8);
    }

    operator dtype::bool_t() const requires(std::is_same_v<T, dtype::bool_t>)
    {
        if (!is_scalar) {
            throw std::invalid_argument("illegal bool_t conversion of an array");
        }
        return dtype::bool_t((buffer[offset / 8].value >> (offset % 8)) & 1);
    }

    // array T() const noexcept {
    //     return array(buffer, {col, row}, offset, col_stride, row_stride, base ? base : this, is_matrix, is_scalar,
    //                  true);
    // }

    array reshape(const size_t rows, const size_t cols) const {
        if (rows * cols != size()) {
            throw std::invalid_argument("reshape size mismatch");
        }
        if (!is_contiguous()) {
            throw std::runtime_error("cannot reshape a non-contiguous array");
        }
        return array(buffer, {rows, cols}, offset, cols, 1, base ? base : this, rows > 1 && cols > 1);
    }

    array copy() const noexcept { return array(buffer, shape()); }

    template <typename V>
    friend size_t offset(const array<V>&) noexcept;
    template <typename V>
    friend buffer_t<V> buffer(const array<V>&) noexcept;
    template <typename V>
    friend bool is_matrix(const array<V>&) noexcept;
    template <typename V>
    friend bool is_scalar(const array<V>&) noexcept;
    template <typename V>
    friend bool is_assignable(const array<V>&) noexcept;
};
