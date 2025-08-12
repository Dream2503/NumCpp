#pragma once
#include "../libs/utils.hpp"

template <typename T>
class numcpp::array {
    buffer_t<T> buffer = buffer_t<T>();
    size_t row = 0, col = 0, offset = 0, row_stride = 0, col_stride = 0;
    const void* base = none::base;
    bool is_matrix = false, is_scalar = false, is_assignable = false;

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
    class iterator;

    array() noexcept = default;

    array(const array& other) noexcept :
        array(other.buffer, {other.row, other.col}, other.offset, other.row_stride, other.col_stride,
              other.base ? other.base : &other, other.is_matrix, other.is_scalar, false) {}

    array(array&& other) noexcept :
        array(std::move(other.buffer), {other.row, other.col}, other.offset, other.row_stride, other.col_stride,
              other.base, other.is_matrix, other.is_scalar, other.is_assignable) {
        other.row = other.col = other.offset = other.row_stride = other.col_stride = 0;
        other.base = none::base;
        other.is_matrix = other.is_scalar = other.is_assignable = false;
    }

    array(std::initializer_list<T> list, const shape_t& shape = none::shape) {
        *this = flat_constructor(list.begin(), list.end(), shape);
    }

    array(const std::vector<T>& list, const shape_t& shape = none::shape) {
        *this = flat_constructor(list.begin(), list.end(), shape);
    }

    array(std::vector<T>&& list, const shape_t& shape = none::shape) :
        row(shape.rows), col(shape.cols), row_stride(col), col_stride(1), is_matrix(row > 1 && col > 1) {
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
        row(lists.size()), col(lists.empty() ? 0 : lists.front().size()), row_stride(col), col_stride(1),
        is_matrix(row > 1 && col > 1) {
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

    array(const buffer_t<T>& buf, const shape_t& shape, const bool copy = true) :
        row(shape.rows), col(shape.cols), row_stride(col), col_stride(1), is_matrix(row > 1 && col > 1) {
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

    array(const T& value) : array(&value, {1, 1}) { is_scalar = true; }

    size_t ndim() const noexcept { return row > 1 && col > 1 ? 2 : 1; }

    size_t size() const noexcept { return row * col; }

    // template <typename dtype = T>
    // requires(is_numeric_v<T>)
    // array<real_t<dtype>> abs(out_t<real_t<dtype>> out = none::out, const where_t& where = none::where) const
    // noexcept; template <typename dtype = T> requires(is_numeric_v<T>) array<real_t<dtype>> abs(const where_t& where)
    // const noexcept;

    array<real_t<T>> real() noexcept requires(is_numeric_v<T>)
    {
        using V = real_t<T>;

        if constexpr (is_complex_v<T>) {
            return array<V>(buffer_t<V>(reinterpret_cast<V*>(buffer.data()), size() * 2, nullptr), {row, col},
                            offset * 2, row_stride * 2, col_stride * 2, this, is_matrix, row == 1 && col == 1, true);
        } else {
            array res = *this;
            res.is_assignable = true;
            return res;
        }
    }

    array<real_t<T>> imag() noexcept requires(is_numeric_v<T>)
    {
        using V = real_t<T>;

        if constexpr (is_complex_v<T>) {
            return array<V>(buffer_t<V>(reinterpret_cast<V*>(buffer.data()), size() * 2, nullptr), {row, col},
                            offset * 2 + 1, row_stride * 2, col_stride * 2, this, is_matrix, row == 1 && col == 1,
                            true);
        } else {
            return zeros(shape());
        }
    }

    shape_t shape() const noexcept { return {row, col}; }

    array operator[](const index_t&) const;

    array& operator=(const array& other) {
        if (other.is_scalar) {
            buffer[offset] = other;
        }
        if (is_assignable) {
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
        is_assignable = false;
        return *this;
    }

    array& operator=(const T& other) {
        if (is_scalar) {
            buffer[offset] = other;
        } else if (is_assignable) {
            for (ll_t i = 0; i < row * col; i++) {
                buffer[i] = other;
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

                if constexpr (is_bool_t<T>) {
                    size = detail::format(bitref_t(other[{i, j}])).size();
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
                out << (other.is_matrix ? std::setw(col_width_vec[is_col_vector ? i : j]) : std::setw(col_width));

                if constexpr (is_bool_t<T>) {
                    out << detail::format(bitref_t(other[{i, j}]));
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

    operator std::conditional_t<is_bool_t<T>, bitref_t, const T&>() const {
        if (is_scalar || size() == 1) {
            return buffer[offset];
        }
        throw std::invalid_argument("illegal scalar conversion of an array");
    }
    operator std::conditional_t<is_bool_t<T>, bitref_t, T&>() {
        if (is_scalar || size() == 1) {
            return buffer[offset];
        }
        throw std::invalid_argument("illegal scalar conversion of an array");
    }

    explicit operator bool() const {
        if (is_scalar || size() == 1) {
            return buffer[offset];
        }
        throw std::invalid_argument("illegal scalar conversion of an array");
    }
    // operator bitref_t() requires(is_bool_t<T>)
    // {
    //     if (!is_scalar) {
    //         throw std::invalid_argument("illegal bool& conversion of an array");
    //     }
    //     return buffer[offset];
    // }
    // operator bool_t() const requires(std::is_same_v<T, bool_t>)
    // {
    //     if (!is_scalar) {
    //         throw std::invalid_argument("illegal bool_t conversion of an array");
    //     }
    //     return buffer[offset];
    // }

    // array T() const noexcept {
    //     return array(buffer, {col, row}, offset, col_stride, row_stride, base ? base : this, is_matrix, is_scalar,
    //                  true);
    // }

    array reshape(const shape_t& shape) const {
        if (shape.size() != size()) {
            throw std::invalid_argument("reshape size mismatch");
        }
        if (!is_contiguous()) {
            throw std::runtime_error("cannot reshape a non-contiguous array");
        }
        return array(buffer, shape, offset, shape.cols, 1, base ? base : this, shape.rows > 1 && shape.cols > 1, false,
                     false);
    }

    array copy() const noexcept { return array(buffer, shape()); }

    template <typename V>
    friend size_t offset(const array<V>&) noexcept;
    template <typename V>
    friend buffer_t<V> buffer(const array<V>&) noexcept;
    template <typename V>
    friend const void* base(const array<V>&) noexcept;
    template <typename V>
    friend bool is_matrix(const array<V>&) noexcept;
    template <typename V>
    friend bool is_scalar(const array<V>&) noexcept;
    template <typename V>
    friend bool is_assignable(const array<V>&) noexcept;

    constexpr iterator begin() noexcept { return iterator(buffer.data() + offset, row_stride, col_stride); }
    constexpr iterator end() noexcept {
        return iterator(buffer.data() + offset + size() * col_stride, row_stride, col_stride);
    }
};

template <typename T>
class numcpp::array<T>::iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;

private:
    pointer current;
    difference_type row_stride, col_stride;

public:
    constexpr explicit iterator(const pointer ptr, const difference_type row_stride_,
                                const difference_type col_stride_) noexcept :
        current(ptr), row_stride(row_stride_), col_stride(col_stride_) {}

    constexpr reference operator*() const noexcept { return *current; }
    constexpr pointer operator->() const noexcept { return current; }

    constexpr iterator& operator++() noexcept {
        current += col_stride;
        return *this;
    }
    constexpr iterator operator++(int) noexcept {
        iterator tmp = *this;
        current += col_stride;
        return tmp;
    }
    constexpr iterator& operator--() noexcept {
        current -= col_stride;
        return *this;
    }
    constexpr iterator operator--(int) noexcept {
        iterator tmp = *this;
        current -= col_stride;
        return tmp;
    }

    constexpr iterator operator+(const difference_type n) const noexcept {
        return iterator(current + n * col_stride, row_stride, col_stride);
    }
    friend constexpr iterator operator+(const difference_type n, const iterator& it) noexcept { return it + n; }
    constexpr iterator operator-(const difference_type n) const noexcept {
        return iterator(current - n * col_stride, row_stride, col_stride);
    }
    constexpr difference_type operator-(const iterator& other) const noexcept {
        return (current - other.current) / col_stride;
    }

    constexpr reference operator[](const difference_type n) const noexcept { return *(current + n * col_stride); }
    constexpr bool operator==(const iterator& other) const noexcept { return current == other.current; }
    constexpr bool operator!=(const iterator& other) const noexcept { return current != other.current; }
    constexpr bool operator<(const iterator& other) const noexcept { return current < other.current; }
    constexpr bool operator>(const iterator& other) const noexcept { return current > other.current; }
    constexpr bool operator<=(const iterator& other) const noexcept { return current <= other.current; }
    constexpr bool operator>=(const iterator& other) const noexcept { return current >= other.current; }
};
