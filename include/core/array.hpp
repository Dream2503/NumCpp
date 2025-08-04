#pragma once
#include "../libs/broadcasting.hpp"
#include "../libs/none.hpp"
#include "array.hpp"

template <typename V>
class numcpp::array {
    buffer_t<V> data;
    size_t row, col, offset, row_stride, col_stride;
    const void* _base;
    bool is_matrix, is_scalar;

    template <typename>
    friend class array;

    void swap(array& other) noexcept {
        using std::swap;
        swap(data, other.data);
        swap(row, other.row);
        swap(col, other.col);
        swap(offset, other.offset);
        swap(row_stride, other.row_stride);
        swap(col_stride, other.col_stride);
        swap(_base, other._base);
        swap(is_matrix, other.is_matrix);
        swap(is_scalar, other.is_scalar);
    }

    void flat_constructor(auto begin, auto end, shape_t shape) noexcept {
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
        is_scalar = false;
        data = buffer_t<V>(row * col);
        std::copy(begin, end, data.data());
    }

    array(buffer_t<V> data, const shape_t& shape, const size_t offset, const size_t row_stride = none::size,
          const size_t col_stride = 1, const void* base = none::base, const bool is_matrix = true,
          const bool is_scalar = false) noexcept :
        data(std::move(data)), row(shape.rows), col(shape.cols), offset(offset),
        row_stride(row_stride == none::size ? shape.rows : row_stride), col_stride(col_stride), _base(base),
        is_matrix(is_matrix), is_scalar(is_scalar) {}

    bool is_contiguous() const { return row_stride == col && col_stride == 1; }

public:
    array() = delete;

    array(const array& other) :
        array(other.data, {other.row, other.col}, other.offset, other.row_stride, other.col_stride,
              other._base ? other._base : &other, other.is_matrix, other.is_scalar) {}

    array(array&& other) noexcept = default;

    array(std::initializer_list<V> list, const shape_t& shape = none::shape) {
        flat_constructor(list.begin(), list.end(), shape);
    }

    array(const std::vector<V>& list, const shape_t& shape = none::shape) {
        flat_constructor(list.begin(), list.end(), shape);
    }

    array(std::vector<V>&& list, const shape_t& shape = none::shape) :
        row(shape.rows), col(shape.cols), offset(0), row_stride(col), col_stride(1), _base(none::base),
        is_matrix(row > 1 && col > 1), is_scalar(false) {
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
        is_matrix(row > 1 && col > 1), is_scalar(false) {
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
        _base(none::base), is_matrix(row > 1 && col > 1), is_scalar(false) {
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
        is_matrix(row > 1 && col > 1), is_scalar(false) {
        if (copy) {
            data = buffer_t<V>(row * col);
            std::copy(buf.data(), buf.data() + row * col, data.data());
        } else {
            data = buf;
        }
    }

    array(V* list, const shape_t& shape, const bool copy = true) :
        row(shape.rows), col(shape.cols), offset(0), row_stride(col), col_stride(1), _base(none::base),
        is_matrix(row > 1 && col > 1), is_scalar(false) {
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

    auto real() {
        if constexpr (is_complex_v<V>) {
            using T = typename V::value_type;

            T* real_ptr = reinterpret_cast<T*>(data.data());
            size_t new_offset = offset * 2;

            return array<T>(buffer_t<T>(real_ptr, size() * 2, nullptr), {row, col}, new_offset, row_stride * 2,
                            col_stride * 2, this, is_matrix, is_scalar);
        } else {
            return *this;
        }
    }

    template <typename T>
    void real(const array<T>& other) {
        static_assert(is_complex_v<V>, "real() assignment only valid for complex array.");

        const shape_t other_shape = other.shape(), res_shape = broadcast_shape(shape(), other_shape);

        if (res_shape.rows != row || res_shape.cols != col) {
            throw std::runtime_error("Broadcasted shape doesn't match array shape.");
        }
        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                const index_t& real_index = broadcast_index({i, j}, other_shape);
                V& val = (*this)[{i, j}];
                val = V(other[real_index], val.imag());
            }
        }
    }

    auto imag() const {
        if constexpr (is_complex_v<V>) {
            using T = typename V::value_type;
            std::vector<T> real_data;
            real_data.reserve(size());

            for (size_t i = 0; i < row; i++) {
                for (size_t j = 0; j < col; j++) {
                    real_data.push_back((*this)[{i, j}].imag());
                }
            }
            return array<T>(std::move(real_data), row, col);
        } else {
            return *this;
        }
    }

    template <typename T>
    void imag(const array<T>& other) {
        if constexpr (!is_complex_v<V>) {
            throw std::runtime_error("Cannot set real part of a non-complex array.");
        }
        auto [res_row, res_col] = broadcast_shape(shape(), other.shape());

        if (res_row != row || res_col != col) {
            throw std::runtime_error("Broadcasted shape doesn't match array shape.");
        }
        for (size_t i = 0; i < row; i++) {
            size_t bi = broadcast_index(i, other.shape().first);

            for (size_t j = 0; j < col; j++) {
                size_t bj = broadcast_index(j, other.shape().second);
                V& val = (*this)[{i, j}];
                val = V(val.real(), other[{bi, bj}]);
            }
        }
    }

    shape_t shape() const noexcept { return {row, col}; }

    array operator[](const index_t&) const;

    array& operator=(array other) noexcept {
        swap(other);
        return *this;
    }

    array& operator=(const V& other) {
        if (!is_scalar) {
            throw std::invalid_argument("illegal assigment of an scalar to a array type");
        }
        data[offset] = other;
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
        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
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
        out << '[';

        for (size_t i = 0; i < row; i++) {
            if (other.is_matrix) {
                out << (i == 0 ? "[" : " [");
            }
            for (size_t j = 0; j < col; j++) {
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
                out << std::endl;
            }
        }
        out << ']' << std::flush;
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
};
