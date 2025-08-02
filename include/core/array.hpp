#pragma once
#include "../libs/broadcasting.hpp"
#include "../libs/traits.hpp"

template <typename V>
class numcpp::array {
    static constexpr size_t none = ULONG_MAX;
    std::shared_ptr<std::vector<V>> data;
    size_t row, col, offset, row_stride, col_stride;
    const array* _base;
    const bool is_matrix;

    array(std::shared_ptr<std::vector<V>> data, const size_t rows, const size_t cols, const size_t offset = 0,
          const size_t row_stride = none, const size_t col_stride = 1, const array* base = nullptr,
          const bool is_matrix = true) noexcept :
        data(std::move(data)), row(rows), col(cols), offset(offset), row_stride(row_stride == none ? cols : row_stride),
        col_stride(col_stride), _base(base), is_matrix(is_matrix) {}

    bool is_contiguous() const { return row_stride == col && col_stride == 1; }

public:
    array(const std::initializer_list<V>& list, const size_t rows = 1, const size_t cols = none) :
        array(std::make_shared<std::vector<V>>(list), rows, cols == none ? list.size() : cols, 0, cols, 1, nullptr,
              rows > 1 && (cols == none ? list.size() : cols) > 1) {}

    array(const std::initializer_list<std::initializer_list<V>>& lists) :
        row(lists.size()), col(lists.begin()->size()), offset(0), row_stride(col), col_stride(1), _base(nullptr),
        is_matrix(row > 1 && col > 1) {
        std::vector<V> res;
        res.reserve(row * col);

        for (const std::initializer_list<V>& list : lists) {
            if (list.size() != col) {
                throw std::invalid_argument("Inconsistent row sizes");
            }
            res.insert(res.end(), list.begin(), list.end());
        }
        data = std::make_shared<std::vector<V>>(std::move(res));
    }

    array(std::vector<V> list, const size_t rows = 1, const size_t cols = none) :
        array(std::make_shared<std::vector<V>>(std::move(list)), rows, cols, 0, cols == none ? list.size() : cols, 1,
              nullptr, rows > 1 && (cols == none ? list.size() : cols) > 1) {
        if (data->size() != row * col) {
            throw std::invalid_argument("rows and columns don't match the array size");
        }
    }

    array(std::vector<std::vector<V>> lists) :
        row(lists.size()), col(lists.begin()->size()), offset(0), row_stride(col), col_stride(1), _base(nullptr),
        is_matrix(row > 1 && col > 1) {
        std::vector<V> res;
        res.reserve(row * col);

        for (std::vector<V>& list : lists) {
            if (list.size() != col) {
                throw std::invalid_argument("Inconsistent row sizes");
            }
            res.insert(res.end(), std::make_move_iterator(list.begin()), std::make_move_iterator(list.end()));
        }
        data = std::make_shared<std::vector<V>>(res);
    }

    const array* base() const noexcept { return _base; }

    size_t ndim() const noexcept { return row == 1 || col == 1 ? 1 : 2; }

    size_t size() const noexcept { return row * col; }

    auto real() const {
        if constexpr (is_complex_v<V>) {
            using T = typename V::value_type;
            std::vector<T> real_data;
            real_data.reserve(size());

            for (size_t i = 0; i < row; i++) {
                for (size_t j = 0; j < col; j++) {
                    real_data.push_back((*this)[{i, j}].real());
                }
            }
            return array<T>(std::move(real_data), row, col);
        } else {
            return *this;
        }
    }

    template <typename T>
    void real(const array<T>& other) {
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
                val = V(other[{bi, bj}], val.imag());
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

    std::pair<size_t, size_t> shape() const noexcept { return {row, col}; }

    V& operator[](std::pair<ll_t, ll_t>);
    const V& operator[](std::pair<ll_t, ll_t>) const;
    array operator[](const index&) const;

    friend std::ostream& operator<<(std::ostream& out, const array& other) {
        auto [row, col] = other.shape();
        const bool is_col_vector = (col == 1), is_matrix = other.is_matrix;
        const size_t width_dim = is_col_vector ? row : col;
        std::vector<size_t> col_width(width_dim, 0);

        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                const size_t pos = is_col_vector ? i : j;
                std::ostringstream ss;
                ss << format(other[{i, j}]);
                col_width[pos] = std::max(col_width[pos], ss.str().size());
            }
        }
        out << '[';

        for (size_t i = 0; i < row; i++) {
            if (is_matrix) {
                out << (i == 0 ? "[" : " [");
            }
            for (size_t j = 0; j < col; j++) {
                if (j > 0 || !is_matrix) {
                    out << ' ';
                }
                std::ostringstream ss;
                ss << format(other[{i, j}]);
                out << std::setw(col_width[is_col_vector ? i : j]) << ss.str();
            }
            if (is_matrix) {
                out << "]";
            }

            if (is_matrix && i < row - 1) {
                out << std::endl;
            }
        }
        out << ']' << std::flush;
        return out;
    }

    array T() const { return array(data, col, row, offset, col_stride, row_stride, _base ? _base : this, is_matrix); }

    array reshape(const size_t rows, const size_t cols) const {
        if (rows * cols != size()) {
            throw std::invalid_argument("reshape size mismatch");
        }
        if (!is_contiguous()) {
            throw std::runtime_error("cannot reshape a non-contiguous array");
        }
        return array(data, rows, cols, offset, cols, 1, _base ? _base : this, rows > 1 && cols > 1);
    }
};

#include "indexing.hpp"
