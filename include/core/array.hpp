#pragma once
#include "../libs/broadcasting.hpp"
#include "../libs/none.hpp"
#include "../libs/traits.hpp"

template <typename V>
class numcpp::array {
    std::shared_ptr<V[]> data;
    size_t row, col, offset, row_stride, col_stride;
    const array* _base;
    bool is_matrix;

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
    }

    array(std::shared_ptr<V[]> data, const size_t rows, const size_t cols, const size_t offset = 0,
          const size_t row_stride = none::size, const size_t col_stride = 1, array* base = nullptr,
          bool is_matrix = true) noexcept :
        data(std::move(data)), row(rows), col(cols), offset(offset),
        row_stride(row_stride == none::size ? cols : row_stride), col_stride(col_stride), _base(base),
        is_matrix(is_matrix) {}

    bool is_contiguous() const { return row_stride == col && col_stride == 1; }

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


public:
    array() = delete;

    array(const array& other) :
        data(other.data), row(other.row), col(other.col), offset(other.offset), row_stride(other.row_stride),
        col_stride(other.col_stride), _base(other._base ? other._base : &other), is_matrix(other.is_matrix) {}

    array(array&& other) noexcept = default;

    array(const std::initializer_list<V>& list, const size_t rows = 1, size_t cols = none::size) :
        row(rows), col(cols), offset(0), row_stride(col), col_stride(1), _base(nullptr), is_matrix(row > 1 && col > 1) {
        if (cols == none::size) {
            cols = list.size();
        }
        if (row * col != list.size()) {
            throw std::invalid_argument("Initializer list size doesn't match rows * cols");
        }
        data = std::shared_ptr<V[]>(new V[row * col]);
        std::copy(list.begin(), list.end(), data.get());
    }

    array(const std::initializer_list<std::initializer_list<V>>& lists) :
        row(lists.size()), col(lists.begin()->size()), offset(0), row_stride(col), col_stride(1), _base(nullptr),
        is_matrix(row > 1 && col > 1) {
        for (const std::initializer_list<V>& list : lists) {
            if (list.size() != col) {
                throw std::invalid_argument("Inconsistent list sizes");
            }
        }
        data = std::shared_ptr<V[]>(new V[row * col]);
        size_t index = 0;

        for (const std::initializer_list<V>& list : lists) {
            for (const V& val : list) {
                data[index++] = val;
            }
        }
    }

    // Construct from std::vector<V>
    array(std::vector<V> list, const size_t rows = 1, size_t cols = none::size) :
        row(rows), col(cols), offset(0), row_stride(col), col_stride(1), _base(nullptr), is_matrix(row > 1 && col > 1) {
        if (cols == none::size) {
            cols = list.size();
        }
        if (row * col != list.size()) {
            throw std::invalid_argument("Vector size doesn't match rows * cols");
        }
        data = std::shared_ptr<V[]>(new V[row * col]);
        std::move(list.begin(), list.end(), data.get());
    }

    array(std::vector<std::vector<V>> lists) :
        row(lists.size()), col(lists.empty() ? 0 : lists.front().size()), offset(0), row_stride(col), col_stride(1),
        _base(nullptr), is_matrix(row > 1 && col > 1) {
        for (const std::vector<V>& list : lists) {
            if (list.size() != col) {
                throw std::invalid_argument("Inconsistent inner vector sizes");
            }
        }
        data = std::shared_ptr<V[]>(new V[row * col]);
        size_t index = 0;

        for (std::vector<V>& list : lists) {
            std::move(list.begin(), list.end(), data.get() + index);
            index += col;
        }
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

    const array* base() const noexcept { return _base; }

    size_t ndim() const noexcept { return row > 1 && col > 1 ? 2 : 1; }

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

    array& operator=(array other) noexcept {
        swap(other);
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& out, const array& other) {
        auto [row, col] = other.shape();
        const bool is_col_vector = (col == 1), is_matrix = other.is_matrix;
        const size_t width_dim = is_col_vector ? col : row;
        size_t col_width = 0;
        std::vector<size_t> col_width_vec;

        if (is_matrix) {
            col_width_vec = std::vector(width_dim, 0ul);
        }
        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                const size_t size = format(other[{i, j}]).size();

                if (is_matrix) {
                    const size_t pos = is_col_vector ? j : i;
                    col_width_vec[pos] = std::max(col_width_vec[pos], size);
                } else {
                    col_width = std::max(col_width, size);
                }
            }
        }
        out << '[';

        for (size_t i = 0; i < row; i++) {
            if (is_matrix) {
                out << (i == 0 ? "[" : " [");
            }
            for (size_t j = 0; j < col; j++) {
                if (j > 0 || (!is_matrix && i > 0)) {
                    out << ' ';
                }
                if (is_matrix) {
                    out << std::setw(col_width_vec[is_col_vector ? j : i]);
                } else {
                    out << std::setw(col_width);
                }
                out << format(other[{i, j}]);
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
