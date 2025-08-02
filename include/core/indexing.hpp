#pragma once

class numcpp::slice {
    bool resolved = false;

public:
    static constexpr ll_t none = LONG_LONG_MAX;
    ll_t start, stop, step;

    explicit slice(const ll_t start = none, const ll_t stop = none, const ll_t step = 1) :
        start(start), stop(stop), step(step) {
        if (step == 0) {
            throw std::invalid_argument("slice step cannot be zero");
        }
    }

    void resolve_bounds(const size_t dim) noexcept {
        if (start < 0) {
            start += dim;
        }
        if (stop < 0) {
            stop += dim;
        }
        if (start == none) {
            start = (step > 0) ? 0 : dim - 1;
        }
        if (stop == none) {
            stop = (step > 0) ? dim : -1;
        }
        start = std::clamp<ll_t>(start, 0ll, dim);
        stop = std::clamp<ll_t>(stop, -1ll, dim);
        resolved = true;
    }

    size_t size(const size_t dim) noexcept {
        if (!resolved) {
            resolve_bounds(dim);
        }
        if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) {
            return 0;
        }
        return (std::abs(stop - start) + std::abs(step) - 1) / std::abs(step);
    }
};

template <typename V>
struct numcpp::range {
    class iterator {
        V current, stop, step;

    public:
        using value_type = V;
        using difference_type = std::ptrdiff_t;
        using reference = const V&;
        using pointer = const V*;
        using iterator_category = std::forward_iterator_tag;

        iterator(V current, V stop, V step) noexcept : current(current), stop(stop), step(step) {}

        V operator*() const noexcept { return current; }

        iterator& operator++() {
            current += step;
            return *this;
        }

        iterator operator++(int) {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const iterator& other) const noexcept { return current == other.current; }

        bool operator!=(const iterator& other) const noexcept { return !(*this == other); }
    };

    V start, stop, step;

    range(const V stop) : range(0, stop) {}

    range(const V start, const V stop, const V step = 1) : start(start), stop(stop), step(step) {
        if (step == V(0)) {
            throw std::invalid_argument("range step cannot be zero");
        }
    }

    size_t size() const {
        if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) {
            return 0;
        }
        if constexpr (std::is_integral_v<V>) {
            return (stop - start + (step > 0 ? step - 1 : step + 1)) / step;
        } else {
            return static_cast<size_t>(std::ceil((stop - start) / step));
        }
    }
    iterator begin() const { return iterator(start, stop, step); }
    iterator end() const { return iterator(start + step * static_cast<V>(size()), stop, step); }
};

struct numcpp::index {
    std::variant<ll_t, slice> row, col;

    index(ll_t i, const slice& j = slice()) : row(i), col(j) {}
    index(const slice& i, ll_t j) : row(i), col(j) {}
    index(const slice& i, const slice& j = slice()) : row(i), col(j) {}
};

namespace numcpp {
    template <typename V>
    const V& array<V>::operator[](std::pair<ll_t, ll_t> idx) const {
        if (idx.first < 0) {
            idx.first += row;
        }
        if (idx.second < 0) {
            idx.second += col;
        }
        if (idx.first < 0 || idx.first >= row || idx.second < 0 || idx.second >= col) {
            throw std::out_of_range("Index out of bounds");
        }
        return (*data)[offset + idx.first * row_stride + idx.second * col_stride];
    }

    template <typename V>
    V& array<V>::operator[](std::pair<ll_t, ll_t> idx) {
        return const_cast<V&>(static_cast<const array&>(*this)[idx]);
    }

    template <typename V>
    array<V> array<V>::operator[](const index& idx) const {
        const bool row_is_slice = std::holds_alternative<slice>(idx.row);
        const bool col_is_slice = std::holds_alternative<slice>(idx.col);

        if (!row_is_slice && col_is_slice) {
            ll_t i = std::get<ll_t>(idx.row);
            auto cols = std::get<slice>(idx.col);

            if (i < 0) {
                i += row;
            }
            if (i < 0 || i >= row) {
                throw std::out_of_range("Row index out of bounds");
            }
            cols.resolve_bounds(col);
            return array(data, 1, cols.size(col), offset + i * row_stride + cols.start * col_stride, row_stride,
                         col_stride * cols.step, _base ? _base : this, false);
        }
        if (row_is_slice && !col_is_slice) {
            auto rows = std::get<slice>(idx.row);
            ll_t j = std::get<ll_t>(idx.col);

            if (j < 0) {
                j += col;
            }
            if (j < 0 || j >= col) {
                throw std::out_of_range("Column index out of bounds");
            }
            rows.resolve_bounds(row);
            return array(data, rows.size(row), 1, offset + rows.start * row_stride + j * col_stride,
                         row_stride * rows.step, col_stride, _base ? _base : this, false);
        }
        {
            auto rows = std::get<slice>(idx.row), cols = std::get<slice>(idx.col);
            rows.resolve_bounds(row);
            cols.resolve_bounds(col);
            return array(data, rows.size(row), cols.size(col),
                         offset + rows.start * row_stride + cols.start * col_stride, row_stride * rows.step,
                         col_stride * cols.step, _base ? _base : this, true);
        }
    }


} // namespace numcpp
