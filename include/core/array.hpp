#pragma once

template <typename T>
class numcpp::array {
    buffer_t<T> buffer = buffer_t<T>();
    size_t row = 0, col = 0, offset = 0, row_stride = 0, col_stride = 0;
    const void* base = none::base;
    bool is_matrix = false, is_scalar = false, is_assignable = false;

    template <typename>
    friend class array;
    template <typename, typename>
    class base_iterator;

    array flat_constructor(auto begin, auto end, shape_t shape) {
        if (shape.cols == none::size) {
            shape.cols = end - begin;
        }
        if (shape.size() != end - begin) {
            throw std::invalid_argument("Size mismatch in flat constructor");
        }
        buffer = buffer_t<T>(shape.size());
        std::copy(begin, end, buffer.data());
        return array(std::move(buffer), shape, 0, shape.cols, 1, none::base, shape.rows > 1 && shape.cols > 1, false, false);
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

    array(buffer_t<T> data, const shape_t& shape, const size_t offset, const size_t row_stride, const size_t col_stride, const void* base,
          const bool is_matrix, const bool is_scalar, const bool is_assignable) noexcept :
        buffer(std::move(data)), row(shape.rows), col(shape.cols), offset(offset), row_stride(row_stride == none::size ? shape.rows : row_stride),
        col_stride(col_stride), base(base), is_matrix(is_matrix), is_scalar(is_scalar), is_assignable(is_assignable) {}

public:
    using iterator = base_iterator<T*, T&>;
    using const_iterator = base_iterator<const T*, const T&>;

    array() noexcept = default;

    array(const array& other) noexcept :
        array(other.buffer, {other.row, other.col}, other.offset, other.row_stride, other.col_stride, other.base ? other.base : &other,
              other.is_matrix, other.is_scalar, false) {}

    array(array&& other) noexcept :
        array(std::move(other.buffer), {other.row, other.col}, other.offset, other.row_stride, other.col_stride, other.base, other.is_matrix,
              other.is_scalar, other.is_assignable) {
        other.row = other.col = other.offset = other.row_stride = other.col_stride = 0;
        other.base = none::base;
        other.is_matrix = other.is_scalar = other.is_assignable = false;
    }

    array(std::initializer_list<T> list, const shape_t& shape = none::shape) { *this = flat_constructor(list.begin(), list.end(), shape); }

    array(const std::vector<T>& list, const shape_t& shape = none::shape) { *this = flat_constructor(list.begin(), list.end(), shape); }

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
        row(lists.size()), col(lists.empty() ? 0 : lists.front().size()), row_stride(col), col_stride(1), is_matrix(row > 1 && col > 1) {
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
            std::copy_n(buf.data(), buffer.size, buffer.data());
        } else {
            buffer = buf;
        }
    }

    array(buffer_t<T>&& buf, const shape_t& shape) : array(buf, shape, false) {}

    array(const T* list, const shape_t& shape, const bool copy = true) :
        array(copy ? buffer_t<T>(const_cast<T*>(list), shape.size()) : buffer_t<T>(const_cast<T*>(list), shape.size(), nullptr), shape) {}

    array(const T& value) : array(&value, {1, 1}) { is_scalar = true; }

    size_t ndim() const noexcept { return row > 1 && col > 1 ? 2 : 1; }

    size_t size() const noexcept { return row * col; }

    array<real_t<T>> real() noexcept requires(is_numeric_v<T>)
    {
        using V = real_t<T>;

        if constexpr (is_complex_v<T>) {
            return array<V>(buffer_t<V>(reinterpret_cast<V*>(buffer.data()), size() * 2, nullptr), {row, col}, offset * 2, row_stride * 2,
                            col_stride * 2, this, is_matrix, row == 1 && col == 1, true);
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
            return array<V>(buffer_t<V>(reinterpret_cast<V*>(buffer.data()), size() * 2, nullptr), {row, col}, offset * 2 + 1, row_stride * 2,
                            col_stride * 2, this, is_matrix, row == 1 && col == 1, true);
        } else {
            return zeros<V>(shape());
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

            if (lhs_shape != res_shape) {
                throw std::runtime_error("Broadcasted shape doesn't match array shape.");
            }
            size_t k = 0;

            for (ll_t i = 0; i < row; i++) {
                for (ll_t j = 0; j < col; j++) {
                    buffer[k++] = other[broadcast_index({i, j}, rhs_shape)];
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

    template <typename V>
    friend std::ostream& operator<<(std::ostream&, const array<V>&);

    constexpr operator const T&() const {
        if (is_scalar || size() == 1) {
            return buffer[offset];
        }
        throw std::invalid_argument("illegal scalar conversion of an array");
    }
    constexpr operator T&() noexcept { return const_cast<T&>(static_cast<const T&>(static_cast<const array&>(*this))); }
    constexpr operator bool() const noexcept requires(!std::is_same_v<T, bool>)
    {
        return static_cast<T>(*this);
    }

    array reshape(const shape_t& shape) const {
        if (shape.size() != size()) {
            throw std::invalid_argument("reshape size mismatch");
        }
        if (row_stride != col || col_stride != 1) {
            throw std::runtime_error("cannot reshape a non-contiguous array");
        }
        return array(buffer, shape, offset, shape.cols, 1, base ? base : this, shape.rows > 1 && shape.cols > 1, false, false);
    }

    array copy() const noexcept { return array(buffer, shape()); }

    template <typename V>
    friend constexpr size_t offset(const array<V>&) noexcept;
    template <typename V>
    friend constexpr buffer_t<V> buffer(const array<V>&) noexcept;
    template <typename V>
    friend constexpr const void* base(const array<V>&) noexcept;
    template <typename V>
    friend constexpr bool is_matrix(const array<V>&) noexcept;
    template <typename V>
    friend constexpr bool is_scalar(const array<V>&) noexcept;
    template <typename V>
    friend constexpr bool is_assignable(const array<V>&) noexcept;

    iterator begin() noexcept { return iterator(buffer.data() + offset, row, col, row_stride, col_stride, 0); }
    iterator end() noexcept { return iterator(buffer.data() + offset, row, col, row_stride, col_stride, row * col); }
    const_iterator begin() const noexcept { return const_iterator(buffer.data() + offset, row, col, row_stride, col_stride, 0); }
    const_iterator end() const noexcept { return const_iterator(buffer.data() + offset, row, col, row_stride, col_stride, row * col); }
};

template <typename T>
template <typename Ptr, typename Ref>
class numcpp::array<T>::base_iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::remove_const_t<T>;
    using difference_type = std::ptrdiff_t;
    using pointer = Ptr;
    using reference = Ref;

private:
    pointer ptr;
    difference_type index = 0;
    difference_type rows, cols, row_stride, col_stride;

    constexpr pointer compute() const noexcept { return ptr + index / cols * row_stride + index % cols * col_stride; }

public:
    constexpr explicit base_iterator(const pointer base_ptr, const difference_type rows, const difference_type cols, const difference_type row_stride,
                                     const difference_type col_stride, const difference_type start_index = 0) noexcept :
        ptr(base_ptr), index(start_index), rows(rows), cols(cols), row_stride(row_stride), col_stride(col_stride) {}

    template <typename P, typename R>
    constexpr base_iterator(const base_iterator<P, R>& other) noexcept :
        ptr(other.ptr), index(other.index), rows(other.rows), cols(other.cols), row_stride(other.row_stride), col_stride(other.col_stride) {}


    constexpr reference operator*() const noexcept { return *compute(); }
    constexpr pointer operator->() const noexcept { return compute(); }
    constexpr reference operator[](difference_type n) const noexcept { return *(*this + n); }

    constexpr base_iterator& operator++() noexcept {
        ++index;
        return *this;
    }
    constexpr base_iterator operator++(int) noexcept {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }
    constexpr base_iterator& operator--() noexcept {
        --index;
        return *this;
    }
    constexpr base_iterator operator--(int) noexcept {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    constexpr base_iterator operator+(difference_type n) const noexcept {
        auto tmp = *this;
        tmp.index += n;
        return tmp;
    }
    constexpr base_iterator& operator+=(difference_type n) noexcept {
        index += n;
        return *this;
    }
    friend constexpr base_iterator operator+(difference_type n, const base_iterator& it) noexcept { return it + n; }

    constexpr base_iterator operator-(difference_type n) const noexcept {
        auto tmp = *this;
        tmp.index -= n;
        return tmp;
    }
    constexpr base_iterator& operator-=(difference_type n) noexcept {
        index -= n;
        return *this;
    }
    constexpr difference_type operator-(const base_iterator& other) const noexcept { return index - other.index; }

    constexpr bool operator==(const base_iterator& other) const noexcept { return index == other.index; }
    constexpr bool operator!=(const base_iterator& other) const noexcept { return index != other.index; }
    constexpr bool operator<(const base_iterator& other) const noexcept { return index < other.index; }
    constexpr bool operator>(const base_iterator& other) const noexcept { return index > other.index; }
    constexpr bool operator<=(const base_iterator& other) const noexcept { return index <= other.index; }
    constexpr bool operator>=(const base_iterator& other) const noexcept { return index >= other.index; }
};
