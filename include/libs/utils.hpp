#pragma once

namespace numcpp {

    template <typename T>
    array<T> fill(const shape_t& shape, const T& value) {
        const size_t size = shape.size();
        buffer_t<T> buf(size);
        std::fill_n(buf.data(), size, value);
        return array<T>(std::move(buf), shape);
    }

    template <typename T = float64_t>
    array<T> ones(const shape_t& shape) {
        return fill(shape, T(1));
    }

    template <typename T = float64_t>
    array<T> empty(const shape_t& shape) {
        return array<T>(buffer_t<T>(shape.size()), shape);
    }

    template <typename T = float64_t>
    array<T> zeros(const shape_t& shape) {
        return fill(shape, T());
    }

    template <typename T>
    buffer_t<T> buffer(const array<T>& arr) noexcept {
        return arr.buffer;
    }
    template <typename T>
    size_t offset(const array<T>& arr) noexcept {
        return arr.offset;
    }
    template <typename T>
    const void* base(const array<T>& arr) noexcept {
        return arr.base;
    }
    template <typename T>
    bool is_matrix(const array<T>& arr) noexcept {
        return arr.is_matrix;
    }
    template <typename T>
    bool is_scalar(const array<T>& arr) noexcept {
        return arr.is_scalar;
    }
    template <typename T>
    bool is_assignable(const array<T>& arr) noexcept {
        return arr.is_assignable;
    }
} // namespace numcpp
