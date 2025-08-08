#pragma once
#include "traits.hpp"

namespace numcpp {
    template <typename T>
    class array;
    class MaskedArray;

    class axis_t {
        enum class type_t { none, single, pair } type;
        int axis0, axis1;

    public:
        axis_t() noexcept : axis_t(none_t()) {}
        axis_t(none_t<>) noexcept : type(type_t::none), axis0(-1), axis1(-1) {}
        axis_t(const int axis) noexcept : type(type_t::single), axis0(axis), axis1(-1) {}
        axis_t(const int i, const int j) noexcept : type(type_t::pair), axis0(i), axis1(j) {}
        axis_t(const std::pair<int, int>& axes) noexcept : axis_t(axes.first, axes.second) {}

        bool is_none() const noexcept { return type == type_t::none; }
        bool is_single() const noexcept { return type == type_t::single; }
        bool is_pair() const noexcept { return type == type_t::pair; }

        int get_single() const noexcept { return axis0; }
        std::pair<int, int> get_pair() const noexcept { return {axis0, axis1}; }
    };

    namespace detail {
        template <typename T>
        std::string to_string(const T&);
    } // namespace detail

    template <typename T>
    struct complex_t {
        using value_type = T;
        T real, imag;

        constexpr complex_t() noexcept : complex_t(0, 0) {}
        constexpr complex_t(const T real) noexcept : complex_t(real, 0) {}
        constexpr complex_t(const T real, const T imag) noexcept : real(real), imag(imag) {}

        constexpr complex_t operator+() const noexcept { return *this; }
        constexpr complex_t operator-() const noexcept { return complex_t(-real, -imag); }

        constexpr complex_t operator+(const complex_t& other) const noexcept {
            return complex_t(real + other.real, imag + other.imag);
        }
        constexpr complex_t operator-(const complex_t& other) const noexcept {
            return complex_t(real - other.real, imag - other.imag);
        }
        constexpr complex_t operator*(const complex_t& other) const noexcept {
            return complex_t(real * other.real - imag * other.imag, real * other.imag + imag * other.real);
        }
        constexpr complex_t operator/(const complex_t& other) const {
            T denominator = other.real * other.real + other.imag * other.imag;

            if (!denominator) {
                throw std::invalid_argument("Division by Zero");
            }
            return complex_t((real * other.real + imag * other.imag) / denominator,
                             (imag * other.real - real * other.imag) / denominator);
        }

        constexpr complex_t operator+(const T value) const noexcept { return complex_t(real + value, imag); }
        constexpr complex_t operator-(const T value) const noexcept { return complex_t(real - value, imag); }
        constexpr complex_t operator*(const T value) const noexcept { return complex_t(real * value, imag * value); }
        constexpr complex_t operator/(const T value) const {
            if (value == T()) {
                throw std::invalid_argument("Division by Zero");
            }
            return complex_t(real / value, imag / value);
        }

        template <typename V>
        requires(!std::is_same_v<V, MaskedArray>)
        friend constexpr complex_t operator+(const V& value, const complex_t& comp) noexcept {
            return complex_t(value + comp.real, comp.imag);
        }
        template <typename V>
        requires(!std::is_same_v<V, MaskedArray>)
        friend constexpr complex_t operator-(const V& value, const complex_t& comp) noexcept {
            return complex_t(value - comp.real, -comp.imag);
        }
        template <typename V>
        requires(!std::is_same_v<V, MaskedArray>)
        friend constexpr complex_t operator*(const V& value, const complex_t& comp) noexcept {
            return complex_t(value * comp.real, value * comp.imag);
        }
        template <typename V>
        requires(!std::is_same_v<V, MaskedArray>)
        friend constexpr complex_t operator/(const V& value, const complex_t& comp) {
            if (value == V()) {
                throw std::invalid_argument("Division by Zero");
            }
            return complex_t(value / comp.real, value / comp.imag);
        }

        complex_t& operator+=(const complex_t& other) noexcept {
            real += other.real;
            imag += other.imag;
            return *this;
        }
        complex_t& operator-=(const complex_t& other) noexcept {
            real -= other.real;
            imag -= other.imag;
            return *this;
        }
        complex_t& operator*=(const complex_t& other) noexcept {
            T r = real * other.real - imag * other.imag, i = real * other.imag + imag * other.real;
            real = r;
            imag = i;
            return *this;
        }
        complex_t& operator/=(const complex_t& other) {
            T denominator = other.real * other.real + other.imag * other.imag;

            if (!denominator) {
                throw std::invalid_argument("Division by Zero");
            }
            T r = (real * other.real + imag * other.imag) / denominator;
            T i = (imag * other.real - real * other.imag) / denominator;
            real = r;
            imag = i;
            return *this;
        }

        complex_t& operator+=(const T value) noexcept {
            real += value;
            return *this;
        }
        complex_t& operator-=(const T value) noexcept {
            real -= value;
            return *this;
        }
        complex_t& operator*=(T value) noexcept {
            real *= value;
            imag *= value;
            return *this;
        }
        complex_t& operator/=(T value) {
            if (value == T()) {
                throw std::invalid_argument("Division by Zero");
            }
            real /= value;
            imag /= value;
            return *this;
        }

        constexpr bool operator==(const complex_t& other) const noexcept {
            return real == other.real && imag == other.imag;
        }
        constexpr bool operator!=(const complex_t& other) const noexcept { return !(*this == other); }
        constexpr operator bool() const noexcept { return real != 0 || imag != 0; }

        T abs() const noexcept { return std::sqrt(norm()); }
        constexpr complex_t conj() const noexcept { return complex_t(real, -imag); }
        constexpr T norm() const noexcept { return real * real + imag * imag; }
        std::complex<T> to_std() const noexcept { return std::complex<T>(real, imag); }

        friend std::ostream& operator<<(std::ostream& out, const complex_t& complex) noexcept {
            return out << detail::to_string(complex);
        }

        static complex_t from_polar(const T magnitude, const T angle_rad) noexcept {
            return complex_t(magnitude * std::cos(angle_rad), magnitude * std::sin(angle_rad));
        }
    };

    class dtype::bitref_t {
        uint8_t& byte;
        uint8_t bit;

    public:
        bitref_t() noexcept = delete;
        bitref_t(uint8_t& byte, const uint8_t bit) noexcept : byte(byte), bit(bit) {}

        operator bool() const noexcept { return byte >> bit & 1; }
        bitref_t& operator=(const bitref_t& other) noexcept { return *this = static_cast<bool>(other); }

        bitref_t& operator=(const bool val) noexcept {
            if (val) {
                byte |= 1 << bit;
            } else {
                byte &= ~(1 << bit);
            }
            return *this;
        }

        bool operator~() const noexcept { return !static_cast<bool>(*this); }
        bool operator&(const bool b) const noexcept { return static_cast<bool>(*this) & b; }
        bool operator|(const bool b) const noexcept { return static_cast<bool>(*this) | b; }
        bool operator^(const bool b) const noexcept { return static_cast<bool>(*this) ^ b; }
        bool operator&(const bitref_t& other) const noexcept { return *this & static_cast<bool>(other); }
        bool operator|(const bitref_t& other) const noexcept { return *this | static_cast<bool>(other); }
        bool operator^(const bitref_t& other) const noexcept { return *this ^ static_cast<bool>(other); }

        bitref_t& operator&=(const bool b) noexcept { return *this = *this & b; }
        bitref_t& operator|=(const bool b) noexcept { return *this = *this | b; }
        bitref_t& operator^=(const bool b) noexcept { return *this = *this ^ b; }
        bitref_t& operator&=(const bitref_t& other) noexcept { return *this &= static_cast<bool>(other); }
        bitref_t& operator|=(const bitref_t& other) noexcept { return *this |= static_cast<bool>(other); }
        bitref_t& operator^=(const bitref_t& other) noexcept { return *this ^= static_cast<bool>(other); }

        bool operator==(const bool b) const noexcept { return static_cast<bool>(*this) == b; }
        bool operator!=(const bool b) const noexcept { return static_cast<bool>(*this) != b; }
        bool operator==(const bitref_t& other) const noexcept { return *this == static_cast<bool>(other); }
        bool operator!=(const bitref_t& other) const noexcept { return *this != static_cast<bool>(other); }
    };

    struct dtype::bool_t {
        uint8_t value = 0;

        bool_t() noexcept = default;
        explicit bool_t(const uint8_t val) noexcept : value(val) {}

        bitref_t operator[](int8_t i) {
            if (i < 0) {
                i += 8;
            }
            if (i < 0 || i >= 8) {
                throw std::out_of_range("bit index out of range");
            }
            return bitref_t(value, i);
        }
        bool operator[](int8_t i) const {
            if (i < 0) {
                i += 8;
            }
            if (i < 0 || i >= 8) {
                throw std::out_of_range("bit index out of range");
            }
            return (value >> i) & 1;
        }

        bool_t operator~() const noexcept { return bool_t(~value); }
        bool_t operator&(const bool b) const noexcept { return bool_t(value & static_cast<uint8_t>(b)); }
        bool_t operator|(const bool b) const noexcept { return bool_t(value | static_cast<uint8_t>(b)); }
        bool_t operator^(const bool b) const noexcept { return bool_t(value ^ static_cast<uint8_t>(b)); }

        bool_t& operator&=(const bool b) noexcept {
            value &= static_cast<uint8_t>(b);
            return *this;
        }
        bool_t& operator|=(const bool b) noexcept {
            value |= static_cast<uint8_t>(b);
            return *this;
        }
        bool_t& operator^=(const bool b) noexcept {
            value ^= static_cast<uint8_t>(b);
            return *this;
        }

        bool_t operator<<(const int shift) const noexcept { return bool_t(value << shift); }
        bool_t operator>>(const int shift) const noexcept { return bool_t(value >> shift); }

        bool_t& operator<<=(const int shift) noexcept {
            value <<= shift;
            return *this;
        }
        bool_t& operator>>=(const int shift) noexcept {
            value >>= shift;
            return *this;
        }

        bool operator==(const int other) const noexcept { return value == static_cast<uint8_t>(other); }
        bool operator!=(const int other) const noexcept { return value != static_cast<uint8_t>(other); }
    };

    template <typename T>
    class buffer_t {
        std::shared_ptr<T[]> value;

    public:
        size_t size = 0;

        buffer_t() noexcept = default;
        buffer_t(const buffer_t&) noexcept = default;
        buffer_t(buffer_t&&) noexcept = default;
        explicit buffer_t(const size_t n) : value(new T[n](), std::default_delete<T[]>()), size(n) {}
        buffer_t(T* ptr, const size_t n) : value(ptr, std::default_delete<T[]>()), size(n) {}
        buffer_t(std::shared_ptr<T[]> ptr, const size_t n) noexcept : value(std::move(ptr)), size(n) {}
        buffer_t(T* raw_ptr, const size_t n, std::nullptr_t) noexcept : value(raw_ptr, [](T*) {}), size(n) {}

        buffer_t& operator=(const buffer_t&) noexcept = default;
        buffer_t& operator=(buffer_t&&) noexcept = default;
        T& operator[](const size_t i) noexcept { return value[i]; }
        const T& operator[](const size_t i) const noexcept { return value[i]; }
        operator bool() const noexcept { return static_cast<bool>(value); }

        T* data() noexcept { return value.get(); }
        const T* data() const noexcept { return value.get(); }

        void reset() noexcept {
            value.reset();
            size = 0;
        }

        void reset(const T* ptr, const size_t n) noexcept {
            value.reset(ptr);
            size = n;
        }
    };

    class slice_t {
        bool resolved = false;

    public:
        static constexpr ll_t none = none_t<ll_t>();
        ll_t start, stop, step;

        explicit slice_t(const ll_t start = none, const ll_t stop = none, const ll_t step = 1) :
            start(start), stop(stop), step(step) {
            if (step == 0) {
                throw std::invalid_argument("slice step cannot be zero");
            }
        }

        void resolve(const size_t dim) noexcept {
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
                resolve(dim);
            }
            if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) {
                return 0;
            }
            return (std::abs(stop - start) + std::abs(step) - 1) / std::abs(step);
        }
    };

    struct index_t {
        const std::variant<ll_t, slice_t> row, col;

        index_t(const ll_t i, const ll_t j) noexcept : row(i), col(j) {}
        index_t(const ll_t i, const slice_t& j = slice_t()) noexcept : row(i), col(j) {}
        index_t(const slice_t& i, const ll_t j) noexcept : row(i), col(j) {}
        index_t(const slice_t& i, const slice_t& j = slice_t()) noexcept : row(i), col(j) {}

        bool is_scalar_row() const noexcept { return std::holds_alternative<ll_t>(row); }
        bool is_scalar_col() const noexcept { return std::holds_alternative<ll_t>(col); }
        bool is_scalar() const noexcept { return is_scalar_row() && is_scalar_col(); }
        bool is_slice_row() const noexcept { return std::holds_alternative<slice_t>(row); }
        bool is_slice_col() const noexcept { return std::holds_alternative<slice_t>(col); }
        bool is_slice() const noexcept { return is_slice_row() && is_slice_col(); }

        ll_t get_scalar_row() const noexcept { return std::get<ll_t>(row); }
        ll_t get_scalar_col() const noexcept { return std::get<ll_t>(col); }
        std::pair<ll_t, ll_t> get_scalar() const noexcept { return {get_scalar_row(), get_scalar_col()}; }
        const slice_t& get_slice_row() const noexcept { return std::get<slice_t>(row); }
        const slice_t& get_slice_col() const noexcept { return std::get<slice_t>(col); }
        std::pair<slice_t, slice_t> get_slice() const noexcept { return {get_slice_row(), get_slice_col()}; }
    };

    struct shape_t {
        size_t rows, cols;

        shape_t() noexcept : rows(1), cols(1) {}
        shape_t(const size_t col) noexcept : rows(1), cols(col) {}
        shape_t(const size_t rows, const size_t cols) noexcept : rows(rows), cols(cols) {}

        bool operator==(const shape_t& shape) const = default;

        constexpr size_t size() const noexcept { return rows * cols; }

        friend std::ostream& operator<<(std::ostream& out, const shape_t& shape) {
            return out << '(' << shape.rows << ", " << shape.cols << ')';
        }
    };

    namespace none {
        inline constexpr int axis = none_t();
        inline constexpr auto base = nullptr, out = nullptr, where = nullptr;
        inline constexpr auto size = static_cast<size_t>(none_t<size_t>());
        inline auto shape = shape_t(1, size);
    } // namespace none

    template <typename T>
    struct out_t {
        array<T>* ptr;

        out_t() noexcept : ptr(nullptr) {}
        out_t(array<T>& arr) noexcept : ptr(&arr) {}
        out_t(std::nullptr_t) noexcept : ptr(nullptr) {}

        constexpr operator bool() const noexcept { return ptr != nullptr; }
        constexpr array<T>& operator*() { return *ptr; }
        constexpr array<T>* operator->() { return ptr; }
    };

    template <typename T>
    struct range_t {
        T start, stop, step;

        range_t(const T stop) : range_t(0, stop) {}

        range_t(const T start, const T stop, const T step = 1) : start(start), stop(stop), step(step) {
            if (step == T(0)) {
                throw std::invalid_argument("range step cannot be zero");
            }
        }

        buffer_t<T> evaluate() const noexcept {
            buffer_t<T> buf(size());
            size_t i = 0;

            for (T value = start; value < stop; value += step) {
                buf[i++];
            }
            return buf;
        }

        size_t size() const noexcept {
            if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) {
                return 0;
            }
            if constexpr (std::is_integral_v<T>) {
                return (stop - start + (step > 0 ? step - 1 : step + 1)) / step;
            } else {
                return static_cast<size_t>(std::ceil((stop - start) / step));
            }
        }
    };

    struct where_t {
        const MaskedArray* ptr;

        where_t() noexcept : ptr(nullptr) {}
        where_t(const MaskedArray& arr) noexcept : ptr(&arr) {}
        where_t(std::nullptr_t) noexcept : ptr(nullptr) {}

        constexpr operator bool() const noexcept { return ptr != nullptr; }
        constexpr const MaskedArray& operator*() const noexcept { return *ptr; }
        constexpr const MaskedArray* operator->() const noexcept { return ptr; }
    };
} // namespace numcpp
