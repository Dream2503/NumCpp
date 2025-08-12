#pragma once
#include "traits.hpp"

namespace numcpp {
    template <typename T>
    class array;
    class MaskedArray;

    constexpr size_t broadcast_index(size_t, size_t) noexcept;

    namespace detail {
        template <typename T>
        std::string to_string(const T&);
        template <typename T>
        T division_by_zero_warning(T, const char[]);
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
                return detail::division_by_zero_warning(*this, __PRETTY_FUNCTION__);
            }
            return complex_t((real * other.real + imag * other.imag) / denominator,
                             (imag * other.real - real * other.imag) / denominator);
        }

        constexpr complex_t operator+(const T value) const noexcept { return complex_t(real + value, imag); }
        constexpr complex_t operator-(const T value) const noexcept { return complex_t(real - value, imag); }
        constexpr complex_t operator*(const T value) const noexcept { return complex_t(real * value, imag * value); }
        constexpr complex_t operator/(const T value) const {
            if (value == T()) {
                return detail::division_by_zero_warning(*this, __PRETTY_FUNCTION__);
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
            if (comp == 0) {
                return detail::division_by_zero_warning(value, __PRETTY_FUNCTION__);
            }
            return complex_t(value / comp.real, value / comp.imag);
        }

        constexpr complex_t& operator+=(const complex_t& other) noexcept {
            real += other.real;
            imag += other.imag;
            return *this;
        }
        constexpr complex_t& operator-=(const complex_t& other) noexcept {
            real -= other.real;
            imag -= other.imag;
            return *this;
        }
        constexpr complex_t& operator*=(const complex_t& other) noexcept {
            T r = real * other.real - imag * other.imag, i = real * other.imag + imag * other.real;
            real = r;
            imag = i;
            return *this;
        }
        constexpr complex_t& operator/=(const complex_t& other) {
            T denominator = other.real * other.real + other.imag * other.imag;

            if (!denominator) {
                return detail::division_by_zero_warning(*this, __PRETTY_FUNCTION__);
            }
            T r = (real * other.real + imag * other.imag) / denominator;
            T i = (imag * other.real - real * other.imag) / denominator;
            real = r;
            imag = i;
            return *this;
        }

        constexpr complex_t& operator+=(const T value) noexcept {
            real += value;
            return *this;
        }
        constexpr complex_t& operator-=(const T value) noexcept {
            real -= value;
            return *this;
        }
        constexpr complex_t& operator*=(T value) noexcept {
            real *= value;
            imag *= value;
            return *this;
        }
        constexpr complex_t& operator/=(T value) {
            if (value == T()) {
                return detail::division_by_zero_warning(*this, __PRETTY_FUNCTION__);
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

        constexpr T abs() const noexcept { return std::sqrt(norm()); }
        constexpr complex_t conj() const noexcept { return complex_t(real, -imag); }
        constexpr T norm() const noexcept { return real * real + imag * imag; }
        constexpr std::complex<T> to_std() const noexcept { return std::complex<T>(real, imag); }

        friend std::ostream& operator<<(std::ostream& out, const complex_t& complex) noexcept {
            return out << detail::to_string(complex);
        }

        constexpr static complex_t from_polar(const T magnitude, const T angle_rad) noexcept {
            return complex_t(magnitude * std::cos(angle_rad), magnitude * std::sin(angle_rad));
        }
    };

    class dtypes::bitref_t {
        uint8_t& byte;
        uint8_t bit;

    public:
        constexpr bitref_t() noexcept = delete;
        constexpr bitref_t(uint8_t& byte, const uint8_t bit) noexcept : byte(byte), bit(bit) {}

        constexpr operator bool() const noexcept { return byte >> bit & 1; }
        constexpr bitref_t& operator=(const bitref_t& other) noexcept { return *this = static_cast<bool>(other); }

        constexpr bitref_t& operator=(const bool val) noexcept {
            if (val) {
                byte |= 1 << bit;
            } else {
                byte &= ~(1 << bit);
            }
            return *this;
        }

        constexpr bool operator~() const noexcept { return !static_cast<bool>(*this); }
        constexpr bool operator&(const bool b) const noexcept { return static_cast<bool>(*this) & b; }
        constexpr bool operator|(const bool b) const noexcept { return static_cast<bool>(*this) | b; }
        constexpr bool operator^(const bool b) const noexcept { return static_cast<bool>(*this) ^ b; }
        constexpr bool operator&(const bitref_t& other) const noexcept { return *this & static_cast<bool>(other); }
        constexpr bool operator|(const bitref_t& other) const noexcept { return *this | static_cast<bool>(other); }
        constexpr bool operator^(const bitref_t& other) const noexcept { return *this ^ static_cast<bool>(other); }

        constexpr bitref_t& operator&=(const bool b) noexcept { return *this = *this & b; }
        constexpr bitref_t& operator|=(const bool b) noexcept { return *this = *this | b; }
        constexpr bitref_t& operator^=(const bool b) noexcept { return *this = *this ^ b; }
        constexpr bitref_t& operator&=(const bitref_t& other) noexcept { return *this &= static_cast<bool>(other); }
        constexpr bitref_t& operator|=(const bitref_t& other) noexcept { return *this |= static_cast<bool>(other); }
        constexpr bitref_t& operator^=(const bitref_t& other) noexcept { return *this ^= static_cast<bool>(other); }

        constexpr bool operator==(const bool b) const noexcept { return static_cast<bool>(*this) == b; }
        constexpr bool operator!=(const bool b) const noexcept { return static_cast<bool>(*this) != b; }
        constexpr bool operator==(const bitref_t& other) const noexcept { return *this == static_cast<bool>(other); }
        constexpr bool operator!=(const bitref_t& other) const noexcept { return *this != static_cast<bool>(other); }
    };

    struct dtypes::bool_t {
        using value_type = uint8_t;
        uint8_t value = 0;

        constexpr bool_t() noexcept = default;
        constexpr bool_t(const uint8_t val) noexcept : value(val) {}
        constexpr bool_t(const std::initializer_list<bool> list) {
            if (list.size() > 8) {
                throw std::invalid_argument("bits size overflow");
            }
            size_t i = 0;

            for (const bool b : list) {
                value |= static_cast<uint8_t>(b) << i++;
            }
        }

        constexpr bitref_t operator[](int8_t i) {
            if (i < 0) {
                i += 8;
            }
            if (i < 0 || i >= 8) {
                throw std::out_of_range("bit index out of range");
            }
            return bitref_t(value, i);
        }
        constexpr bool operator[](int8_t i) const {
            if (i < 0) {
                i += 8;
            }
            if (i < 0 || i >= 8) {
                throw std::out_of_range("bit index out of range");
            }
            return (value >> i) & 1;
        }
        constexpr operator bool() const noexcept { return value & 1; };

        constexpr bool_t operator~() const noexcept { return bool_t(~value); }
        constexpr bool_t operator&(const bool b) const noexcept { return bool_t(value & static_cast<uint8_t>(b)); }
        constexpr bool_t operator|(const bool b) const noexcept { return bool_t(value | static_cast<uint8_t>(b)); }
        constexpr bool_t operator^(const bool b) const noexcept { return bool_t(value ^ static_cast<uint8_t>(b)); }

        constexpr bool_t& operator&=(const bool b) noexcept {
            value &= static_cast<uint8_t>(b);
            return *this;
        }
        constexpr bool_t& operator|=(const bool b) noexcept {
            value |= static_cast<uint8_t>(b);
            return *this;
        }
        constexpr bool_t& operator^=(const bool b) noexcept {
            value ^= static_cast<uint8_t>(b);
            return *this;
        }

        constexpr bool_t operator<<(const int shift) const noexcept { return bool_t(value << shift); }
        constexpr bool_t operator>>(const int shift) const noexcept { return bool_t(value >> shift); }

        constexpr bool_t& operator<<=(const int shift) noexcept {
            value <<= shift;
            return *this;
        }
        constexpr bool_t& operator>>=(const int shift) noexcept {
            value >>= shift;
            return *this;
        }

        constexpr bool operator==(const int other) const noexcept { return value == static_cast<uint8_t>(other); }
        constexpr bool operator!=(const int other) const noexcept { return value != static_cast<uint8_t>(other); }
    };

    template <typename T>
    class buffer_t {
        std::shared_ptr<T[]> value;

    public:
        size_t size = 0;

        buffer_t() noexcept = default;
        buffer_t(const buffer_t&) noexcept = default;
        buffer_t(buffer_t&&) noexcept = default;

        explicit buffer_t(const size_t n) {
            if constexpr (std::is_same_v<T, dtypes::bool_t>) {
                size = (n + 7) / 8;
            } else {
                size = n;
            }
            value = std::shared_ptr<T[]>(new T[size](), std::default_delete<T[]>());
        }
        buffer_t(const T* ptr, const size_t n) {
            if constexpr (std::is_same_v<T, dtypes::bool_t>) {
                size = (n + 7) / 8;
            } else {
                size = n;
            }
            value = std::shared_ptr<T[]>(new T[size], std::default_delete<T[]>());
            std::copy_n(ptr, size, value.get());
        }
        buffer_t(T* raw_ptr, const size_t n, std::nullptr_t) noexcept {
            if constexpr (std::is_same_v<T, dtypes::bool_t>) {
                size = (n + 7) / 8;
            } else {
                size = n;
            }
            value = std::shared_ptr<T[]>(raw_ptr, [](T*) {});
        }

        buffer_t& operator=(const buffer_t&) noexcept = default;
        buffer_t& operator=(buffer_t&&) noexcept = default;
        constexpr operator bool() const noexcept { return static_cast<bool>(value); }

        constexpr std::conditional_t<std::is_same_v<T, dtypes::bool_t>, dtypes::bitref_t, T&>
        operator[](const size_t i) noexcept {
            if constexpr (std::is_same_v<T, dtypes::bool_t>) {
                return value[i / 8][i % 8];
            } else {
                return value[i];
            }
        }
        constexpr std::conditional_t<std::is_same_v<T, dtypes::bool_t>, dtypes::bitref_t, const T&>
        operator[](const size_t i) const noexcept {
            if constexpr (std::is_same_v<T, dtypes::bool_t>) {
                return value[i / 8][i % 8];
            } else {
                return value[i];
            }
        }

        constexpr T* data() noexcept { return value.get(); }
        constexpr const T* data() const noexcept { return value.get(); }

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

        constexpr explicit slice_t(const ll_t start = none, const ll_t stop = none, const ll_t step = 1) :
            start(start), stop(stop), step(step) {
            if (step == 0) {
                throw std::invalid_argument("slice step cannot be zero");
            }
        }

        constexpr slice_t& resolve(const size_t dim) noexcept {
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
            return *this;
        }

        size_t size(const size_t dim = none) {
            if (!resolved) {
                if (dim == none) {
                    throw std::invalid_argument("dimension required to resolve slice bounds");
                }
                resolve(dim);
            }
            if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) {
                return 0;
            }
            return (std::abs(stop - start) + std::abs(step) - 1) / std::abs(step);
        }
    };

    struct shape_t {
        size_t rows, cols;

        constexpr shape_t() noexcept : rows(1), cols(1) {}
        constexpr shape_t(const size_t col) noexcept : rows(1), cols(col) {}
        constexpr shape_t(const size_t rows, const size_t cols) noexcept : rows(rows), cols(cols) {}

        constexpr bool operator==(const shape_t& shape) const = default;

        constexpr size_t size() const noexcept { return rows * cols; }

        friend std::ostream& operator<<(std::ostream& out, const shape_t& shape) {
            return out << '(' << shape.rows << ", " << shape.cols << ')';
        }
    };

    template <typename T>
    struct out_t {
        array<T>* ptr;

        constexpr out_t() noexcept : ptr(nullptr) {}
        constexpr out_t(array<T>& arr) noexcept : ptr(&arr) {}
        constexpr out_t(std::nullptr_t) noexcept : ptr(nullptr) {}

        constexpr operator bool() const noexcept { return ptr != nullptr; }
        constexpr array<T>& operator*() const { return *ptr; }
        constexpr array<T>* operator->() const { return ptr; }
    };

    template <typename T>
    struct range_t {
        T start, stop, step;

        constexpr range_t(const T stop) : range_t(0, stop) {}
        constexpr range_t(const slice_t& slice) : range_t(slice.start, slice.stop, slice.step) {}

        constexpr range_t(const T start, const T stop, const T step = 1) : start(start), stop(stop), step(step) {
            if (step == T(0)) {
                throw std::invalid_argument("range step cannot be zero");
            }
        }

        buffer_t<T> evaluate() const noexcept {
            buffer_t<T> buf(size());
            size_t i = 0;

            for (T value = start; value < stop; value += step) {
                buf[i++] = value;
            }
            return buf;
        }

        constexpr size_t size() const noexcept {
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

        constexpr where_t() noexcept : ptr(nullptr) {}
        constexpr where_t(const MaskedArray& arr) noexcept : ptr(&arr) {}
        constexpr where_t(std::nullptr_t) noexcept : ptr(nullptr) {}

        constexpr operator bool() const noexcept { return ptr != nullptr; }
        constexpr const MaskedArray& operator*() const noexcept { return *ptr; }
        constexpr const MaskedArray* operator->() const noexcept { return ptr; }
    };

    namespace none {
        inline constexpr int axis = none_t();
        template <typename T>
        inline constexpr out_t<T> out(nullptr);
        inline constexpr where_t where(nullptr);
        inline constexpr auto base = nullptr;
        inline constexpr auto size = static_cast<size_t>(none_t<size_t>());
        inline constexpr auto shape = shape_t(1, size);
        inline constexpr auto slice = slice_t();
        template <typename T, typename dtype = T>
        inline constexpr auto func = [](const T& value) -> dtype { return static_cast<dtype>(value); };
    } // namespace none

    class index_t {
        using array_ptr = std::unique_ptr<array<ll_t>>;
        const std::variant<ll_t, slice_t, array_ptr> row, col;

    public:
        constexpr index_t(const ll_t i, const ll_t j) noexcept : row(i), col(j) {}
        constexpr index_t(const ll_t i, const slice_t& j = none::slice) noexcept : row(i), col(j) {}
        index_t(const ll_t i, const array<ll_t>& j) noexcept : row(i), col(std::make_unique<array<ll_t>>(j)) {}
        constexpr index_t(const slice_t& i, const ll_t j) noexcept : row(i), col(j) {}
        constexpr index_t(const slice_t& i, const slice_t& j = none::slice) noexcept : row(i), col(j) {}
        index_t(const slice_t& i, const array<ll_t>& j) noexcept : row(i), col(std::make_unique<array<ll_t>>(j)) {}
        index_t(const array<ll_t>& i, const ll_t j) noexcept : row(std::make_unique<array<ll_t>>(i)), col(j) {}
        index_t(const array<ll_t>& i, const slice_t& j = none::slice) noexcept :
            row(std::make_unique<array<ll_t>>(i)), col(j) {}
        index_t(const array<ll_t>& i, const array<ll_t>& j) noexcept :
            row(std::make_unique<array<ll_t>>(i)), col(std::make_unique<array<ll_t>>(j)) {}

        constexpr bool is_scalar_row() const noexcept { return std::holds_alternative<ll_t>(row); }
        constexpr bool is_scalar_col() const noexcept { return std::holds_alternative<ll_t>(col); }
        constexpr bool is_scalar() const noexcept { return is_scalar_row() && is_scalar_col(); }

        constexpr bool is_slice_row() const noexcept { return std::holds_alternative<slice_t>(row); }
        constexpr bool is_slice_col() const noexcept { return std::holds_alternative<slice_t>(col); }
        constexpr bool is_slice() const noexcept { return is_slice_row() && is_slice_col(); }

        constexpr bool is_array_row() const noexcept { return std::holds_alternative<array_ptr>(row); }
        constexpr bool is_array_col() const noexcept { return std::holds_alternative<array_ptr>(col); }
        constexpr bool is_array() const noexcept { return is_array_row() && is_array_col(); }

        constexpr ll_t get_scalar_row() const noexcept { return std::get<ll_t>(row); }
        constexpr ll_t get_scalar_col() const noexcept { return std::get<ll_t>(col); }
        constexpr std::pair<ll_t, ll_t> get_scalar() const noexcept { return {get_scalar_row(), get_scalar_col()}; }

        constexpr slice_t get_slice_row() const noexcept { return std::get<slice_t>(row); }
        constexpr slice_t get_slice_col() const noexcept { return std::get<slice_t>(col); }
        constexpr std::pair<slice_t, slice_t> get_slice() const noexcept { return {get_slice_row(), get_slice_col()}; }

        array<ll_t>* get_array_row() const noexcept { return std::get<array_ptr>(row).get(); }
        array<ll_t>* get_array_col() const noexcept { return std::get<array_ptr>(col).get(); }
        std::pair<array<ll_t>*, array<ll_t>*> get_array() const noexcept { return {get_array_row(), get_array_col()}; }
    };
} // namespace numcpp
