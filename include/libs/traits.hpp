#pragma once

namespace numcpp {
    template <typename>
    class array;
    class MaskedArray;

    template <typename V = int>
    struct none_t {
        constexpr operator V() const { return std::numeric_limits<V>::max(); }
    };

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
        template <typename V>
        std::string to_string(const V&);
    }

    template <typename V>
    struct complex_t {
        using value_type = V;
        V real, imag;

        constexpr complex_t() noexcept : complex_t(0, 0) {}
        constexpr complex_t(const V real) noexcept : complex_t(real, 0) {}
        constexpr complex_t(const V real, const V imag) noexcept : real(real), imag(imag) {}

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
        constexpr complex_t operator/(const complex_t& other) const noexcept {
            V denominator = other.real * other.real + other.imag * other.imag;
            return complex_t((real * other.real + imag * other.imag) / denominator,
                             (imag * other.real - real * other.imag) / denominator);
        }

        constexpr complex_t operator+(const V value) const noexcept { return complex_t(real + value, imag); }
        constexpr complex_t operator-(const V value) const noexcept { return complex_t(real - value, imag); }
        constexpr complex_t operator*(const V value) const noexcept { return complex_t(real * value, imag * value); }
        constexpr complex_t operator/(const V value) const noexcept { return complex_t(real / value, imag / value); }

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
            V r = real * other.real - imag * other.imag, i = real * other.imag + imag * other.real;
            real = r;
            imag = i;
            return *this;
        }
        complex_t& operator/=(const complex_t& other) noexcept {
            V denominator = other.real * other.real + other.imag * other.imag;
            V r = (real * other.real + imag * other.imag) / denominator;
            V i = (imag * other.real - real * other.imag) / denominator;
            real = r;
            imag = i;
            return *this;
        }

        complex_t& operator+=(const V value) noexcept {
            real += value;
            return *this;
        }
        complex_t& operator-=(const V value) noexcept {
            real -= value;
            return *this;
        }
        complex_t& operator*=(V value) noexcept {
            real *= value;
            imag *= value;
            return *this;
        }
        complex_t& operator/=(V value) noexcept {
            real /= value;
            imag /= value;
            return *this;
        }

        constexpr bool operator==(const complex_t& other) const noexcept {
            return real == other.real && imag == other.imag;
        }
        constexpr bool operator!=(const complex_t& other) const noexcept { return !(*this == other); }
        constexpr operator bool() const noexcept { return real != 0 || imag != 0; }

        V abs() const noexcept { return std::sqrt(norm()); }
        constexpr complex_t conj() const noexcept { return complex_t(real, -imag); }
        constexpr V norm() const noexcept { return real * real + imag * imag; }
        std::complex<V> to_std() const noexcept { return std::complex<V>(real, imag); }

        friend std::ostream& operator<<(std::ostream& out, const complex_t& complex) noexcept {
            out << detail::to_string(complex);
            return out;
        }

        static complex_t from_polar(const V magnitude, const V angle_rad) noexcept {
            return complex_t(magnitude * std::cos(angle_rad), magnitude * std::sin(angle_rad));
        }
    };

    namespace dtype {
        class bitref_t;
        struct bool_t;
        using int8_t = int8_t;
        using int16_t = int16_t;
        using int32_t = int32_t;
        using int64_t = int64_t;
        using int128_t = __int128_t;

        using uint8_t = uint8_t;
        using uint16_t = uint16_t;
        using uint32_t = uint32_t;
        using uint64_t = uint64_t;
        using uint128_t = __uint128_t;

        using float32_t = float;
        using float64_t = double;
        using float128_t = long double;

        using complex64_t = complex_t<float>;
        using complex128_t = complex_t<double>;
        using complex256_t = complex_t<long double>;
    } // namespace dtype

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

    template <typename V>
    class buffer_t {
        std::shared_ptr<V[]> value;

    public:
        size_t size = 0;

        buffer_t() noexcept = default;
        buffer_t(const buffer_t&) noexcept = default;
        buffer_t(buffer_t&&) noexcept = default;
        explicit buffer_t(const size_t n) : value(new V[n](), std::default_delete<V[]>()), size(n) {}
        buffer_t(V* ptr, const size_t n) : value(ptr, std::default_delete<V[]>()), size(n) {}
        buffer_t(std::shared_ptr<V[]> ptr, const size_t n) noexcept : value(std::move(ptr)), size(n) {}
        buffer_t(V* raw_ptr, const size_t n, std::nullptr_t) noexcept : value(raw_ptr, [](V*) {}), size(n) {}

        buffer_t& operator=(const buffer_t&) noexcept = default;
        buffer_t& operator=(buffer_t&&) noexcept = default;
        V& operator[](const size_t i) noexcept { return value[i]; }
        const V& operator[](const size_t i) const noexcept { return value[i]; }
        operator bool() const noexcept { return static_cast<bool>(value); }

        V* data() noexcept { return value.get(); }
        const V* data() const noexcept { return value.get(); }

        void reset() noexcept {
            value.reset();
            size = 0;
        }

        void reset(const V* ptr, const size_t n) noexcept {
            value.reset(ptr);
            size = n;
        }
    };

    using ll_t = long long;

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

    template <typename>
    struct is_complex : std::false_type {};

    template <typename V>
    struct is_complex<complex_t<V>> : std::true_type {};

    template <typename V>
    constexpr bool is_complex_v = is_complex<V>::value;

    template <typename V>
    struct out_t {
        array<V>* ptr;

        out_t() noexcept : ptr(nullptr) {}
        out_t(array<V>& arr) noexcept : ptr(&arr) {}
        out_t(std::nullptr_t) noexcept : ptr(nullptr) {}

        constexpr operator bool() const noexcept { return ptr != nullptr; }
        constexpr array<V>& operator*() { return *ptr; }
        constexpr array<V>* operator->() { return ptr; }
    };

    template <typename L, typename R>
    struct promote {
        using type = std::conditional_t<std::is_same_v<L, dtype::bool_t> && std::is_same_v<R, dtype::bool_t>,
                                        dtype::bool_t, std::common_type_t<L, R>>;
    };

    template <typename L, typename R>
    using promote_t = typename promote<L, R>::type;


    template <typename V>
    struct range_t {
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
            bool operator==(const iterator& other) const noexcept { return current == other.current; }
            bool operator!=(const iterator& other) const noexcept { return !(*this == other); }

            iterator& operator++() noexcept {
                current += step;
                return *this;
            }

            iterator operator++(int) noexcept {
                iterator tmp = *this;
                ++(*this);
                return tmp;
            }
        };

        V start, stop, step;

        range_t(const V stop) noexcept : range_t(0, stop) {}

        range_t(const V start, const V stop, const V step = 1) noexcept : start(start), stop(stop), step(step) {
            if (step == V(0)) {
                throw std::invalid_argument("range step cannot be zero");
            }
        }

        size_t size() const noexcept {
            if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) {
                return 0;
            }
            if constexpr (std::is_integral_v<V>) {
                return (stop - start + (step > 0 ? step - 1 : step + 1)) / step;
            } else {
                return static_cast<size_t>(std::ceil((stop - start) / step));
            }
        }
        iterator begin() const noexcept { return iterator(start, stop, step); }
        iterator end() const noexcept { return iterator(start + step * static_cast<V>(size()), stop, step); }
    };

    template <typename V, bool = is_complex_v<V>>
    struct real_type {
        using type = V;
    };

    template <typename V>
    struct real_type<V, true> {
        using type = typename V::value_type;
    };

    template <typename V>
    using real_t = typename real_type<V>::type;

    struct shape_t {
        size_t rows, cols;

        shape_t() noexcept : rows(1), cols(1) {}
        shape_t(const size_t col) noexcept : rows(1), cols(col) {}
        shape_t(const size_t rows, const size_t cols) noexcept : rows(rows), cols(cols) {}
    };

    struct where_t {
        const array<bool>* ptr;

        where_t() noexcept : ptr(nullptr) {}
        where_t(const array<bool>& arr) noexcept : ptr(&arr) {}
        where_t(std::nullptr_t) noexcept : ptr(nullptr) {}

        constexpr operator bool() const noexcept { return ptr != nullptr; }
        constexpr const array<bool>& operator*() const noexcept { return *ptr; }
        constexpr const array<bool>* operator->() const noexcept { return ptr; }
    };
} // namespace numcpp
