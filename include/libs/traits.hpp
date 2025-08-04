#pragma once

namespace numcpp {
    template <typename>
    class array;

    template <typename V = int>
    struct none_t {
        explicit constexpr operator V() const { return std::numeric_limits<V>::max(); }
    };

    struct axis_t {
        enum class type_t { none, single, pair } type;
        int axis0, axis1;

        axis_t() noexcept : type(type_t::none), axis0(-1), axis1(-1) {}
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

    template <typename V>
    class buffer_t {
        std::shared_ptr<V[]> value;
        size_t size = 0;

    public:
        buffer_t() noexcept = default;
        buffer_t(const buffer_t&) noexcept = default;
        buffer_t(buffer_t&&) noexcept = default;
        explicit buffer_t(const size_t n) : value(new V[n](), std::default_delete<V[]>()), size(n) {}
        buffer_t(V* ptr, const size_t n) : value(ptr, std::default_delete<V[]>()), size(n) {}
        buffer_t(std::shared_ptr<V[]> ptr, const size_t n) noexcept : value(std::move(ptr)), size(n) {}
        buffer_t(V* raw_ptr, const size_t n, std::nullptr_t) noexcept : value(raw_ptr, [](V*) {}), size(n) {}

        buffer_t(std::vector<V>&& vec) : size(vec.size()) {
            V* raw = vec.data();
            value = std::shared_ptr<V[]>(raw, [vec = std::move(vec)](V*) mutable {});
        }

        buffer_t& operator=(const buffer_t&) noexcept = default;
        buffer_t& operator=(buffer_t&&) noexcept = default;
        V& operator[](const size_t i) noexcept { return value[i]; }
        const V& operator[](const size_t i) const noexcept { return value[i]; }
        explicit operator bool() const noexcept { return static_cast<bool>(value); }

        V* data() noexcept { return value.get(); }
        const V* data() const noexcept { return value.get(); }

        void reset() noexcept {
            value.reset();
            size = 0;
        }

        void reset(const V* ptr, const size_t n) {
            value.reset(ptr);
            size = n;
        }
    };

    using ll_t = long long;

    class slice_t {
        bool resolved = false;

    public:
        static constexpr ll_t none = static_cast<ll_t>(none_t<ll_t>());
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
        std::variant<ll_t, slice_t> row, col;

        index_t(ll_t i, ll_t j) : row(i), col(j) {}
        index_t(ll_t i, const slice_t& j) : row(i), col(j) {}
        index_t(const slice_t& i, ll_t j) : row(i), col(j) {}
        index_t(const slice_t& i, const slice_t& j) : row(i), col(j) {}

        bool is_scalar_row() const { return std::holds_alternative<ll_t>(row); }
        bool is_scalar_col() const { return std::holds_alternative<ll_t>(col); }
        bool is_scalar() const { return is_scalar_row() && is_scalar_col(); }
        bool is_slice_row() const { return std::holds_alternative<slice_t>(row); }
        bool is_slice_col() const { return std::holds_alternative<slice_t>(col); }
        bool is_slice() const { return is_slice_row() && is_slice_col(); }

        ll_t get_scalar_row() const { return std::get<ll_t>(row); }
        ll_t get_scalar_col() const { return std::get<ll_t>(col); }
        std::pair<ll_t, ll_t> get_scalar() const { return {get_scalar_row(), get_scalar_col()}; }
        const slice_t& get_slice_row() const { return std::get<slice_t>(row); }
        const slice_t& get_slice_col() const { return std::get<slice_t>(col); }
        std::pair<slice_t, slice_t> get_slice() const { return {get_slice_row(), get_slice_col()}; }
    };

    template <typename>
    struct is_complex : std::false_type {};

    template <typename V>
    struct is_complex<std::complex<V>> : std::true_type {};

    template <typename V>
    constexpr bool is_complex_v = is_complex<V>::value;

    struct out_t {
        array<bool>* ptr;

        out_t() noexcept = default;
        explicit out_t(array<bool>& arr) noexcept : ptr(&arr) {}
        out_t(std::nullptr_t) noexcept : ptr(nullptr) {}

        constexpr operator bool() const noexcept { return ptr != nullptr; }

        bool is_set() const noexcept { return *this; }
        array<bool>& get() const noexcept { return *ptr; }
    };

    template <typename L, typename R>
    using promote_t = decltype(std::declval<L>() + std::declval<R>());

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

            iterator& operator++() {
                current += step;
                return *this;
            }

            iterator operator++(int) {
                iterator tmp = *this;
                ++(*this);
                return tmp;
            }
        };

        V start, stop, step;

        range_t(const V stop) noexcept : range_t(0, stop) {}

        range_t(const V start, const V stop, const V step = 1) : start(start), stop(stop), step(step) {
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

    struct shape_t {
        size_t rows, cols;

        shape_t() noexcept : rows(1), cols(1) {}
        shape_t(const size_t col) noexcept : rows(1), cols(col) {}
        shape_t(const size_t rows, const size_t cols) noexcept : rows(rows), cols(cols) {}
    };

    struct where_t {
        const array<bool>* ptr;

        where_t() noexcept = default;
        explicit where_t(const array<bool>& arr) noexcept : ptr(&arr) {}
        where_t(std::nullptr_t) noexcept : ptr(nullptr) {}

        constexpr operator bool() const noexcept { return ptr != nullptr; }

        bool is_set() const noexcept { return *this; }
        const array<bool>& get() const noexcept { return *ptr; }
    };
} // namespace numcpp
