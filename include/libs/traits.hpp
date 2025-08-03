#pragma once

namespace numcpp {
    template <typename>
    class array;

    struct none_t {
        constexpr explicit operator int() const { return std::numeric_limits<int>::max(); }
    };

    struct axis_t {
        std::variant<none_t, int, std::pair<int, int>> value;

        axis_t() : value(none_t()) {}
        axis_t(const int axis) : value(axis) {}
        axis_t(const std::pair<int, int>& axes) : value(axes) {}
        axis_t(const none_t& n) : value(n) {}
    };

    struct out_t {
        std::optional<std::reference_wrapper<array<bool>>> value;

        out_t() = default;
        out_t(array<bool>& arr) : value(std::ref(arr)) {}
        out_t(const std::nullopt_t&) : value(std::nullopt) {}
    };

    struct where_t {
        std::optional<std::reference_wrapper<const array<bool>>> value;

        where_t() = default;
        where_t(const array<bool>& arr) : value(std::cref(arr)) {}
        where_t(const std::nullopt_t&) : value(std::nullopt) {}
    };

    struct shape_t {
        std::variant<size_t, std::pair<size_t, size_t>> value;

        shape_t() = default;
        shape_t(const size_t& n) : value(n) {}
        shape_t(const std::pair<size_t, size_t>& pair) : value(pair) {}
    };

    template <typename>
    struct is_complex : std::false_type {};

    template <typename V>
    struct is_complex<std::complex<V>> : std::true_type {};

    template <typename V>
    constexpr bool is_complex_v = is_complex<V>::value;

    template <typename L, typename R>
    struct promote_type {
        using type = decltype(std::declval<L>() + std::declval<R>());
    };

    template <typename L, typename R>
    using promote_t = typename promote_type<L, R>::type;

    using ll_t = long long;
} // namespace numcpp
