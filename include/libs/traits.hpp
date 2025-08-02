#pragma once

namespace numcpp {
    template <typename V>
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
} // namespace numcpp
