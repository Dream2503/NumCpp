#pragma once

namespace numcpp {
    template <typename V = int>
    struct none_t {
        constexpr operator V() const { return std::numeric_limits<V>::max(); }
    };

    struct in_place_t {};
    struct swap_t {};
    struct comparison_t {};

    template <typename>
    struct complex_t;

    template <typename> struct is_complex : std::false_type {};
    template <typename T> struct is_complex<complex_t<T>> : std::true_type {};
    template <typename T> constexpr bool is_complex_v = is_complex<T>::value;

    template <typename T, bool = is_complex_v<T>> struct real_type { using type = T; };
    template <typename T> struct real_type<T, true> { using type = typename T::value_type;};
    template <typename T> using real_t = typename real_type<T>::type;

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

        using str = std::string;
    } // namespace dtype

    using ll_t = long long;

    template <typename> struct is_numeric : std::false_type {};

    template <> struct is_numeric<bool> : std::true_type {};
    template <> struct is_numeric<dtype::bool_t> : std::true_type {};

    template <> struct is_numeric<dtype::int8_t> : std::true_type {};
    template <> struct is_numeric<dtype::int16_t> : std::true_type {};
    template <> struct is_numeric<dtype::int32_t> : std::true_type{};
    template <> struct is_numeric<dtype::int64_t> : std::true_type {};
    template <> struct is_numeric<dtype::int128_t> : std::true_type {};

    template <> struct is_numeric<dtype::uint8_t> : std::true_type {};
    template <> struct is_numeric<dtype::uint16_t> : std::true_type {};
    template <> struct is_numeric<dtype::uint32_t> : std::true_type {};
    template <> struct is_numeric<dtype::uint64_t> : std::true_type {};
    template <> struct is_numeric<dtype::uint128_t> : std::true_type {};

    template <> struct is_numeric<dtype::float32_t> : std::true_type {};
    template <> struct is_numeric<dtype::float64_t> : std::true_type {};
    template <> struct is_numeric<dtype::float128_t> : std::true_type {};

    template <> struct is_numeric<dtype::complex64_t> : std::true_type {};
    template <> struct is_numeric<dtype::complex128_t> : std::true_type {};
    template <> struct is_numeric<dtype::complex256_t> : std::true_type {};

    template <typename T> constexpr bool is_numeric_v = is_numeric<T>::value;

    enum class category { boolean, signed_int, unsigned_int, floating, complex, unknown };
    template <typename> struct type_category { static constexpr auto value = category::unknown; };

    template <> struct type_category<bool> : std::integral_constant<category, category::boolean> {};
    template <> struct type_category<dtype::bool_t> : std::integral_constant<category, category::boolean> {};

    template <> struct type_category<dtype::int8_t> : std::integral_constant<category, category::signed_int> {};
    template <> struct type_category<dtype::int16_t> : std::integral_constant<category, category::signed_int> {};
    template <> struct type_category<dtype::int32_t> : std::integral_constant<category, category::signed_int> {};
    template <> struct type_category<dtype::int64_t> : std::integral_constant<category, category::signed_int> {};
    template <> struct type_category<dtype::int128_t> : std::integral_constant<category, category::signed_int> {};

    template <> struct type_category<dtype::uint8_t> : std::integral_constant<category, category::unsigned_int> {};
    template <> struct type_category<dtype::uint16_t> : std::integral_constant<category, category::unsigned_int> {};
    template <> struct type_category<dtype::uint32_t> : std::integral_constant<category, category::unsigned_int> {};
    template <> struct type_category<dtype::uint64_t> : std::integral_constant<category, category::unsigned_int> {};
    template <> struct type_category<dtype::uint128_t> : std::integral_constant<category, category::unsigned_int> {};

    template <> struct type_category<dtype::float32_t> : std::integral_constant<category, category::floating> {};
    template <> struct type_category<dtype::float64_t> : std::integral_constant<category, category::floating> {};
    template <> struct type_category<dtype::float128_t> : std::integral_constant<category, category::floating> {};

    template <> struct type_category<dtype::complex64_t> : std::integral_constant<category, category::complex> {};
    template <> struct type_category<dtype::complex128_t> : std::integral_constant<category, category::complex> {};
    template <> struct type_category<dtype::complex256_t> : std::integral_constant<category, category::complex> {};

    template <size_t bits>
    struct bits_of_int {
        using type = std::conditional_t<bits <= 8,dtype::int8_t,
                         std::conditional_t<bits <= 16, dtype::int16_t,
                             std::conditional_t<bits <= 32, dtype::int32_t,
                                 std::conditional_t<bits <= 64, dtype::int64_t,
                                     std::conditional_t<bits <= 128, dtype::int128_t, dtype::float128_t>>>>>;
    };
    template <size_t bits>
    struct bits_of_float {
        using type = std::conditional_t<bits <= 32, dtype::float32_t,
                         std::conditional_t<bits <= 64, dtype::float64_t, dtype::float128_t>>;
    };
    template <size_t bits>
    struct bits_of_complex {
        using type = std::conditional_t<bits <= 64, dtype::complex64_t,
                         std::conditional_t<bits <= 128, dtype::complex128_t, dtype::complex256_t>>;
    };

    template <typename L, typename R, typename Operation = none_t<>>
    class promote {
        static constexpr auto catA = type_category<L>::value, catB = type_category<R>::value;
        static constexpr int bitsA = std::integral_constant<int, sizeof(L) * 8>::value;
        static constexpr int bitsB = std::integral_constant<int, sizeof(R) * 8>::value;

        using selected_type =
            std::conditional_t<catA == catB,
                std::conditional_t<std::is_same_v<R, bool>, bool, std::conditional_t<bitsA >= bitsB, L, R>>,
                std::conditional_t<catA == category::boolean && catB == category::boolean, bool,
                    std::conditional_t<catA == category::boolean && (catB > category::boolean), R,
                        std::conditional_t<(catA > category::boolean) && catB == category::boolean, L,
                            std::conditional_t<catA == category::signed_int && catB == category::unsigned_int,
                                typename bits_of_int<(bitsA > bitsB ? bitsA : bitsB * 2)>::type,
                                std::conditional_t<catA == category::unsigned_int && catB == category::signed_int,
                                    typename bits_of_int<bitsA >= bitsB ? bitsA * 2 : bitsB>::type,
                                    std::conditional_t<(catA == category::signed_int || catA == category::unsigned_int) && catB == category::floating,
                                        typename bits_of_float<bitsA >= bitsB ? bitsA * 2 : bitsB>::type,
                                        std::conditional_t<catA == category::floating && (catB == category::signed_int || catB == category::unsigned_int),
                                            typename bits_of_float<(bitsA > bitsB ? bitsA : bitsB * 2)>::type,
                                            std::conditional_t<(catA == category::signed_int || catA == category::unsigned_int) && catB == category::complex,
                                                typename bits_of_complex<bitsA * 2 >= bitsB ? bitsA * 4 : bitsB>::type,
                                                std::conditional_t<catA == category::complex && (catB == category::signed_int || catB == category::unsigned_int),
                                                    typename bits_of_complex<(bitsA > bitsB * 2 ? bitsA: bitsB * 4)>::type,
                                                    std::conditional_t<catA == category::floating && catB == category::complex,
                                                        typename bits_of_complex<(bitsA * 2 > bitsB ? bitsA * 2 : bitsB)>::type,
                                                        std::conditional_t<catA == category::complex && catB == category::floating,
                                                            typename bits_of_complex<bitsA >= bitsB * 2 ? bitsA : bitsB * 2>::type,
                                                            std::conditional_t<catA == category::complex || catB == category::complex,
                                                                void, void>>>>>>>>>>>>>;

    public:
        using type = std::conditional_t<std::is_same_v<Operation, in_place_t>, L,
                        std::conditional_t<std::is_same_v<Operation, comparison_t>, dtype::bool_t, selected_type>>;
    };

    template <typename A, typename B, typename Operation = none_t<>>
    using promote_t = typename promote<A, B, Operation>::type;
} // namespace numcpp