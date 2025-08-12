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

    namespace dtypes {
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
    inline constexpr dtypes::float32_t nan = std::numeric_limits<dtypes::float32_t>::quiet_NaN();
    inline constexpr dtypes::float32_t inf = std::numeric_limits<dtypes::float32_t>::infinity();

    template <typename> struct is_integral : std::false_type {};
    template <> struct is_integral<bool> : std::true_type {};
    template <> struct is_integral<dtypes::bool_t> : std::true_type {};

    template <> struct is_integral<dtypes::int8_t> : std::true_type {};
    template <> struct is_integral<dtypes::int16_t> : std::true_type {};
    template <> struct is_integral<dtypes::int32_t> : std::true_type{};
    template <> struct is_integral<dtypes::int64_t> : std::true_type {};
    template <> struct is_integral<dtypes::int128_t> : std::true_type {};

    template <> struct is_integral<dtypes::uint8_t> : std::true_type {};
    template <> struct is_integral<dtypes::uint16_t> : std::true_type {};
    template <> struct is_integral<dtypes::uint32_t> : std::true_type {};
    template <> struct is_integral<dtypes::uint64_t> : std::true_type {};
    template <> struct is_integral<dtypes::uint128_t> : std::true_type {};
    template <typename T> inline constexpr bool is_integral_v = is_integral<T>::value;

    template <typename> struct is_floating_point : std::false_type {};
    template <> struct is_floating_point<dtypes::float32_t> : std::true_type {};
    template <> struct is_floating_point<dtypes::float64_t> : std::true_type {};
    template <> struct is_floating_point<dtypes::float128_t> : std::true_type {};
    template <typename T> inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

    template <typename> struct is_complex : std::false_type {};
    template <> struct is_complex<dtypes::complex64_t> : std::true_type {};
    template <> struct is_complex<dtypes::complex128_t> : std::true_type {};
    template <> struct is_complex<dtypes::complex256_t> : std::true_type {};
    template <typename T> inline constexpr bool is_complex_v = is_complex<T>::value;

    template <typename T> struct is_numeric : std::bool_constant<is_integral_v<T> || is_floating_point_v<T> || is_complex_v<T>> {};
    template <typename T> inline constexpr bool is_numeric_v = is_numeric<T>::value;

    template <typename T, bool = is_complex_v<T>> struct real_type { using type = T; };
    template <typename T> struct real_type<T, true> { using type = typename T::value_type;};
    template <typename T> using real_t = typename real_type<T>::type;
    template <typename T> inline constexpr bool is_real_v = is_integral_v<T> || is_floating_point_v<T>;

    template <typename T> inline constexpr bool is_bool_t = std::is_same_v<T, dtypes::bool_t>;

    enum class category { boolean, signed_int, unsigned_int, floating, complex, unknown };
    template <typename> struct type_category { static constexpr auto value = category::unknown; };

    template <> struct type_category<bool> : std::integral_constant<category, category::boolean> {};
    template <> struct type_category<dtypes::bool_t> : std::integral_constant<category, category::boolean> {};

    template <> struct type_category<dtypes::int8_t> : std::integral_constant<category, category::signed_int> {};
    template <> struct type_category<dtypes::int16_t> : std::integral_constant<category, category::signed_int> {};
    template <> struct type_category<dtypes::int32_t> : std::integral_constant<category, category::signed_int> {};
    template <> struct type_category<dtypes::int64_t> : std::integral_constant<category, category::signed_int> {};
    template <> struct type_category<dtypes::int128_t> : std::integral_constant<category, category::signed_int> {};

    template <> struct type_category<dtypes::uint8_t> : std::integral_constant<category, category::unsigned_int> {};
    template <> struct type_category<dtypes::uint16_t> : std::integral_constant<category, category::unsigned_int> {};
    template <> struct type_category<dtypes::uint32_t> : std::integral_constant<category, category::unsigned_int> {};
    template <> struct type_category<dtypes::uint64_t> : std::integral_constant<category, category::unsigned_int> {};
    template <> struct type_category<dtypes::uint128_t> : std::integral_constant<category, category::unsigned_int> {};

    template <> struct type_category<dtypes::float32_t> : std::integral_constant<category, category::floating> {};
    template <> struct type_category<dtypes::float64_t> : std::integral_constant<category, category::floating> {};
    template <> struct type_category<dtypes::float128_t> : std::integral_constant<category, category::floating> {};

    template <> struct type_category<dtypes::complex64_t> : std::integral_constant<category, category::complex> {};
    template <> struct type_category<dtypes::complex128_t> : std::integral_constant<category, category::complex> {};
    template <> struct type_category<dtypes::complex256_t> : std::integral_constant<category, category::complex> {};

    template <size_t bits>
    struct bits_of_int {
        using type = std::conditional_t<bits <= 8,dtypes::int8_t,
                         std::conditional_t<bits <= 16, dtypes::int16_t,
                             std::conditional_t<bits <= 32, dtypes::int32_t,
                                 std::conditional_t<bits <= 64, dtypes::int64_t,
                                     std::conditional_t<bits <= 128, dtypes::int128_t, dtypes::float128_t>>>>>;
    };
    template <size_t bits>
    struct bits_of_float {
        using type = std::conditional_t<bits <= 32, dtypes::float32_t,
                         std::conditional_t<bits <= 64, dtypes::float64_t, dtypes::float128_t>>;
    };
    template <size_t bits>
    struct bits_of_complex {
        using type = std::conditional_t<bits <= 64, dtypes::complex64_t,
                         std::conditional_t<bits <= 128, dtypes::complex128_t, dtypes::complex256_t>>;
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
                        std::conditional_t<std::is_same_v<Operation, comparison_t>, dtypes::bool_t, selected_type>>;
    };

    template <typename A, typename B, typename Operation = none_t<>>
    using promote_t = typename promote<A, B, Operation>::type;
} // namespace numcpp