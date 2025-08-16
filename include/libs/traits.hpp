#pragma once

namespace numcpp {
    template <typename V = int>
    struct none_t {
        constexpr operator V() const { return std::numeric_limits<V>::max(); }
    };

    namespace operations {
        struct in_place_t {};
        struct swap_t {};
        struct comparison_t {};
    } // namespace operations

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

    template <typename>
    struct complex_t;
    using complex64_t = complex_t<float>;
    using complex128_t = complex_t<double>;
    using complex256_t = complex_t<long double>;

    using str = std::string;

    using ll_t = long long;
    inline constexpr float128_t pi = M_PI;
    inline constexpr float128_t e = M_E;
    inline constexpr float128_t nan = std::numeric_limits<float128_t>::quiet_NaN();
    inline constexpr float128_t inf = std::numeric_limits<float128_t>::infinity();

    template <typename> struct is_integral : std::false_type {};
    template <> struct is_integral<bool> : std::true_type {};
    template <> struct is_integral<int8_t> : std::true_type {};
    template <> struct is_integral<int16_t> : std::true_type {};
    template <> struct is_integral<int32_t> : std::true_type{};
    template <> struct is_integral<int64_t> : std::true_type {};
    template <> struct is_integral<int128_t> : std::true_type {};

    template <> struct is_integral<uint8_t> : std::true_type {};
    template <> struct is_integral<uint16_t> : std::true_type {};
    template <> struct is_integral<uint32_t> : std::true_type {};
    template <> struct is_integral<uint64_t> : std::true_type {};
    template <> struct is_integral<uint128_t> : std::true_type {};
    template <typename T> inline constexpr bool is_integral_v = is_integral<T>::value;

    template <typename> struct is_floating_point : std::false_type {};
    template <> struct is_floating_point<float32_t> : std::true_type {};
    template <> struct is_floating_point<float64_t> : std::true_type {};
    template <> struct is_floating_point<float128_t> : std::true_type {};
    template <typename T> inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

    template <typename> struct is_complex : std::false_type {};
    template <> struct is_complex<complex64_t> : std::true_type {};
    template <> struct is_complex<complex128_t> : std::true_type {};
    template <> struct is_complex<complex256_t> : std::true_type {};
    template <typename T> inline constexpr bool is_complex_v = is_complex<T>::value;

    template <typename T> struct is_numeric : std::bool_constant<is_integral_v<T> || is_floating_point_v<T> || is_complex_v<T>> {};
    template <typename T> inline constexpr bool is_numeric_v = is_numeric<T>::value;

    template <typename T, bool = is_complex_v<T>> struct real_type { using type = T; };
    template <typename T> struct real_type<T, true> { using type = typename T::value_type;};
    template <typename T> using real_t = typename real_type<T>::type;
    template <typename T> inline constexpr bool is_real_v = is_integral_v<T> || is_floating_point_v<T>;

    enum class category { boolean, signed_int, unsigned_int, floating, complex, unknown };
    template <typename> struct type_category { static constexpr auto value = category::unknown; };

    template <> struct type_category<bool> : std::integral_constant<category, category::boolean> {};

    template <> struct type_category<int8_t> : std::integral_constant<category, category::signed_int> {};
    template <> struct type_category<int16_t> : std::integral_constant<category, category::signed_int> {};
    template <> struct type_category<int32_t> : std::integral_constant<category, category::signed_int> {};
    template <> struct type_category<int64_t> : std::integral_constant<category, category::signed_int> {};
    template <> struct type_category<int128_t> : std::integral_constant<category, category::signed_int> {};

    template <> struct type_category<uint8_t> : std::integral_constant<category, category::unsigned_int> {};
    template <> struct type_category<uint16_t> : std::integral_constant<category, category::unsigned_int> {};
    template <> struct type_category<uint32_t> : std::integral_constant<category, category::unsigned_int> {};
    template <> struct type_category<uint64_t> : std::integral_constant<category, category::unsigned_int> {};
    template <> struct type_category<uint128_t> : std::integral_constant<category, category::unsigned_int> {};

    template <> struct type_category<float32_t> : std::integral_constant<category, category::floating> {};
    template <> struct type_category<float64_t> : std::integral_constant<category, category::floating> {};
    template <> struct type_category<float128_t> : std::integral_constant<category, category::floating> {};

    template <> struct type_category<complex64_t> : std::integral_constant<category, category::complex> {};
    template <> struct type_category<complex128_t> : std::integral_constant<category, category::complex> {};
    template <> struct type_category<complex256_t> : std::integral_constant<category, category::complex> {};

    template <size_t bits>
    struct bits_of_int {
        using type = std::conditional_t<bits <= 8,int8_t,
                         std::conditional_t<bits <= 16, int16_t,
                             std::conditional_t<bits <= 32, int32_t,
                                 std::conditional_t<bits <= 64, int64_t,
                                     std::conditional_t<bits <= 128, int128_t, float128_t>>>>>;
    };
    template <size_t bits>
    struct bits_of_float {
        using type = std::conditional_t<bits <= 32, float32_t,
                         std::conditional_t<bits <= 64, float64_t, float128_t>>;
    };
    template <size_t bits>
    struct bits_of_complex {
        using type = std::conditional_t<bits <= 64, complex64_t,
                         std::conditional_t<bits <= 128, complex128_t, complex256_t>>;
    };

    template <typename L, typename R, typename Operation = none_t<>>
    class promote {
        static constexpr auto catL = type_category<L>::value, catR = type_category<R>::value;
        static constexpr int bitsL = std::integral_constant<int, sizeof(L) * 8>::value;
        static constexpr int bitsR = std::integral_constant<int, sizeof(R) * 8>::value;

        using selected_type =
            std::conditional_t<catL == catR,
                std::conditional_t<std::is_same_v<R, bool>, bool, std::conditional_t<bitsL >= bitsR, L, R>>,
                std::conditional_t<catL == category::boolean && catR == category::boolean, bool,
                    std::conditional_t<catL == category::boolean && (catR > category::boolean), R,
                        std::conditional_t<(catL > category::boolean) && catR == category::boolean, L,
                            std::conditional_t<catL == category::signed_int && catR == category::unsigned_int,
                                typename bits_of_int<(bitsL > bitsR ? bitsL : bitsR * 2)>::type,
                                std::conditional_t<catL == category::unsigned_int && catR == category::signed_int,
                                    typename bits_of_int<bitsL >= bitsR ? bitsL * 2 : bitsR>::type,
                                    std::conditional_t<(catL == category::signed_int ||
                                                        catL == category::unsigned_int) && catR == category::floating,
                                        typename bits_of_float<bitsL >= bitsR ? bitsL * 2 : bitsR>::type,
                                        std::conditional_t<catL == category::floating &&
                                                          (catR == category::signed_int || catR == category::unsigned_int),
                                            typename bits_of_float<(bitsL > bitsR ? bitsL : bitsR * 2)>::type,
                                            std::conditional_t<(catL == category::signed_int || catL == category::unsigned_int) &&
                                                                catR == category::complex,
                                                typename bits_of_complex<bitsL * 2 >= bitsR ? bitsL * 4 : bitsR>::type,
                                                std::conditional_t<catL == category::complex &&
                                                                  (catR == category::signed_int || catR == category::unsigned_int),
                                                    typename bits_of_complex<(bitsL > bitsR * 2 ? bitsL: bitsR * 4)>::type,
                                                    std::conditional_t<catL == category::floating && catR == category::complex,
                                                        typename bits_of_complex<(bitsL * 2 > bitsR ? bitsL * 2 : bitsR)>::type,
                                                        std::conditional_t<catL == category::complex && catR == category::floating,
                                                            typename bits_of_complex<bitsL >= bitsR * 2 ? bitsL : bitsR * 2>::type,
                                                            std::conditional_t<catL == category::complex || catR == category::complex,
                                                                typename bits_of_complex<(bitsL > bitsR ? bitsL : bitsR)>::type,
                                                                void>>>>>>>>>>>>>;

    public:
        using type = std::conditional_t<std::is_same_v<Operation, operations::in_place_t>, L,
                         std::conditional_t<std::is_same_v<Operation, operations::comparison_t>, bool,
                             selected_type>>;
    };

    template <typename L, typename R, typename Operation = none_t<>>
    using promote_t = typename promote<L, R, Operation>::type;
} // namespace numcpp