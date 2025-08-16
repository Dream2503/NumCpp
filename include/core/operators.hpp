#pragma once
#include "../libs/indexing.hpp"

namespace numcpp {
    template <typename L, typename R>
    array<promote_t<L, R>> operator+(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::plus());
    }
    template <typename L, typename R>
    array<promote_t<L, R>> operator+(const array<L>& lhs, const R& value) {
        return binary_opr_element_wise(lhs, value, std::plus());
    }
    template <typename L, typename R>
    array<promote_t<L, R>> operator+(const L& value, const array<R>& rhs) {
        return rhs + value;
    }

    inline array<bool> operator&(const array<bool>& lhs, const array<bool>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::bit_and());
    }
    inline array<bool> operator&(const array<bool>& lhs, const bool value) {
        return binary_opr_element_wise(lhs, value, std::bit_and());
    }
    inline array<bool> operator&(const bool value, const array<bool>& rhs) { return rhs & value; }

    template <typename L, typename R>
    array<promote_t<L, R>> operator&(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::bit_and());
    }
    template <typename L, typename R>
    array<promote_t<L, R>> operator&(const array<L>& lhs, const R& value) {
        return binary_opr_element_wise(lhs, value, std::bit_and());
    }
    template <typename L, typename R>
    array<promote_t<L, R>> operator&(const L& value, const array<R>& rhs) {
        return rhs & value;
    }


    template <typename V>
    array<V>& operator+(const array<V>& arr) {
        return arr;
    }

    template <typename L, typename R>
    array<L>& operator+=(array<L>& lhs, const array<R>& rhs) {
        binary_opr_broadcast(lhs, rhs, std::plus(), operations::in_place_t());
        return lhs;
    }
    template <typename L, typename R>
    array<L>& operator+=(array<L>& lhs, const R& value) {
        binary_opr_element_wise(lhs, value, std::plus(), operations::in_place_t());
        return lhs;
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator-(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::minus());
    }
    template <typename L, typename R>
    array<promote_t<L, R>> operator-(const array<L>& lhs, const R& value) {
        return binary_opr_element_wise(lhs, value, std::minus());
    }
    template <typename L, typename R>
    array<promote_t<L, R>> operator-(const L& value, const array<R>& rhs) {
        return binary_opr_element_wise(rhs, value, std::minus(), operations::swap_t());
    }
    template <typename V>
    array<V> operator-(const array<V>& arr) {
        return unary_opr_element_wise(arr, std::negate());
    }

    template <typename L, typename R>
    array<L>& operator-=(array<L>& lhs, const array<R>& rhs) {
        binary_opr_broadcast(lhs, rhs, std::minus(), operations::in_place_t());
        return lhs;
    }
    template <typename L, typename R>
    array<L>& operator-=(array<L>& lhs, const R& value) {
        binary_opr_element_wise(lhs, value, std::minus(), operations::in_place_t());
        return lhs;
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator*(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::multiplies());
    }
    template <typename L, typename R>
    array<promote_t<L, R>> operator*(const array<L>& lhs, const R& value) {
        return binary_opr_element_wise(lhs, value, std::multiplies());
    }
    template <typename L, typename R>
    array<promote_t<L, R>> operator*(const L& value, const array<R>& rhs) {
        return rhs * value;
    }

    template <typename L, typename R>
    array<L>& operator*=(array<L>& lhs, const array<R>& rhs) {
        binary_opr_broadcast(lhs, rhs, std::multiplies(), operations::in_place_t());
        return lhs;
    }
    template <typename L, typename R>
    array<L>& operator*=(array<L>& lhs, const R& value) {
        binary_opr_element_wise(lhs, value, std::multiplies(), operations::in_place_t());
        return lhs;
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator/(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, detail::divides());
    }

    template <typename L, typename R>
    array<promote_t<L, R>> operator/(const array<L>& lhs, const R& value) {
        return binary_opr_element_wise(lhs, value, detail::divides());
    }
    template <typename L, typename R>
    array<promote_t<L, R>> operator/(const L& value, const array<R>& rhs) {
        return binary_opr_element_wise(rhs, value, detail::divides(), operations::swap_t());
    }

    template <typename L, typename R>
    array<L>& operator/=(array<L>& lhs, const array<R>& rhs) {
        binary_opr_broadcast(lhs, rhs, detail::divides(), operations::in_place_t());
        return lhs;
    }
    template <typename L, typename R>
    array<L>& operator/=(array<L>& lhs, const R& value) {
        binary_opr_element_wise(lhs, value, detail::divides(), operations::in_place_t());
        return lhs;
    }

    template <typename L, typename R>
    requires(!std::is_floating_point_v<L> && !std::is_floating_point_v<R>)
    array<promote_t<L, R>> operator%(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, detail::modulus());
    }

    template <typename L, typename R>
    requires(!std::is_floating_point_v<L> && !std::is_floating_point_v<R>)
    array<promote_t<L, R>> operator%(const array<L>& lhs, const R& value) {
        return binary_opr_element_wise(lhs, value, detail::modulus());
    }
    template <typename L, typename R>
    requires(!std::is_floating_point_v<L> && !std::is_floating_point_v<R>)
    array<promote_t<L, R>> operator%(const L& value, const array<R>& rhs) {
        return binary_opr_element_wise(rhs, value, detail::modulus(), operations::swap_t());
    }

    template <typename L, typename R>
    requires(!std::is_floating_point_v<L> && !std::is_floating_point_v<R>)
    array<L>& operator%=(array<L>& lhs, const array<R>& rhs) {
        binary_opr_broadcast(lhs, rhs, detail::modulus(), operations::in_place_t());
        return lhs;
    }
    template <typename L, typename R>
    requires(!std::is_floating_point_v<L> && !std::is_floating_point_v<R>)
    array<L>& operator%=(array<L>& lhs, const R& value) {
        binary_opr_element_wise(lhs, value, detail::modulus(), operations::in_place_t());
        return lhs;
    }


    template <typename L, typename R>
    array<bool> operator==(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::equal_to(), operations::comparison_t());
    }
    template <typename L, typename R>
    array<bool> operator==(const array<L>& lhs, const R& rhs) {
        return binary_opr_element_wise(lhs, rhs, std::equal_to(), operations::comparison_t());
    }
    template <typename L, typename R>
    array<bool> operator==(const L& value, const array<R>& rhs) {
        return rhs == value;
    }

    template <typename L, typename R>
    array<bool> operator!=(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::not_equal_to(), operations::comparison_t());
    }
    template <typename L, typename R>
    array<bool> operator!=(const array<L>& lhs, const R& rhs) {
        return binary_opr_element_wise(lhs, rhs, std::not_equal_to(), operations::comparison_t());
    }
    template <typename L, typename R>
    array<bool> operator!=(const L& value, const array<R>& rhs) {
        return rhs != value;
    }

    template <typename L, typename R>
    array<bool> operator>(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::greater(), operations::comparison_t());
    }
    template <typename L, typename R>
    array<bool> operator>(const array<L>& lhs, const R& rhs) {
        return binary_opr_element_wise(lhs, rhs, std::greater(), operations::comparison_t());
    }
    template <typename L, typename R>
    array<bool> operator>(const L& value, const array<R>& rhs) {
        return rhs <= value;
    }

    template <typename L, typename R>
    array<bool> operator>=(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::greater_equal(), operations::comparison_t());
    }
    template <typename L, typename R>
    array<bool> operator>=(const array<L>& lhs, const R& rhs) {
        return binary_opr_element_wise(lhs, rhs, std::greater_equal(), operations::comparison_t());
    }
    template <typename L, typename R>
    array<bool> operator>=(const L& value, const array<R>& rhs) {
        return rhs < value;
    }

    template <typename L, typename R>
    array<bool> operator<(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::less(), operations::comparison_t());
    }
    template <typename L, typename R>
    array<bool> operator<(const array<L>& lhs, const R& rhs) {
        return binary_opr_element_wise(lhs, rhs, std::less(), operations::comparison_t());
    }
    template <typename L, typename R>
    array<bool> operator<(const L& value, const array<R>& rhs) {
        return rhs >= value;
    }

    template <typename L, typename R>
    array<bool> operator<=(const array<L>& lhs, const array<R>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::less_equal(), operations::comparison_t());
    }
    template <typename L, typename R>
    array<bool> operator<=(const array<L>& lhs, const R& rhs) {
        return binary_opr_element_wise(lhs, rhs, std::less_equal(), operations::comparison_t());
    }
    template <typename L, typename R>
    array<bool> operator<=(const L& value, const array<R>& rhs) {
        return rhs > value;
    }

    inline array<bool>& operator&=(array<bool>& lhs, const array<bool>& rhs) {
        binary_opr_broadcast(lhs, rhs, std::bit_and(), operations::in_place_t());
        return lhs;
    }
    inline array<bool>& operator&=(array<bool>& lhs, const bool value) {
        binary_opr_element_wise(lhs, value, std::bit_and(), operations::in_place_t());
        return lhs;
    }

    inline array<bool> operator|(const array<bool>& lhs, const array<bool>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::bit_or());
    }
    inline array<bool> operator|(const array<bool>& lhs, const bool value) {
        return binary_opr_element_wise(lhs, value, std::bit_or());
    }

    inline array<bool>& operator|=(array<bool>& lhs, const array<bool>& rhs) {
        binary_opr_broadcast(lhs, rhs, std::bit_or(), operations::in_place_t());
        return lhs;
    }
    inline array<bool>& operator|=(array<bool>& lhs, const bool value) {
        binary_opr_element_wise(lhs, value, std::bit_or(), operations::in_place_t());
        return lhs;
    }

    inline array<bool> operator^(const array<bool>& lhs, const array<bool>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::bit_xor());
    }

    inline array<bool> operator^(const array<bool>& lhs, const bool value) {
        return binary_opr_element_wise(lhs, value, std::bit_xor());
    }

    inline array<bool>& operator^=(array<bool>& lhs, const array<bool>& rhs) {
        binary_opr_broadcast(lhs, rhs, std::bit_xor(), operations::in_place_t());
        return lhs;
    }
    inline array<bool>& operator^=(array<bool>& lhs, const bool value) {
        binary_opr_element_wise(lhs, value, std::bit_xor(), operations::in_place_t());
        return lhs;
    }

    inline array<bool> operator==(const array<bool>& lhs, const array<bool>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::equal_to());
    }
    inline array<bool> operator==(const array<bool>& lhs, const bool value) {
        return binary_opr_element_wise(lhs, value, std::equal_to());
    }

    inline array<bool> operator!=(const array<bool>& lhs, const array<bool>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::not_equal_to());
    }
    inline array<bool> operator!=(const array<bool>& lhs, const bool value) {
        return binary_opr_element_wise(lhs, value, std::not_equal_to());
    }

    inline array<bool> operator>(const array<bool>& lhs, const array<bool>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::greater());
    }
    inline array<bool> operator>(const array<bool>& lhs, const bool value) {
        return binary_opr_element_wise(lhs, value, std::greater());
    }

    inline array<bool> operator>=(const array<bool>& lhs, const array<bool>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::greater_equal());
    }
    inline array<bool> operator>=(const array<bool>& lhs, const bool value) {
        return binary_opr_element_wise(lhs, value, std::greater_equal());
    }

    inline array<bool> operator<(const array<bool>& lhs, const array<bool>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::less());
    }
    inline array<bool> operator<(const array<bool>& lhs, const bool value) {
        return binary_opr_element_wise(lhs, value, std::less());
    }

    inline array<bool> operator<=(const array<bool>& lhs, const array<bool>& rhs) {
        return binary_opr_broadcast(lhs, rhs, std::less_equal());
    }
    inline array<bool> operator<=(const array<bool>& lhs, const bool value) {
        return binary_opr_element_wise(lhs, value, std::less_equal());
    }

    inline array<bool> operator+(const array<bool>& lhs, const array<bool>& rhs) { return lhs | rhs; }

    template <typename V>
    array<V> operator+(const array<bool>& lhs, const V& value) {
        return binary_opr_element_wise(lhs, value, std::plus());
    }
    template <typename V>
    array<V> operator+(const V& value, const array<bool>& rhs) {
        return rhs + value;
    }

    inline array<bool>& operator+=(array<bool>& lhs, const array<bool>& rhs) { return lhs |= rhs; }

    template <typename V>
    array<V> operator-(const array<bool>& lhs, const V& value) {
        return binary_opr_element_wise(lhs, value, std::minus());
    }
    template <typename V>
    array<V> operator-(const V& value, const array<bool>& rhs) {
        return binary_opr_element_wise(rhs, value, std::minus(), operations::swap_t());
    }

    inline array<bool> operator*(const array<bool>& lhs, const array<bool>& rhs) { return lhs & rhs; }

    template <typename V>
    array<V> operator*(const array<bool>& lhs, const V& value) {
        return binary_opr_element_wise(lhs, value, std::multiplies());
    }
    template <typename V>
    array<V> operator*(const V& value, const array<bool>& rhs) {
        return rhs * value;
    }

    inline array<bool>& operator*=(array<bool>& lhs, const array<bool>& rhs) { return lhs &= rhs; }

    inline array<bool> operator/(const array<bool>& lhs, const array<bool>& rhs) {
        return lhs; // if rhs.all() return lhs.as_type(dtype::uint_8)
    }

    template <typename V>
    array<V> operator/(const array<bool>& lhs, const V& value) {
        return binary_opr_element_wise(lhs, value, detail::divides());
    }
    template <typename V>
    array<V> operator/(const V& value, const array<bool>& rhs) {
        return binary_opr_element_wise(rhs, value, detail::divides(), operations::swap_t());
    }

    inline array<bool>& operator/=(array<bool>& lhs, const array<bool>& rhs) {
        return lhs; // if rhs.all() return lhs.as_type(dtype::uint_8)
    }


    inline array<bool> operator~(const array<bool>& arr) { return unary_opr_element_wise(arr, std::bit_not()); }
    inline array<bool> operator|(const bool value, const array<bool>& rhs) { return rhs | value; }
    inline array<bool> operator^(const bool value, const array<bool>& rhs) { return rhs ^ value; }
    inline array<bool> operator==(const bool value, const array<bool>& rhs) { return rhs == value; }
    inline array<bool> operator!=(const bool value, const array<bool>& rhs) { return rhs != value; }
    inline array<bool> operator>(const bool value, const array<bool>& rhs) { return rhs <= value; }
    inline array<bool> operator>=(const bool value, const array<bool>& rhs) { return rhs < value; }
    inline array<bool> operator<(const bool value, const array<bool>& rhs) { return rhs >= value; }
    inline array<bool> operator<=(const bool value, const array<bool>& rhs) { return rhs > value; }
} // namespace numcpp
