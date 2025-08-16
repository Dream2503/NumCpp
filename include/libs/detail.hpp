#pragma once

namespace numcpp::detail {
    template <typename T>
    constexpr T division_by_zero_warning(T left, const char error[]) noexcept {
        std::cerr << "RuntimeWarning: divide by zero encountered in " << error << std::endl;

        if constexpr (is_floating_point_v<T>) {
            if (left == T()) {
                return nan;
            }
            return std::copysign(inf, left);
        } else if constexpr (is_complex_v<T>) {
            if (left == T()) {
                return T(nan, nan);
            }
            return T(std::copysign(inf, left.real()), std::copysign(inf, left.imag()));
        } else if constexpr (is_integral_v<T>) {
            return T();
        }
        return left;
    }

    constexpr auto divides() noexcept {
        return []<typename L, typename R>(L left, R right) noexcept -> promote_t<L, R> {
            if (right == L()) {
                return division_by_zero_warning(left, __PRETTY_FUNCTION__);
            }
            return left / right;
        };
    }

    constexpr auto modulus() noexcept {
        return []<typename L, typename R>(L left, R right) noexcept -> promote_t<L, R> {
            if (right == L()) {
                return division_by_zero_warning(left, __PRETTY_FUNCTION__);
            }
            return left % right;
        };
    }
} // namespace numcpp::detail
