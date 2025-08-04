#pragma once
#include "traits.hpp"

namespace numcpp::none {
    inline constexpr int axis = static_cast<int>(none_t());
    inline constexpr auto base = nullptr, out = nullptr, where = nullptr;
    inline constexpr auto size = static_cast<size_t>(none_t<size_t>());
    inline auto shape = shape_t(1, size);
} // namespace numcpp::none
