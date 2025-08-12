#pragma once
#include <algorithm>
#include <complex>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <type_traits>
#include <variant>
#include "libs/detail.hpp"

namespace numcpp {
    struct index_t;
    shape_t broadcast_shape(const shape_t&, const shape_t&);
    index_t broadcast_index(const index_t&, const shape_t&);
} // namespace numcpp

#include "core/array.hpp"
#include "core/masked_array.hpp"
#include "core/operators.hpp"
#include "libs/indexing.hpp"
#include "libs/math.hpp"
#include "libs/numeric.hpp"
#include "libs/traits.hpp"
#include "libs/types.hpp"
#include "libs/ufunc.hpp"
#include "libs/utils.hpp"
