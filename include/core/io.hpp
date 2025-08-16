#pragma once
#include "../libs/math.hpp"

namespace numcpp {
    inline struct printoptions {
        bool suppress = false;
        int8_t sign = '-';
        size_t edgeitems = 3;
        size_t linewidth = 75;
        size_t precision = 8;
        size_t threshold = 1000;
        std::string infstr = "inf";
        std::string floatmode = "maxprec";
        std::string nanstr = "nan";
        std::string sepeartor = " ";
    } _format_options;

    template <typename T>
    std::string format(const T& value, const int equal_decimals = -1) {
        if constexpr (is_floating_point_v<T>) {
            if (std::isnan(value)) {
                return _format_options.nanstr;
            }
            if (std::isinf(value)) {
                return (value < 0 ? "-" : "") + _format_options.infstr;
            }
            std::ostringstream ss;
            // oss.setf(static_cast<std::ios::fmtflags>(0), std::ios::floatfield);

            if (_format_options.sign == '+') {
                ss << std::showpos;
            } else if (_format_options.sign == ' ') {
                ss << std::showpos;
            }
            ss << std::setprecision(_format_options.floatmode == "maxprec_equal" && equal_decimals >= 0
                                        ? equal_decimals
                                        : _format_options.precision);


            if (_format_options.floatmode == "fixed" || _format_options.floatmode == "maxprec_equal") {
                ss << std::fixed;
            } else if (_format_options.floatmode == "maxprec") {
                ss << std::fixed << std::setprecision(_format_options.precision);
            } else if (_format_options.floatmode == "scientific") {
                ss << std::scientific << std::showpoint << std::setprecision(_format_options.precision);
            }
            ss << value;
            std::string res = ss.str();

            if (_format_options.sign == ' ' && !res.empty() && res[0] == '+') {
                res[0] = ' ';
            }
            if (_format_options.floatmode == "maxprec") {
                const auto dot_pos = res.find('.');

                if (dot_pos != std::string::npos) {
                    size_t end = res.size();

                    while (end > dot_pos + 1 && res[end - 1] == '0') {
                        --end;
                    }
                    res.resize(end);
                }
            } else if (_format_options.floatmode == "scientific") {
                const auto dot_pos = res.find('.'), e_pos = res.find('e');

                if (dot_pos != std::string::npos) {
                    size_t end = e_pos;

                    while (end > dot_pos + 1 && res[end - 1] == '0') {
                        --end;
                    }
                    res = res.substr(0, end) + res.substr(e_pos);
                }
            }
            return res;
        } else if constexpr (is_complex_v<T>) {
            using V = real_t<T>;
            return format<V>(value.real) + (value.imag < 0 ? '-' : '+') + format<V>(std::abs(value.imag)) + 'j';
        } else if constexpr (std::is_same_v<T, bool>) {
            return value ? "true" : "false";
        } else if constexpr (std::is_same_v<T, str>) {
            return value;
        } else {
            return std::to_string(value);
        }
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& out, const array<T>& other) {
        if (!_format_options.suppress) {
            using V = real_t<T>;
            V min_abs_value = std::numeric_limits<V>::max();
            V max_abs_value = V(0);

            for (const auto& v : other) {
                V av = math::absolute(v);
                if (av != V(0) && av < min_abs_value) {
                    min_abs_value = av;
                }
                if (av > max_abs_value) {
                    max_abs_value = av;
                }
            }
            if (min_abs_value < 1e-4 || max_abs_value / min_abs_value > 1e3) {
                _format_options.floatmode = "scientific";
            } else {
                _format_options.floatmode = "maxprec";
            }
        } else {
            _format_options.floatmode = "maxprec";
        }
        auto [row, col] = other.shape();
        const bool is_col_vector = (col == 1);
        const size_t width_dim = is_col_vector ? row : col;
        size_t col_width_int = 0, col_width_float = 0, pos = 0;
        std::vector col_width_vec_primary(_format_options.threshold < row * col &&
                                                  _format_options.edgeitems * 2 < std::max(row, col)
                                              ? _format_options.edgeitems * 2
                                              : width_dim,
                                          0ul);
        std::vector<size_t> col_width_vec_secondary;

        if constexpr (is_floating_point_v<T> || is_complex_v<T>) {
            col_width_vec_secondary = col_width_vec_primary;
        }
        for (ll_t i = 0; i < row; i++) {
            if (_format_options.threshold < row * col && _format_options.edgeitems * 2 < row &&
                i == _format_options.edgeitems) {
                i = row - _format_options.edgeitems - (_format_options.edgeitems ? 0 : 1);
            }
            for (ll_t j = 0; j < col; j++) {
                if (_format_options.threshold < row * col && _format_options.edgeitems * 2 < col &&
                    j == _format_options.edgeitems) {
                    j = col - _format_options.edgeitems - (_format_options.edgeitems ? 0 : 1);
                }
                const std::string s = format(static_cast<T>(other[{i, j}]));

                if constexpr (is_floating_point_v<T>) {
                    if (!s.contains('n')) { // nan // inf
                        const size_t dot = s.find('.'), size_primary = dot, size_secondary = s.size() - dot;

                        if (is_matrix(other)) {
                            col_width_vec_primary[pos] = std::max(col_width_vec_primary[pos], size_primary);
                            col_width_vec_secondary[pos] = std::max(col_width_vec_secondary[pos], size_secondary);
                        } else {
                            col_width_int = std::max(col_width_int, size_primary);
                            col_width_float = std::max(col_width_float, size_secondary);
                        }
                    }
                } else if (is_complex_v<T>) {
                    size_t sign = s.find('+');

                    if (sign == std::string::npos) {
                        sign = s.find('-');
                    }
                    const size_t size_primary = sign, size_secondary = s.size() - sign;

                    if (is_matrix(other)) {
                        col_width_vec_primary[pos] = std::max(col_width_vec_primary[pos], size_primary);
                        col_width_vec_secondary[pos] = std::max(col_width_vec_secondary[pos], size_secondary);
                    } else {
                        col_width_int = std::max(col_width_int, size_primary);
                        col_width_float = std::max(col_width_float, size_secondary);
                    }
                } else {
                    if (is_matrix(other)) {
                        col_width_vec_primary[pos] = std::max(col_width_vec_primary[pos], s.size());
                    } else {
                        col_width_int = std::max(col_width_int, s.size());
                    }
                }
                pos++;
            }
            pos = 0;
        }
        std::stringstream ss;
        size_t line_len = 0;
        auto write_with_wrap = [&](const std::string& s) {
            if (line_len + s.size() >= _format_options.linewidth) {
                out << "\n  ";
                line_len = 2;
            }
            out << s;
            line_len += s.size();
        };

        if (!is_scalar(other)) {
            ss << "[";
        }
        for (ll_t i = 0; i < row; i++) {
            if (_format_options.threshold < row * col && _format_options.edgeitems * 2 < row &&
                i == _format_options.edgeitems) {
                out << " ...";

                if (is_matrix(other) && i < row - 1) {
                    out << '\n';
                    line_len = 0;
                }
                i = row - _format_options.edgeitems - (_format_options.edgeitems ? 0 : 1);
            }
            if (is_matrix(other)) {
                ss << (i == 0 ? "[" : " [");
            }
            pos = 0;

            for (ll_t j = 0; j < col; j++) {
                if (j > 0 || (!is_matrix(other) && i > 0)) {
                    ss << _format_options.sepeartor;
                    write_with_wrap(ss.str());
                    ss.str("");
                }
                if (_format_options.threshold < row * col && _format_options.edgeitems * 2 < col &&
                    j == _format_options.edgeitems) {
                    write_with_wrap("..." + _format_options.sepeartor);
                    j = col - _format_options.edgeitems - (_format_options.edgeitems ? 0 : 1);
                }
                const std::string s = format(static_cast<T>(other[{i, j}]));

                if constexpr (is_floating_point_v<T>) {
                    if (!s.contains('n')) { // nan // inf
                        const size_t dot = s.find('.');
                        ss << std::right << std::setw(is_matrix(other) ? col_width_vec_primary[pos] : col_width_int);
                        ss << s.substr(0, dot);
                        ss << std::left << std::setw(is_matrix(other) ? col_width_vec_secondary[pos] : col_width_float);
                        ss << s.substr(dot);
                    } else {
                        ss << std::right
                           << std::setw(is_matrix(other) ? col_width_vec_primary[pos] + col_width_vec_secondary[pos]
                                                         : col_width_int + col_width_float);
                        ss << s;
                    }
                } else if (is_complex_v<T>) {
                    size_t sign = s.find('+');

                    if (sign == std::string::npos) {
                        sign = s.find('-');
                    }
                    ss << std::right << std::setw(is_matrix(other) ? col_width_vec_primary[pos] : col_width_int);
                    ss << s.substr(0, sign);
                    ss << std::left << std::setw(is_matrix(other) ? col_width_vec_secondary[pos] : col_width_float);
                    ss << s.substr(sign);
                } else {
                    ss << std::setw(is_matrix(other) ? col_width_vec_primary[pos] : col_width_int);
                    ss << s;
                }
                pos++;
            }
            if (is_matrix(other)) {
                ss << "]";
            }
            if (is_matrix(other) && i < row - 1) {
                write_with_wrap(ss.str());
                ss.str("");
                out << '\n';
                line_len = 0;
            }
        }
        if (!is_scalar(other)) {
            ss << "]";
            write_with_wrap(ss.str());
        }
        return out << std::flush;
    }
} // namespace numcpp
