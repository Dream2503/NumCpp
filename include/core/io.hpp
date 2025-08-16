#pragma once
#include "../libs/math.hpp"

namespace numcpp {
    inline struct print_options {
        bool suppress = false;
        int8_t sign = '-';
        size_t edgeitems = 3;
        size_t linewidth = 75;
        size_t precision = 8;
        size_t threshold = 1000;
        std::string infstr = "inf";
        std::string floatmode = "maxprec";
        std::string nanstr = "nan";
        std::string seperator = " ";
    } format_options;

    template <typename T>
    std::string format(const T& value, const int equal_decimals) noexcept {
        if constexpr (is_floating_point_v<T>) {
            if (std::isnan(value)) {
                return format_options.nanstr;
            }
            if (std::isinf(value)) {
                return (value < 0 ? '-' : format_options.sign != '-' ? char() : format_options.sign) + format_options.infstr;
            }
            std::ostringstream ss;

            if (format_options.sign == '+') {
                ss << std::showpos;
            } else if (format_options.sign == ' ') {
                ss << std::showpos;
            }
            if (format_options.floatmode == "fixed" || format_options.floatmode == "maxprec_equal") {
                ss << std::fixed;
                // } else if (format_options.floatmode == "maxprec") {
                //     ss << std::fixed;
            } else if (format_options.floatmode == "scientific") {
                ss << std::scientific << std::showpoint;
            }
            ss << std::setprecision(format_options.floatmode == "maxprec_equal" && equal_decimals >= 0 ? equal_decimals : format_options.precision);
            ss << value;
            std::string res = ss.str();

            if (format_options.sign == ' ' && !res.empty() && res[0] == '+') {
                res[0] = ' ';
            }
            if (format_options.floatmode == "maxprec") {
                const auto dot_pos = res.find('.');

                if (dot_pos != std::string::npos) {
                    size_t end = res.size();

                    while (end > dot_pos + 1 && res[end - 1] == '0') {
                        --end;
                    }
                    res.resize(end);
                }
            } else if (format_options.floatmode == "scientific") {
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
        print_options temp = format_options;

        if (!format_options.suppress) {
            using V = real_t<T>;
            V min_abs_value = std::numeric_limits<V>::max();
            V max_abs_value = V(0);

            for (const auto& element : other) {
                V av = math::absolute(element);

                if (av != V(0) && av < min_abs_value) {
                    min_abs_value = av;
                }
                if (av > max_abs_value) {
                    max_abs_value = av;
                }
            }
            if (min_abs_value < 1e-4 || max_abs_value / min_abs_value > 1e3) {
                format_options.floatmode = "scientific";
            } else {
                format_options.floatmode = "maxprec";
            }
        } else {
            format_options.floatmode = "maxprec";
        }
        auto [row, col] = other.shape();
        auto will_truncate = [&](const size_t limit) -> bool { return format_options.threshold < row * col && format_options.edgeitems * 2 < limit; };
        const bool is_col_vector = (col == 1);
        const size_t width_dim = is_col_vector ? row : col;
        size_t col_width_int = 0, col_width_float = 0, pos = 0;
        std::vector col_width_vec_primary(will_truncate(std::max(row, col)) ? format_options.edgeitems * 2 : width_dim, 0ul);
        std::vector<size_t> col_width_vec_secondary;

        if constexpr (is_floating_point_v<T> || is_complex_v<T>) {
            col_width_vec_secondary = col_width_vec_primary;
        }
        for (ll_t i = 0; i < row; i++) {
            if (will_truncate(row) && i == format_options.edgeitems) {
                i = row - format_options.edgeitems - (format_options.edgeitems ? 0 : 1);
            }
            for (ll_t j = 0; j < col; j++) {
                if (will_truncate(col) && j == format_options.edgeitems) {
                    j = col - format_options.edgeitems - (format_options.edgeitems ? 0 : 1);
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
            if (line_len + s.size() >= format_options.linewidth) {
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
            if (will_truncate(row) && i == format_options.edgeitems) {
                out << " ...";

                if (is_matrix(other) && i < row - 1) {
                    out << '\n';
                    line_len = 0;
                }
                i = row - format_options.edgeitems - (format_options.edgeitems ? 0 : 1);
            }
            if (is_matrix(other)) {
                ss << (i == 0 ? "[" : " [");
            }
            pos = 0;

            for (ll_t j = 0; j < col; j++) {
                if (j > 0 || (!is_matrix(other) && i > 0)) {
                    ss << format_options.seperator;
                    write_with_wrap(ss.str());
                    ss.str("");
                }
                if (will_truncate(col) && j == format_options.edgeitems) {
                    write_with_wrap("..." + format_options.seperator);
                    j = col - format_options.edgeitems - (format_options.edgeitems ? 0 : 1);
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
                        ss << std::right;
                        ss << std::setw(is_matrix(other) ? col_width_vec_primary[pos] + col_width_vec_secondary[pos]
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
        format_options = temp;
        return out << std::flush;
    }
} // namespace numcpp
