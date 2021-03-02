#include <nvbench/option_parser.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/benchmark_manager.cuh>
#include <nvbench/csv_format.cuh>
#include <nvbench/markdown_format.cuh>
#include <nvbench/output_format.cuh>
#include <nvbench/range.cuh>

#include <nvbench/detail/throw.cuh>

#include <fmt/format.h>

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace
{

//==============================================================================
// helpers types for using std::string_view with std::regex
using sv_citer          = std::string_view::const_iterator;
using sv_match          = std::match_results<sv_citer>;
using sv_submatch       = std::sub_match<sv_citer>;
using sv_regex_iterator = std::regex_iterator<sv_citer>;
std::string_view submatch_to_sv(const sv_submatch &in)
{
  // This will be much easier in C++20, but this string_view constructor is
  // painfully absent until then:
  // return {in.first, in.second};

  // C++17 version:
  if (in.first == in.second)
  {
    return {};
  }

  // We have to use the (ptr, len) ctor
  return {&*in.first, static_cast<std::size_t>(in.length())};
}
//==============================================================================

// These numeric overloads /could/ be written in a single function using
// std::from_chars, but charconv is a mess on GCC. Even GCC 10 only partially
// implements it (missing support for floats).
//
// So we're stuck with materializing a std::string and calling std::stoX(). Ah
// well. At least it's not istream.
void parse(std::string_view input, nvbench::int32_t &val)
{
  val = std::stoi(std::string(input));
}

void parse(std::string_view input, nvbench::int64_t &val)
{
  val = std::stoll(std::string(input));
}

void parse(std::string_view input, nvbench::float64_t &val)
{
  val = std::stod(std::string(input));
}

void parse(std::string_view input, std::string &val) { val = input; }

// Parses a list of values "<val1>, <val2>, <val3>, ..." into a vector:
template <typename T>
std::vector<T> parse_list_values(std::string_view list_spec)
{
  std::vector<T> result;

  static const std::regex value_regex{
    "\\s*"     // Whitespace
    "([^,]+?)" // Single value
    "\\s*"     // Whitespace
    "(?:,|$)"  // Delimiters
  };

  auto values_begin =
    sv_regex_iterator(list_spec.cbegin(), list_spec.cend(), value_regex);
  auto values_end = sv_regex_iterator{};
  while (values_begin != values_end)
  {
    auto match          = *values_begin++;
    std::string_view sv = submatch_to_sv(match[1]);

    T val;
    parse(sv, val);
    result.push_back(std::move(val));
  }

  return result;
}

// Parses a range specification "<start> : <stop> [ : <stride> ]" and returns
// a vector filled with the specified range.
template <typename T>
std::vector<T> parse_range_values(std::string_view range_spec,
                                  nvbench::wrapped_type<T>)
{
  std::vector<T> range_params;

  static const std::regex value_regex{
    "\\s*"     // Whitespace
    "([^:]+?)" // Single value
    "\\s*"     // Whitespace
    "(?:$|:)"  // Delimiters
  };

  auto values_begin =
    sv_regex_iterator(range_spec.cbegin(), range_spec.cend(), value_regex);
  auto values_end = sv_regex_iterator{};
  for (; values_begin != values_end; ++values_begin)
  {
    auto match          = *values_begin;
    std::string_view sv = submatch_to_sv(match[1]);
    T val;
    parse(sv, val);
    range_params.push_back(std::move(val));
  }

  // Convert the parsed values into a range:
  if (range_params.size() != 2 && range_params.size() != 3)
  {
    NVBENCH_THROW(std::runtime_error,
                  "Expected 2 or 3 values for range specification: {}",
                  range_spec);
  }

  const T first  = range_params[0];
  const T last   = range_params[1];
  const T stride = range_params.size() == 3 ? range_params[2] : T{1};

  return nvbench::range<T, T>(first, last, stride);
}

// Disable range parsing for string types
std::vector<std::string> parse_range_values(std::string_view range_spec,
                                            nvbench::wrapped_type<std::string>)
{
  NVBENCH_THROW(std::runtime_error,
                "Cannot use range syntax for string axis specification: `{}`.",
                range_spec);
}

template <typename T>
std::vector<T> parse_values(std::string_view value_spec)
{
  // Match a list of values, e.g. [3, 4, 5, 6, 7]
  static const std::regex list_regex{"^"        // Start of string
                                     "\\["      // Literal [
                                     "\\s*"     // Whitespace
                                     "("        // Start value capture
                                     "[^\\]]+?" // One or more "not ]"
                                     ","        // Literal '.'
                                     "[^\\]]+?" // One or more "not ]"
                                     ")"        // End value capture
                                     "\\s*"     // Whitespace
                                     "\\]"      // Literal ]
                                     "$"};      // EOS

  // Match a range specification, e.g. "[1:10:2]"
  static const std::regex range_regex{"^"        // Start of string
                                      "\\["      // Literal [
                                      "\\s*"     // Whitespace
                                      "("        // Start value capture
                                      "[^\\]]+?" // One or more "not ]"
                                      ":"        // Literal ':'
                                      "[^\\]]+?" // One or more "not ]"
                                      ")"        // End value capture
                                      "\\s*"     // Whitespace
                                      "\\]"      // Literal ]
                                      "$"};      // EOS

  // Match a single value, e.g. "XXX" or "[XXX]"
  static const std::regex single_regex{"^"          // Start of string
                                       "\\[?"       // Literal `[`
                                       "\\s*"       // Whitespace
                                       "("          // Start value capture
                                       "[^,\\]:]+?" // One or more "not ,]:"
                                       ")"          // End value capture
                                       "\\s*"       // Whitespace
                                       "\\]?"       // Optional Literal ]
                                       "$"};        // EOS

  sv_match match;
  if (std::regex_search(value_spec.cbegin(),
                        value_spec.cend(),
                        match,
                        list_regex))
  {
    return parse_list_values<T>(submatch_to_sv(match[1]));
  }
  else if (std::regex_search(value_spec.cbegin(),
                             value_spec.cend(),
                             match,
                             range_regex))
  {
    return parse_range_values(submatch_to_sv(match[1]),
                              nvbench::wrapped_type<T>{});
  }
  else if (std::regex_search(value_spec.cbegin(),
                             value_spec.cend(),
                             match,
                             single_regex))
  {
    T val;
    parse(submatch_to_sv(match[1]), val);
    return {val};
  }
  else
  {
    NVBENCH_THROW(std::runtime_error,
                  "Invalid axis value spec: {}",
                  value_spec);
  }
}

// Parse an axis specification into a 3-tuple of string_views containing the
// axis name, flags, and values.
auto parse_axis_key_flag_value_spec(const std::string &spec)
{
  static const std::regex spec_regex{
    "\\s*"        // Optional Whitespace
    "([^\\[:]+?)" // Axis name
    "\\s*"        // Optional Whitespace

    "(?:"        // Start optional non-capture group for tag
    "\\["        //  - Literal [
    "\\s*"       //  - Optional Whitespace
    "([^\\]]*?)" //  - Flag spec
    "\\s*"       //  - Optional Whitespace
    "\\]"        //  - Literal ]
    ")?"         // End optional tag group

    "\\s*"  // Optional Whitespace
    "="     // Literal =
    "\\s*"  // Optional Whitespace
    "(.+?)" // Value spec
    "\\s*"  // Optional Whitespace
    "$"     // end
  };

  sv_match match;
  const std::string_view spec_sv = spec;
  if (!std::regex_search(spec_sv.cbegin(), spec_sv.cend(), match, spec_regex))
  {
    NVBENCH_THROW(std::runtime_error, "{}", "Bad format.");
  }

  // Extract the matches:
  const auto name = submatch_to_sv(match[1]);
  const auto flag = submatch_to_sv(match[2]);
  const auto vals = submatch_to_sv(match[3]);
  return std::tie(name, flag, vals);
}

} // namespace

namespace nvbench
{

// Defined here to avoid including <fstream> in the header.
option_parser::option_parser()  = default;
option_parser::~option_parser() = default;

void option_parser::parse(int argc, char const *const *argv)
{
  m_args.clear();
  m_args.reserve(static_cast<std::size_t>(argc));
  for (int i = 0; i < argc; ++i)
  {
    m_args.emplace_back(argv[i]);
  }

  parse_impl();
}

void option_parser::parse(std::vector<std::string> args)
{
  m_args = std::move(args);
  parse_impl();
}

void option_parser::parse_impl()
{
  m_global_args.clear();

  this->parse_range(m_args.cbegin(), m_args.cend());

  if (m_benchmarks.empty())
  {
    // If no benchmarks were specified, add all:
    const auto &benches = nvbench::benchmark_manager::get().get_benchmarks();
    for (const auto &bench_ptr : benches)
    {
      m_benchmarks.push_back(bench_ptr->clone());
      this->replay_global_args();
    }
  }
}

void option_parser::parse_range(option_parser::arg_iterator_t first,
                                option_parser::arg_iterator_t last)
{
  // The first arg may be the executable name:
  if (first != last && !first->empty() && first->front() != '-')
  {
    first++;
  }

  auto check_params = [&first, &last](std::size_t num_params) {
    const std::size_t rem_args = std::distance(first, last) - 1;
    if (rem_args < num_params)
    {
      NVBENCH_THROW(std::runtime_error,
                    "Option '{}' requires {} parameters, {} provided.",
                    *first,
                    num_params,
                    rem_args);
    }
  };

  while (first < last)
  {
    const auto &arg = *first;

    if (arg == "--list" || arg == "-l")
    {
      this->print_list();
      std::exit(0);
    }
    else if (arg == "--color")
    {
      m_color_md_stdout_printer = true;
      first += 1;
    }
    else if (arg == "--markdown" || arg == "--md")
    {
      check_params(1);
      this->add_markdown_format(first[1]);
      first += 2;
    }
    else if (arg == "--csv")
    {
      check_params(1);
      this->add_csv_format(first[1]);
      first += 2;
    }
    else if (arg == "--benchmark" || arg == "-b")
    {
      check_params(1);
      this->add_benchmark(first[1]);
      first += 2;
    }
    else if (arg == "--devices" || arg == "-d" || arg == "--device")
    {
      check_params(1);
      this->update_devices(first[1]);
      first += 2;
    }
    else if (arg == "--axis" || arg == "-a")
    {
      check_params(1);
      this->update_axis(first[1]);
      first += 2;
    }
    else if (arg == "--min-samples")
    {
      check_params(1);
      this->update_int64_prop(first[0], first[1]);
      first += 2;
    }
    else if (arg == "--min-time" || arg == "--max-noise" ||
             arg == "--skip-time" || arg == "--timeout")
    {
      check_params(1);
      this->update_float64_prop(first[0], first[1]);
      first += 2;
    }
    else
    {
      NVBENCH_THROW(std::runtime_error,
                    "Unrecognized command-line argument: `{}`.",
                    arg);
    }
  }
}

void option_parser::add_markdown_format(const std::string &spec)
try
{
  std::ostream &stream = this->output_format_spec_to_ostream(spec);
  auto &printer        = m_printer.emplace<nvbench::markdown_format>(stream);
  if (spec == "stdout")
  {
    printer.set_color(m_color_md_stdout_printer);
  }
}
catch (std::exception &e)
{
  NVBENCH_THROW(std::runtime_error,
                "Error while adding markdown output for `{}`:\n{}",
                spec,
                e.what());
}

void option_parser::add_csv_format(const std::string &spec)
try
{
  std::ostream &stream = this->output_format_spec_to_ostream(spec);
  m_printer.emplace<nvbench::csv_format>(stream);
}
catch (std::exception &e)
{
  NVBENCH_THROW(std::runtime_error,
                "Error while adding csv output for `{}`:\n{}",
                spec,
                e.what());
}

std::ostream &
option_parser::output_format_spec_to_ostream(const std::string &spec)
{
  if (spec == "stdout")
  {
    return std::cout;
  }
  else if (spec == "stderr")
  {
    return std::cerr;
  }
  else
  {
    m_ofstream_storage.push_back(std::make_unique<std::ofstream>());
    auto &file_stream = *m_ofstream_storage.back();

    // Throw if file can't open
    file_stream.exceptions(file_stream.exceptions() | std::ios::failbit);

    try
    {
      file_stream.open(spec);
    }
    catch (...)
    {
      m_ofstream_storage.pop_back();
      throw;
    }

    return file_stream;
  }
}

void option_parser::print_list() const
{
  const auto &bench_mgr = nvbench::benchmark_manager::get();

  nvbench::markdown_format printer{std::cout};
  printer.print_device_info();
  printer.print_benchmark_list(bench_mgr.get_benchmarks());
}

void option_parser::add_benchmark(const std::string &name)
try
{
  const auto &mgr = nvbench::benchmark_manager::get();

  std::unique_ptr<nvbench::benchmark_base> new_bench;

  nvbench::int64_t idx{-1};
  try
  {
    ::parse(name, idx);
  }
  catch (std::invalid_argument &)
  {}

  m_benchmarks.push_back(idx >= 0 ? mgr.get_benchmark(idx).clone()
                                  : mgr.get_benchmark(name).clone());

  // Initialize the new benchmark with any global arguments:
  this->replay_global_args();
}
catch (std::exception &e)
{
  NVBENCH_THROW(std::runtime_error,
                "Error parsing --benchmark `{}`:\n{}",
                name,
                e.what());
}

void option_parser::replay_global_args()
{
  this->parse_range(m_global_args.cbegin(), m_global_args.cend());
}

void option_parser::update_devices(const std::string &devices)
try
{
  // If no active benchmark, save args as global.
  if (m_benchmarks.empty())
  {
    m_global_args.push_back("-devices");
    m_global_args.push_back(devices);
    return;
  }

  benchmark_base &bench = *m_benchmarks.back();
  bench.set_devices(parse_values<nvbench::int32_t>(devices));
}
catch (std::exception &e)
{
  NVBENCH_THROW(std::runtime_error,
                "Error parsing --devices `{}`:\n{}",
                devices,
                e.what());
}

void option_parser::update_axis(const std::string &spec)
try
{
  // Valid examples:
  // - "NumInputs [pow2] = [10 : 30 : 5]" <- Range specification (::)
  // - "UniqueKeys[]=[10,15,20,25,30]"    <- List spec {,,...}
  // - "Quality=0.781"                    <- Single value
  // - "Quality=[0.0 : 1.0 : 0.1]"
  // - "ValueType = [ I32, F32, U64 ]"
  // - "ValueType=I32"
  // - "RNG [] = [ Uniform, Gaussian ]"
  //
  // Generally: "<AxisName> [<optional flags>] = <input spec>"
  //
  // Axis/Flag spec: "<AxisName>" (no flags)
  // Axis/Flag spec: "<AxisName> []" (no flags)
  // Axis/Flag spec: "<AxisName> [pow2]" (flags=`pow2`)
  // Value spec: "[ <v1, <v2>, ... ]" <- Explicit values
  // Value spec: "[<start> : <stop>]" <- Range, inclusive start/stop
  // Value spec: "[<start> : <stop> : <stride>]" <- Range, explicit stride

  // If no active benchmark, save args as global.
  if (m_benchmarks.empty())
  {
    m_global_args.push_back("--axis");
    m_global_args.push_back(spec);
    return;
  }

  benchmark_base &bench = *m_benchmarks.back();

  const auto [name, flags, values] = parse_axis_key_flag_value_spec(spec);
  nvbench::axis_base &axis         = bench.get_axes().get_axis(name);
  switch (axis.get_type())
  {
    case axis_type::type:
      this->update_type_axis(static_cast<nvbench::type_axis &>(axis),
                             values,
                             flags);
      break;

    case axis_type::int64:
      this->update_int64_axis(static_cast<nvbench::int64_axis &>(axis),
                              values,
                              flags);
      break;

    case axis_type::float64:
      this->update_float64_axis(static_cast<nvbench::float64_axis &>(axis),
                                values,
                                flags);

      break;

    case axis_type::string:
      this->update_string_axis(static_cast<nvbench::string_axis &>(axis),
                               values,
                               flags);

      break;

    default:
      // Internal error, this should never happen:
      NVBENCH_THROW(std::runtime_error,
                    "Internal error: invalid axis type enum '{}'",
                    static_cast<int>(axis.get_type()));
  }
}
catch (std::exception &e)
{
  NVBENCH_THROW(std::runtime_error,
                "Error parsing --axis `{}`:\n{}",
                spec,
                e.what());
}

void option_parser::update_int64_axis(int64_axis &axis,
                                      std::string_view value_spec,
                                      std::string_view flag_spec)
{
  // Validate flags:
  int64_axis_flags flags;
  if (flag_spec.empty())
  {
    flags = int64_axis_flags::none;
  }
  else if (flag_spec == "pow2")
  {
    flags = int64_axis_flags::power_of_two;
  }
  else
  {
    NVBENCH_THROW(std::runtime_error,
                  "Invalid flag for int64 axis: `{}`",
                  flag_spec);
  }

  auto input_values = parse_values<nvbench::int64_t>(value_spec);

  axis.set_inputs(std::move(input_values), flags);
}

void option_parser::update_float64_axis(float64_axis &axis,
                                        std::string_view value_spec,
                                        std::string_view flag_spec)
{
  // Validate flags:
  if (!flag_spec.empty())
  {
    NVBENCH_THROW(std::runtime_error,
                  "Invalid flag for float64 axis: `{}`",
                  flag_spec);
  }

  auto input_values = parse_values<nvbench::float64_t>(value_spec);

  axis.set_inputs(std::move(input_values));
}

void option_parser::update_string_axis(string_axis &axis,
                                       std::string_view value_spec,
                                       std::string_view flag_spec)
{
  // Validate flags:
  if (!flag_spec.empty())
  {
    NVBENCH_THROW(std::runtime_error,
                  "Invalid flag for string axis: `{}`",
                  flag_spec);
  }

  auto input_values = parse_values<std::string>(value_spec);

  axis.set_inputs(std::move(input_values));
}

void option_parser::update_type_axis(type_axis &axis,
                                     std::string_view value_spec,
                                     std::string_view flag_spec)
{
  // Validate flags:
  if (!flag_spec.empty())
  {
    NVBENCH_THROW(std::runtime_error,
                  "Invalid flag for type axis: `{}`",
                  flag_spec);
  }

  auto input_values = parse_values<std::string>(value_spec);

  axis.set_active_inputs(input_values);
}

void option_parser::update_int64_prop(const std::string &prop_arg,
                                      const std::string &prop_val)
try
{
  // If no active benchmark, save args as global.
  if (m_benchmarks.empty())
  {
    m_global_args.push_back(prop_arg);
    m_global_args.push_back(prop_val);
    return;
  }

  benchmark_base &bench = *m_benchmarks.back();

  nvbench::int64_t value{};
  ::parse(prop_val, value);
  if (prop_arg == "--min-samples")
  {
    bench.set_min_samples(value);
  }
  else
  {
    NVBENCH_THROW(std::runtime_error, "Unrecognized property: `{}`", prop_arg);
  }
}
catch (std::exception &e)
{
  NVBENCH_THROW(std::runtime_error,
                "Error parsing `{} {}`:\n{}",
                prop_arg,
                prop_val,
                e.what());
}

void option_parser::update_float64_prop(const std::string &prop_arg,
                                        const std::string &prop_val)
try
{
  // If no active benchmark, save args as global.
  if (m_benchmarks.empty())
  {
    m_global_args.push_back(prop_arg);
    m_global_args.push_back(prop_val);
    return;
  }

  benchmark_base &bench = *m_benchmarks.back();

  nvbench::float64_t value{};
  ::parse(prop_val, value);
  if (prop_arg == "--min-time")
  {
    bench.set_min_time(value);
  }
  else if (prop_arg == "--max-noise")
  {
    bench.set_max_noise(value);
  }
  else if (prop_arg == "--skip-time")
  {
    bench.set_skip_time(value);
  }
  else if (prop_arg == "--timeout")
  {
    bench.set_timeout(value);
  }
  else
  {
    NVBENCH_THROW(std::runtime_error, "Unrecognized property: `{}`", prop_arg);
  }
}
catch (std::exception &e)
{
  NVBENCH_THROW(std::runtime_error,
                "Error parsing `{} {}`:\n{}",
                prop_arg,
                prop_val,
                e.what());
}

nvbench::output_format &option_parser::get_printer()
{
  if (m_printer.get_output_count() == 0)
  {
    this->add_markdown_format("stdout");
  }
  return m_printer;
}

} // namespace nvbench
