#include <nvbench/detail/state_generator.cuh>

#include <nvbench/axes_metadata.cuh>
#include <nvbench/axis_base.cuh>

#include "test_asserts.cuh"

#include <fmt/format.h>

struct state_generator_tester : nvbench::detail::state_generator
{
  using nvbench::detail::state_generator::add_axis;
  using nvbench::detail::state_generator::get_current_indices;
  using nvbench::detail::state_generator::get_number_of_states;
  using nvbench::detail::state_generator::init;
  using nvbench::detail::state_generator::iter_valid;
  using nvbench::detail::state_generator::next;
};

void test_empty()
{
  // no axes = one state
  state_generator_tester sg;
  ASSERT(sg.get_number_of_states() == 1);
  sg.init();
  ASSERT(sg.iter_valid());
  sg.next();
  ASSERT(!sg.iter_valid());
}

void test_single_state()
{
  // one single-value axis = one state
  state_generator_tester sg;
  sg.add_axis("OnlyAxis", nvbench::axis_type::string, 1);
  ASSERT(sg.get_number_of_states() == 1);
  sg.init();
  ASSERT(sg.iter_valid());
  ASSERT(sg.get_current_indices().size() == 1);
  ASSERT(sg.get_current_indices()[0].axis == "OnlyAxis");
  ASSERT(sg.get_current_indices()[0].index == 0);
  ASSERT(sg.get_current_indices()[0].size == 1);
  ASSERT(sg.get_current_indices()[0].type == nvbench::axis_type::string);

  sg.next();
  ASSERT(!sg.iter_valid());
}

void test_basic()
{
  state_generator_tester sg;
  sg.add_axis("Axis1", nvbench::axis_type::string, 2);
  sg.add_axis("Axis2", nvbench::axis_type::string, 3);
  sg.add_axis("Axis3", nvbench::axis_type::string, 3);
  sg.add_axis("Axis4", nvbench::axis_type::string, 2);

  ASSERT_MSG(sg.get_number_of_states() == (2 * 3 * 3 * 2),
             "Actual: {} Expected: {}",
             sg.get_number_of_states(),
             2 * 3 * 3 * 2);

  fmt::memory_buffer buffer;
  fmt::memory_buffer line;
  std::size_t line_num{0};
  for (sg.init(); sg.iter_valid(); sg.next())
  {
    line.clear();
    fmt::format_to(line, "| {:^2}", line_num++);
    for (auto &axis_index : sg.get_current_indices())
    {
      ASSERT(axis_index.type == nvbench::axis_type::string);
      fmt::format_to(line,
                     " | {}: {}/{}",
                     axis_index.axis,
                     axis_index.index,
                     axis_index.size);
    }
    fmt::format_to(buffer, "{} |\n", fmt::to_string(line));
  }

  const std::string ref =
    R"expected(| 0  | Axis1: 0/2 | Axis2: 0/3 | Axis3: 0/3 | Axis4: 0/2 |
| 1  | Axis1: 1/2 | Axis2: 0/3 | Axis3: 0/3 | Axis4: 0/2 |
| 2  | Axis1: 0/2 | Axis2: 1/3 | Axis3: 0/3 | Axis4: 0/2 |
| 3  | Axis1: 1/2 | Axis2: 1/3 | Axis3: 0/3 | Axis4: 0/2 |
| 4  | Axis1: 0/2 | Axis2: 2/3 | Axis3: 0/3 | Axis4: 0/2 |
| 5  | Axis1: 1/2 | Axis2: 2/3 | Axis3: 0/3 | Axis4: 0/2 |
| 6  | Axis1: 0/2 | Axis2: 0/3 | Axis3: 1/3 | Axis4: 0/2 |
| 7  | Axis1: 1/2 | Axis2: 0/3 | Axis3: 1/3 | Axis4: 0/2 |
| 8  | Axis1: 0/2 | Axis2: 1/3 | Axis3: 1/3 | Axis4: 0/2 |
| 9  | Axis1: 1/2 | Axis2: 1/3 | Axis3: 1/3 | Axis4: 0/2 |
| 10 | Axis1: 0/2 | Axis2: 2/3 | Axis3: 1/3 | Axis4: 0/2 |
| 11 | Axis1: 1/2 | Axis2: 2/3 | Axis3: 1/3 | Axis4: 0/2 |
| 12 | Axis1: 0/2 | Axis2: 0/3 | Axis3: 2/3 | Axis4: 0/2 |
| 13 | Axis1: 1/2 | Axis2: 0/3 | Axis3: 2/3 | Axis4: 0/2 |
| 14 | Axis1: 0/2 | Axis2: 1/3 | Axis3: 2/3 | Axis4: 0/2 |
| 15 | Axis1: 1/2 | Axis2: 1/3 | Axis3: 2/3 | Axis4: 0/2 |
| 16 | Axis1: 0/2 | Axis2: 2/3 | Axis3: 2/3 | Axis4: 0/2 |
| 17 | Axis1: 1/2 | Axis2: 2/3 | Axis3: 2/3 | Axis4: 0/2 |
| 18 | Axis1: 0/2 | Axis2: 0/3 | Axis3: 0/3 | Axis4: 1/2 |
| 19 | Axis1: 1/2 | Axis2: 0/3 | Axis3: 0/3 | Axis4: 1/2 |
| 20 | Axis1: 0/2 | Axis2: 1/3 | Axis3: 0/3 | Axis4: 1/2 |
| 21 | Axis1: 1/2 | Axis2: 1/3 | Axis3: 0/3 | Axis4: 1/2 |
| 22 | Axis1: 0/2 | Axis2: 2/3 | Axis3: 0/3 | Axis4: 1/2 |
| 23 | Axis1: 1/2 | Axis2: 2/3 | Axis3: 0/3 | Axis4: 1/2 |
| 24 | Axis1: 0/2 | Axis2: 0/3 | Axis3: 1/3 | Axis4: 1/2 |
| 25 | Axis1: 1/2 | Axis2: 0/3 | Axis3: 1/3 | Axis4: 1/2 |
| 26 | Axis1: 0/2 | Axis2: 1/3 | Axis3: 1/3 | Axis4: 1/2 |
| 27 | Axis1: 1/2 | Axis2: 1/3 | Axis3: 1/3 | Axis4: 1/2 |
| 28 | Axis1: 0/2 | Axis2: 2/3 | Axis3: 1/3 | Axis4: 1/2 |
| 29 | Axis1: 1/2 | Axis2: 2/3 | Axis3: 1/3 | Axis4: 1/2 |
| 30 | Axis1: 0/2 | Axis2: 0/3 | Axis3: 2/3 | Axis4: 1/2 |
| 31 | Axis1: 1/2 | Axis2: 0/3 | Axis3: 2/3 | Axis4: 1/2 |
| 32 | Axis1: 0/2 | Axis2: 1/3 | Axis3: 2/3 | Axis4: 1/2 |
| 33 | Axis1: 1/2 | Axis2: 1/3 | Axis3: 2/3 | Axis4: 1/2 |
| 34 | Axis1: 0/2 | Axis2: 2/3 | Axis3: 2/3 | Axis4: 1/2 |
| 35 | Axis1: 1/2 | Axis2: 2/3 | Axis3: 2/3 | Axis4: 1/2 |
)expected";

  const std::string test = fmt::to_string(buffer);
  ASSERT_MSG(test == ref,
             fmt::format("Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test));
}

void test_create()
{
  using floats = nvbench::type_list<nvbench::float32_t, nvbench::float64_t>;
  using ints   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
  using misc   = nvbench::type_list<void, bool>;

  using type_axes = nvbench::type_list<floats, ints, misc>;

  nvbench::axes_metadata axes;
  axes.set_type_axes_names<type_axes>({"Floats", "Ints", "Misc"});
  axes.add_float64_axis("Radians", {3.14, 6.28});
  axes.add_int64_axis("VecSize", {2, 3, 4}, nvbench::int64_axis_flags::none);
  axes.add_int64_axis("NumInputs",
                      {10, 15, 20},
                      nvbench::int64_axis_flags::power_of_two);
  axes.add_string_axis("Strategy", {"Recursive", "Iterative"});

  const std::vector<std::vector<nvbench::state>> states =
    nvbench::detail::state_generator::create(axes);

  // Outer vector has one entry per type_config
  // 2 (Floats) * 2 (Ints) * 2 (Misc) = 8 total type_configs
  ASSERT(states.size() == 8);

  // Inner vectors have one entry per non-type config:
  // 2 (Radians) * 3 (VecSize) * 3 (NumInputs) * 2 (Strategy) = 36
  for (const auto &inner_states : states)
  {
    ASSERT(inner_states.size() == 36);
  }

  fmt::memory_buffer buffer;
  std::string table_format =
    "| {:^5} | {:^10} | {:^6} | {:^4} | {:^4} | {:^7} | {:^7} | {:^9} | {:^9} |\n";

  fmt::format_to(buffer, "\n");
  fmt::format_to(buffer,
                 table_format,
                 "State",
                 "TypeConfig",
                 "Floats",
                 "Ints",
                 "Misc",
                 "Radians",
                 "VecSize",
                 "NumInputs",
                 "Strategy");

  std::size_t type_config = 0;
  std::size_t config = 0;
  for (const auto &inner_states : states)
  {
    for (const nvbench::state &state : inner_states)
    {
      fmt::format_to(buffer,
                     table_format,
                     config++,
                     type_config,
                     state.get_string("Floats"),
                     state.get_string("Ints"),
                     state.get_string("Misc"),
                     state.get_float64("Radians"),
                     state.get_int64("VecSize"),
                     state.get_int64("NumInputs"),
                     state.get_string("Strategy"));
    }
    type_config++;
  }

  const std::string ref =
    R"expected(
| State | TypeConfig | Floats | Ints | Misc | Radians | VecSize | NumInputs | Strategy  |
|   0   |     0      |  F32   | I32  | void |  3.14   |    2    |   1024    | Recursive |
|   1   |     0      |  F32   | I32  | void |  6.28   |    2    |   1024    | Recursive |
|   2   |     0      |  F32   | I32  | void |  3.14   |    3    |   1024    | Recursive |
|   3   |     0      |  F32   | I32  | void |  6.28   |    3    |   1024    | Recursive |
|   4   |     0      |  F32   | I32  | void |  3.14   |    4    |   1024    | Recursive |
|   5   |     0      |  F32   | I32  | void |  6.28   |    4    |   1024    | Recursive |
|   6   |     0      |  F32   | I32  | void |  3.14   |    2    |   32768   | Recursive |
|   7   |     0      |  F32   | I32  | void |  6.28   |    2    |   32768   | Recursive |
|   8   |     0      |  F32   | I32  | void |  3.14   |    3    |   32768   | Recursive |
|   9   |     0      |  F32   | I32  | void |  6.28   |    3    |   32768   | Recursive |
|  10   |     0      |  F32   | I32  | void |  3.14   |    4    |   32768   | Recursive |
|  11   |     0      |  F32   | I32  | void |  6.28   |    4    |   32768   | Recursive |
|  12   |     0      |  F32   | I32  | void |  3.14   |    2    |  1048576  | Recursive |
|  13   |     0      |  F32   | I32  | void |  6.28   |    2    |  1048576  | Recursive |
|  14   |     0      |  F32   | I32  | void |  3.14   |    3    |  1048576  | Recursive |
|  15   |     0      |  F32   | I32  | void |  6.28   |    3    |  1048576  | Recursive |
|  16   |     0      |  F32   | I32  | void |  3.14   |    4    |  1048576  | Recursive |
|  17   |     0      |  F32   | I32  | void |  6.28   |    4    |  1048576  | Recursive |
|  18   |     0      |  F32   | I32  | void |  3.14   |    2    |   1024    | Iterative |
|  19   |     0      |  F32   | I32  | void |  6.28   |    2    |   1024    | Iterative |
|  20   |     0      |  F32   | I32  | void |  3.14   |    3    |   1024    | Iterative |
|  21   |     0      |  F32   | I32  | void |  6.28   |    3    |   1024    | Iterative |
|  22   |     0      |  F32   | I32  | void |  3.14   |    4    |   1024    | Iterative |
|  23   |     0      |  F32   | I32  | void |  6.28   |    4    |   1024    | Iterative |
|  24   |     0      |  F32   | I32  | void |  3.14   |    2    |   32768   | Iterative |
|  25   |     0      |  F32   | I32  | void |  6.28   |    2    |   32768   | Iterative |
|  26   |     0      |  F32   | I32  | void |  3.14   |    3    |   32768   | Iterative |
|  27   |     0      |  F32   | I32  | void |  6.28   |    3    |   32768   | Iterative |
|  28   |     0      |  F32   | I32  | void |  3.14   |    4    |   32768   | Iterative |
|  29   |     0      |  F32   | I32  | void |  6.28   |    4    |   32768   | Iterative |
|  30   |     0      |  F32   | I32  | void |  3.14   |    2    |  1048576  | Iterative |
|  31   |     0      |  F32   | I32  | void |  6.28   |    2    |  1048576  | Iterative |
|  32   |     0      |  F32   | I32  | void |  3.14   |    3    |  1048576  | Iterative |
|  33   |     0      |  F32   | I32  | void |  6.28   |    3    |  1048576  | Iterative |
|  34   |     0      |  F32   | I32  | void |  3.14   |    4    |  1048576  | Iterative |
|  35   |     0      |  F32   | I32  | void |  6.28   |    4    |  1048576  | Iterative |
|  36   |     1      |  F64   | I32  | void |  3.14   |    2    |   1024    | Recursive |
|  37   |     1      |  F64   | I32  | void |  6.28   |    2    |   1024    | Recursive |
|  38   |     1      |  F64   | I32  | void |  3.14   |    3    |   1024    | Recursive |
|  39   |     1      |  F64   | I32  | void |  6.28   |    3    |   1024    | Recursive |
|  40   |     1      |  F64   | I32  | void |  3.14   |    4    |   1024    | Recursive |
|  41   |     1      |  F64   | I32  | void |  6.28   |    4    |   1024    | Recursive |
|  42   |     1      |  F64   | I32  | void |  3.14   |    2    |   32768   | Recursive |
|  43   |     1      |  F64   | I32  | void |  6.28   |    2    |   32768   | Recursive |
|  44   |     1      |  F64   | I32  | void |  3.14   |    3    |   32768   | Recursive |
|  45   |     1      |  F64   | I32  | void |  6.28   |    3    |   32768   | Recursive |
|  46   |     1      |  F64   | I32  | void |  3.14   |    4    |   32768   | Recursive |
|  47   |     1      |  F64   | I32  | void |  6.28   |    4    |   32768   | Recursive |
|  48   |     1      |  F64   | I32  | void |  3.14   |    2    |  1048576  | Recursive |
|  49   |     1      |  F64   | I32  | void |  6.28   |    2    |  1048576  | Recursive |
|  50   |     1      |  F64   | I32  | void |  3.14   |    3    |  1048576  | Recursive |
|  51   |     1      |  F64   | I32  | void |  6.28   |    3    |  1048576  | Recursive |
|  52   |     1      |  F64   | I32  | void |  3.14   |    4    |  1048576  | Recursive |
|  53   |     1      |  F64   | I32  | void |  6.28   |    4    |  1048576  | Recursive |
|  54   |     1      |  F64   | I32  | void |  3.14   |    2    |   1024    | Iterative |
|  55   |     1      |  F64   | I32  | void |  6.28   |    2    |   1024    | Iterative |
|  56   |     1      |  F64   | I32  | void |  3.14   |    3    |   1024    | Iterative |
|  57   |     1      |  F64   | I32  | void |  6.28   |    3    |   1024    | Iterative |
|  58   |     1      |  F64   | I32  | void |  3.14   |    4    |   1024    | Iterative |
|  59   |     1      |  F64   | I32  | void |  6.28   |    4    |   1024    | Iterative |
|  60   |     1      |  F64   | I32  | void |  3.14   |    2    |   32768   | Iterative |
|  61   |     1      |  F64   | I32  | void |  6.28   |    2    |   32768   | Iterative |
|  62   |     1      |  F64   | I32  | void |  3.14   |    3    |   32768   | Iterative |
|  63   |     1      |  F64   | I32  | void |  6.28   |    3    |   32768   | Iterative |
|  64   |     1      |  F64   | I32  | void |  3.14   |    4    |   32768   | Iterative |
|  65   |     1      |  F64   | I32  | void |  6.28   |    4    |   32768   | Iterative |
|  66   |     1      |  F64   | I32  | void |  3.14   |    2    |  1048576  | Iterative |
|  67   |     1      |  F64   | I32  | void |  6.28   |    2    |  1048576  | Iterative |
|  68   |     1      |  F64   | I32  | void |  3.14   |    3    |  1048576  | Iterative |
|  69   |     1      |  F64   | I32  | void |  6.28   |    3    |  1048576  | Iterative |
|  70   |     1      |  F64   | I32  | void |  3.14   |    4    |  1048576  | Iterative |
|  71   |     1      |  F64   | I32  | void |  6.28   |    4    |  1048576  | Iterative |
|  72   |     2      |  F32   | I64  | void |  3.14   |    2    |   1024    | Recursive |
|  73   |     2      |  F32   | I64  | void |  6.28   |    2    |   1024    | Recursive |
|  74   |     2      |  F32   | I64  | void |  3.14   |    3    |   1024    | Recursive |
|  75   |     2      |  F32   | I64  | void |  6.28   |    3    |   1024    | Recursive |
|  76   |     2      |  F32   | I64  | void |  3.14   |    4    |   1024    | Recursive |
|  77   |     2      |  F32   | I64  | void |  6.28   |    4    |   1024    | Recursive |
|  78   |     2      |  F32   | I64  | void |  3.14   |    2    |   32768   | Recursive |
|  79   |     2      |  F32   | I64  | void |  6.28   |    2    |   32768   | Recursive |
|  80   |     2      |  F32   | I64  | void |  3.14   |    3    |   32768   | Recursive |
|  81   |     2      |  F32   | I64  | void |  6.28   |    3    |   32768   | Recursive |
|  82   |     2      |  F32   | I64  | void |  3.14   |    4    |   32768   | Recursive |
|  83   |     2      |  F32   | I64  | void |  6.28   |    4    |   32768   | Recursive |
|  84   |     2      |  F32   | I64  | void |  3.14   |    2    |  1048576  | Recursive |
|  85   |     2      |  F32   | I64  | void |  6.28   |    2    |  1048576  | Recursive |
|  86   |     2      |  F32   | I64  | void |  3.14   |    3    |  1048576  | Recursive |
|  87   |     2      |  F32   | I64  | void |  6.28   |    3    |  1048576  | Recursive |
|  88   |     2      |  F32   | I64  | void |  3.14   |    4    |  1048576  | Recursive |
|  89   |     2      |  F32   | I64  | void |  6.28   |    4    |  1048576  | Recursive |
|  90   |     2      |  F32   | I64  | void |  3.14   |    2    |   1024    | Iterative |
|  91   |     2      |  F32   | I64  | void |  6.28   |    2    |   1024    | Iterative |
|  92   |     2      |  F32   | I64  | void |  3.14   |    3    |   1024    | Iterative |
|  93   |     2      |  F32   | I64  | void |  6.28   |    3    |   1024    | Iterative |
|  94   |     2      |  F32   | I64  | void |  3.14   |    4    |   1024    | Iterative |
|  95   |     2      |  F32   | I64  | void |  6.28   |    4    |   1024    | Iterative |
|  96   |     2      |  F32   | I64  | void |  3.14   |    2    |   32768   | Iterative |
|  97   |     2      |  F32   | I64  | void |  6.28   |    2    |   32768   | Iterative |
|  98   |     2      |  F32   | I64  | void |  3.14   |    3    |   32768   | Iterative |
|  99   |     2      |  F32   | I64  | void |  6.28   |    3    |   32768   | Iterative |
|  100  |     2      |  F32   | I64  | void |  3.14   |    4    |   32768   | Iterative |
|  101  |     2      |  F32   | I64  | void |  6.28   |    4    |   32768   | Iterative |
|  102  |     2      |  F32   | I64  | void |  3.14   |    2    |  1048576  | Iterative |
|  103  |     2      |  F32   | I64  | void |  6.28   |    2    |  1048576  | Iterative |
|  104  |     2      |  F32   | I64  | void |  3.14   |    3    |  1048576  | Iterative |
|  105  |     2      |  F32   | I64  | void |  6.28   |    3    |  1048576  | Iterative |
|  106  |     2      |  F32   | I64  | void |  3.14   |    4    |  1048576  | Iterative |
|  107  |     2      |  F32   | I64  | void |  6.28   |    4    |  1048576  | Iterative |
|  108  |     3      |  F64   | I64  | void |  3.14   |    2    |   1024    | Recursive |
|  109  |     3      |  F64   | I64  | void |  6.28   |    2    |   1024    | Recursive |
|  110  |     3      |  F64   | I64  | void |  3.14   |    3    |   1024    | Recursive |
|  111  |     3      |  F64   | I64  | void |  6.28   |    3    |   1024    | Recursive |
|  112  |     3      |  F64   | I64  | void |  3.14   |    4    |   1024    | Recursive |
|  113  |     3      |  F64   | I64  | void |  6.28   |    4    |   1024    | Recursive |
|  114  |     3      |  F64   | I64  | void |  3.14   |    2    |   32768   | Recursive |
|  115  |     3      |  F64   | I64  | void |  6.28   |    2    |   32768   | Recursive |
|  116  |     3      |  F64   | I64  | void |  3.14   |    3    |   32768   | Recursive |
|  117  |     3      |  F64   | I64  | void |  6.28   |    3    |   32768   | Recursive |
|  118  |     3      |  F64   | I64  | void |  3.14   |    4    |   32768   | Recursive |
|  119  |     3      |  F64   | I64  | void |  6.28   |    4    |   32768   | Recursive |
|  120  |     3      |  F64   | I64  | void |  3.14   |    2    |  1048576  | Recursive |
|  121  |     3      |  F64   | I64  | void |  6.28   |    2    |  1048576  | Recursive |
|  122  |     3      |  F64   | I64  | void |  3.14   |    3    |  1048576  | Recursive |
|  123  |     3      |  F64   | I64  | void |  6.28   |    3    |  1048576  | Recursive |
|  124  |     3      |  F64   | I64  | void |  3.14   |    4    |  1048576  | Recursive |
|  125  |     3      |  F64   | I64  | void |  6.28   |    4    |  1048576  | Recursive |
|  126  |     3      |  F64   | I64  | void |  3.14   |    2    |   1024    | Iterative |
|  127  |     3      |  F64   | I64  | void |  6.28   |    2    |   1024    | Iterative |
|  128  |     3      |  F64   | I64  | void |  3.14   |    3    |   1024    | Iterative |
|  129  |     3      |  F64   | I64  | void |  6.28   |    3    |   1024    | Iterative |
|  130  |     3      |  F64   | I64  | void |  3.14   |    4    |   1024    | Iterative |
|  131  |     3      |  F64   | I64  | void |  6.28   |    4    |   1024    | Iterative |
|  132  |     3      |  F64   | I64  | void |  3.14   |    2    |   32768   | Iterative |
|  133  |     3      |  F64   | I64  | void |  6.28   |    2    |   32768   | Iterative |
|  134  |     3      |  F64   | I64  | void |  3.14   |    3    |   32768   | Iterative |
|  135  |     3      |  F64   | I64  | void |  6.28   |    3    |   32768   | Iterative |
|  136  |     3      |  F64   | I64  | void |  3.14   |    4    |   32768   | Iterative |
|  137  |     3      |  F64   | I64  | void |  6.28   |    4    |   32768   | Iterative |
|  138  |     3      |  F64   | I64  | void |  3.14   |    2    |  1048576  | Iterative |
|  139  |     3      |  F64   | I64  | void |  6.28   |    2    |  1048576  | Iterative |
|  140  |     3      |  F64   | I64  | void |  3.14   |    3    |  1048576  | Iterative |
|  141  |     3      |  F64   | I64  | void |  6.28   |    3    |  1048576  | Iterative |
|  142  |     3      |  F64   | I64  | void |  3.14   |    4    |  1048576  | Iterative |
|  143  |     3      |  F64   | I64  | void |  6.28   |    4    |  1048576  | Iterative |
|  144  |     4      |  F32   | I32  | bool |  3.14   |    2    |   1024    | Recursive |
|  145  |     4      |  F32   | I32  | bool |  6.28   |    2    |   1024    | Recursive |
|  146  |     4      |  F32   | I32  | bool |  3.14   |    3    |   1024    | Recursive |
|  147  |     4      |  F32   | I32  | bool |  6.28   |    3    |   1024    | Recursive |
|  148  |     4      |  F32   | I32  | bool |  3.14   |    4    |   1024    | Recursive |
|  149  |     4      |  F32   | I32  | bool |  6.28   |    4    |   1024    | Recursive |
|  150  |     4      |  F32   | I32  | bool |  3.14   |    2    |   32768   | Recursive |
|  151  |     4      |  F32   | I32  | bool |  6.28   |    2    |   32768   | Recursive |
|  152  |     4      |  F32   | I32  | bool |  3.14   |    3    |   32768   | Recursive |
|  153  |     4      |  F32   | I32  | bool |  6.28   |    3    |   32768   | Recursive |
|  154  |     4      |  F32   | I32  | bool |  3.14   |    4    |   32768   | Recursive |
|  155  |     4      |  F32   | I32  | bool |  6.28   |    4    |   32768   | Recursive |
|  156  |     4      |  F32   | I32  | bool |  3.14   |    2    |  1048576  | Recursive |
|  157  |     4      |  F32   | I32  | bool |  6.28   |    2    |  1048576  | Recursive |
|  158  |     4      |  F32   | I32  | bool |  3.14   |    3    |  1048576  | Recursive |
|  159  |     4      |  F32   | I32  | bool |  6.28   |    3    |  1048576  | Recursive |
|  160  |     4      |  F32   | I32  | bool |  3.14   |    4    |  1048576  | Recursive |
|  161  |     4      |  F32   | I32  | bool |  6.28   |    4    |  1048576  | Recursive |
|  162  |     4      |  F32   | I32  | bool |  3.14   |    2    |   1024    | Iterative |
|  163  |     4      |  F32   | I32  | bool |  6.28   |    2    |   1024    | Iterative |
|  164  |     4      |  F32   | I32  | bool |  3.14   |    3    |   1024    | Iterative |
|  165  |     4      |  F32   | I32  | bool |  6.28   |    3    |   1024    | Iterative |
|  166  |     4      |  F32   | I32  | bool |  3.14   |    4    |   1024    | Iterative |
|  167  |     4      |  F32   | I32  | bool |  6.28   |    4    |   1024    | Iterative |
|  168  |     4      |  F32   | I32  | bool |  3.14   |    2    |   32768   | Iterative |
|  169  |     4      |  F32   | I32  | bool |  6.28   |    2    |   32768   | Iterative |
|  170  |     4      |  F32   | I32  | bool |  3.14   |    3    |   32768   | Iterative |
|  171  |     4      |  F32   | I32  | bool |  6.28   |    3    |   32768   | Iterative |
|  172  |     4      |  F32   | I32  | bool |  3.14   |    4    |   32768   | Iterative |
|  173  |     4      |  F32   | I32  | bool |  6.28   |    4    |   32768   | Iterative |
|  174  |     4      |  F32   | I32  | bool |  3.14   |    2    |  1048576  | Iterative |
|  175  |     4      |  F32   | I32  | bool |  6.28   |    2    |  1048576  | Iterative |
|  176  |     4      |  F32   | I32  | bool |  3.14   |    3    |  1048576  | Iterative |
|  177  |     4      |  F32   | I32  | bool |  6.28   |    3    |  1048576  | Iterative |
|  178  |     4      |  F32   | I32  | bool |  3.14   |    4    |  1048576  | Iterative |
|  179  |     4      |  F32   | I32  | bool |  6.28   |    4    |  1048576  | Iterative |
|  180  |     5      |  F64   | I32  | bool |  3.14   |    2    |   1024    | Recursive |
|  181  |     5      |  F64   | I32  | bool |  6.28   |    2    |   1024    | Recursive |
|  182  |     5      |  F64   | I32  | bool |  3.14   |    3    |   1024    | Recursive |
|  183  |     5      |  F64   | I32  | bool |  6.28   |    3    |   1024    | Recursive |
|  184  |     5      |  F64   | I32  | bool |  3.14   |    4    |   1024    | Recursive |
|  185  |     5      |  F64   | I32  | bool |  6.28   |    4    |   1024    | Recursive |
|  186  |     5      |  F64   | I32  | bool |  3.14   |    2    |   32768   | Recursive |
|  187  |     5      |  F64   | I32  | bool |  6.28   |    2    |   32768   | Recursive |
|  188  |     5      |  F64   | I32  | bool |  3.14   |    3    |   32768   | Recursive |
|  189  |     5      |  F64   | I32  | bool |  6.28   |    3    |   32768   | Recursive |
|  190  |     5      |  F64   | I32  | bool |  3.14   |    4    |   32768   | Recursive |
|  191  |     5      |  F64   | I32  | bool |  6.28   |    4    |   32768   | Recursive |
|  192  |     5      |  F64   | I32  | bool |  3.14   |    2    |  1048576  | Recursive |
|  193  |     5      |  F64   | I32  | bool |  6.28   |    2    |  1048576  | Recursive |
|  194  |     5      |  F64   | I32  | bool |  3.14   |    3    |  1048576  | Recursive |
|  195  |     5      |  F64   | I32  | bool |  6.28   |    3    |  1048576  | Recursive |
|  196  |     5      |  F64   | I32  | bool |  3.14   |    4    |  1048576  | Recursive |
|  197  |     5      |  F64   | I32  | bool |  6.28   |    4    |  1048576  | Recursive |
|  198  |     5      |  F64   | I32  | bool |  3.14   |    2    |   1024    | Iterative |
|  199  |     5      |  F64   | I32  | bool |  6.28   |    2    |   1024    | Iterative |
|  200  |     5      |  F64   | I32  | bool |  3.14   |    3    |   1024    | Iterative |
|  201  |     5      |  F64   | I32  | bool |  6.28   |    3    |   1024    | Iterative |
|  202  |     5      |  F64   | I32  | bool |  3.14   |    4    |   1024    | Iterative |
|  203  |     5      |  F64   | I32  | bool |  6.28   |    4    |   1024    | Iterative |
|  204  |     5      |  F64   | I32  | bool |  3.14   |    2    |   32768   | Iterative |
|  205  |     5      |  F64   | I32  | bool |  6.28   |    2    |   32768   | Iterative |
|  206  |     5      |  F64   | I32  | bool |  3.14   |    3    |   32768   | Iterative |
|  207  |     5      |  F64   | I32  | bool |  6.28   |    3    |   32768   | Iterative |
|  208  |     5      |  F64   | I32  | bool |  3.14   |    4    |   32768   | Iterative |
|  209  |     5      |  F64   | I32  | bool |  6.28   |    4    |   32768   | Iterative |
|  210  |     5      |  F64   | I32  | bool |  3.14   |    2    |  1048576  | Iterative |
|  211  |     5      |  F64   | I32  | bool |  6.28   |    2    |  1048576  | Iterative |
|  212  |     5      |  F64   | I32  | bool |  3.14   |    3    |  1048576  | Iterative |
|  213  |     5      |  F64   | I32  | bool |  6.28   |    3    |  1048576  | Iterative |
|  214  |     5      |  F64   | I32  | bool |  3.14   |    4    |  1048576  | Iterative |
|  215  |     5      |  F64   | I32  | bool |  6.28   |    4    |  1048576  | Iterative |
|  216  |     6      |  F32   | I64  | bool |  3.14   |    2    |   1024    | Recursive |
|  217  |     6      |  F32   | I64  | bool |  6.28   |    2    |   1024    | Recursive |
|  218  |     6      |  F32   | I64  | bool |  3.14   |    3    |   1024    | Recursive |
|  219  |     6      |  F32   | I64  | bool |  6.28   |    3    |   1024    | Recursive |
|  220  |     6      |  F32   | I64  | bool |  3.14   |    4    |   1024    | Recursive |
|  221  |     6      |  F32   | I64  | bool |  6.28   |    4    |   1024    | Recursive |
|  222  |     6      |  F32   | I64  | bool |  3.14   |    2    |   32768   | Recursive |
|  223  |     6      |  F32   | I64  | bool |  6.28   |    2    |   32768   | Recursive |
|  224  |     6      |  F32   | I64  | bool |  3.14   |    3    |   32768   | Recursive |
|  225  |     6      |  F32   | I64  | bool |  6.28   |    3    |   32768   | Recursive |
|  226  |     6      |  F32   | I64  | bool |  3.14   |    4    |   32768   | Recursive |
|  227  |     6      |  F32   | I64  | bool |  6.28   |    4    |   32768   | Recursive |
|  228  |     6      |  F32   | I64  | bool |  3.14   |    2    |  1048576  | Recursive |
|  229  |     6      |  F32   | I64  | bool |  6.28   |    2    |  1048576  | Recursive |
|  230  |     6      |  F32   | I64  | bool |  3.14   |    3    |  1048576  | Recursive |
|  231  |     6      |  F32   | I64  | bool |  6.28   |    3    |  1048576  | Recursive |
|  232  |     6      |  F32   | I64  | bool |  3.14   |    4    |  1048576  | Recursive |
|  233  |     6      |  F32   | I64  | bool |  6.28   |    4    |  1048576  | Recursive |
|  234  |     6      |  F32   | I64  | bool |  3.14   |    2    |   1024    | Iterative |
|  235  |     6      |  F32   | I64  | bool |  6.28   |    2    |   1024    | Iterative |
|  236  |     6      |  F32   | I64  | bool |  3.14   |    3    |   1024    | Iterative |
|  237  |     6      |  F32   | I64  | bool |  6.28   |    3    |   1024    | Iterative |
|  238  |     6      |  F32   | I64  | bool |  3.14   |    4    |   1024    | Iterative |
|  239  |     6      |  F32   | I64  | bool |  6.28   |    4    |   1024    | Iterative |
|  240  |     6      |  F32   | I64  | bool |  3.14   |    2    |   32768   | Iterative |
|  241  |     6      |  F32   | I64  | bool |  6.28   |    2    |   32768   | Iterative |
|  242  |     6      |  F32   | I64  | bool |  3.14   |    3    |   32768   | Iterative |
|  243  |     6      |  F32   | I64  | bool |  6.28   |    3    |   32768   | Iterative |
|  244  |     6      |  F32   | I64  | bool |  3.14   |    4    |   32768   | Iterative |
|  245  |     6      |  F32   | I64  | bool |  6.28   |    4    |   32768   | Iterative |
|  246  |     6      |  F32   | I64  | bool |  3.14   |    2    |  1048576  | Iterative |
|  247  |     6      |  F32   | I64  | bool |  6.28   |    2    |  1048576  | Iterative |
|  248  |     6      |  F32   | I64  | bool |  3.14   |    3    |  1048576  | Iterative |
|  249  |     6      |  F32   | I64  | bool |  6.28   |    3    |  1048576  | Iterative |
|  250  |     6      |  F32   | I64  | bool |  3.14   |    4    |  1048576  | Iterative |
|  251  |     6      |  F32   | I64  | bool |  6.28   |    4    |  1048576  | Iterative |
|  252  |     7      |  F64   | I64  | bool |  3.14   |    2    |   1024    | Recursive |
|  253  |     7      |  F64   | I64  | bool |  6.28   |    2    |   1024    | Recursive |
|  254  |     7      |  F64   | I64  | bool |  3.14   |    3    |   1024    | Recursive |
|  255  |     7      |  F64   | I64  | bool |  6.28   |    3    |   1024    | Recursive |
|  256  |     7      |  F64   | I64  | bool |  3.14   |    4    |   1024    | Recursive |
|  257  |     7      |  F64   | I64  | bool |  6.28   |    4    |   1024    | Recursive |
|  258  |     7      |  F64   | I64  | bool |  3.14   |    2    |   32768   | Recursive |
|  259  |     7      |  F64   | I64  | bool |  6.28   |    2    |   32768   | Recursive |
|  260  |     7      |  F64   | I64  | bool |  3.14   |    3    |   32768   | Recursive |
|  261  |     7      |  F64   | I64  | bool |  6.28   |    3    |   32768   | Recursive |
|  262  |     7      |  F64   | I64  | bool |  3.14   |    4    |   32768   | Recursive |
|  263  |     7      |  F64   | I64  | bool |  6.28   |    4    |   32768   | Recursive |
|  264  |     7      |  F64   | I64  | bool |  3.14   |    2    |  1048576  | Recursive |
|  265  |     7      |  F64   | I64  | bool |  6.28   |    2    |  1048576  | Recursive |
|  266  |     7      |  F64   | I64  | bool |  3.14   |    3    |  1048576  | Recursive |
|  267  |     7      |  F64   | I64  | bool |  6.28   |    3    |  1048576  | Recursive |
|  268  |     7      |  F64   | I64  | bool |  3.14   |    4    |  1048576  | Recursive |
|  269  |     7      |  F64   | I64  | bool |  6.28   |    4    |  1048576  | Recursive |
|  270  |     7      |  F64   | I64  | bool |  3.14   |    2    |   1024    | Iterative |
|  271  |     7      |  F64   | I64  | bool |  6.28   |    2    |   1024    | Iterative |
|  272  |     7      |  F64   | I64  | bool |  3.14   |    3    |   1024    | Iterative |
|  273  |     7      |  F64   | I64  | bool |  6.28   |    3    |   1024    | Iterative |
|  274  |     7      |  F64   | I64  | bool |  3.14   |    4    |   1024    | Iterative |
|  275  |     7      |  F64   | I64  | bool |  6.28   |    4    |   1024    | Iterative |
|  276  |     7      |  F64   | I64  | bool |  3.14   |    2    |   32768   | Iterative |
|  277  |     7      |  F64   | I64  | bool |  6.28   |    2    |   32768   | Iterative |
|  278  |     7      |  F64   | I64  | bool |  3.14   |    3    |   32768   | Iterative |
|  279  |     7      |  F64   | I64  | bool |  6.28   |    3    |   32768   | Iterative |
|  280  |     7      |  F64   | I64  | bool |  3.14   |    4    |   32768   | Iterative |
|  281  |     7      |  F64   | I64  | bool |  6.28   |    4    |   32768   | Iterative |
|  282  |     7      |  F64   | I64  | bool |  3.14   |    2    |  1048576  | Iterative |
|  283  |     7      |  F64   | I64  | bool |  6.28   |    2    |  1048576  | Iterative |
|  284  |     7      |  F64   | I64  | bool |  3.14   |    3    |  1048576  | Iterative |
|  285  |     7      |  F64   | I64  | bool |  6.28   |    3    |  1048576  | Iterative |
|  286  |     7      |  F64   | I64  | bool |  3.14   |    4    |  1048576  | Iterative |
|  287  |     7      |  F64   | I64  | bool |  6.28   |    4    |  1048576  | Iterative |
)expected";

  const std::string test = fmt::to_string(buffer);
  ASSERT_MSG(test == ref,
             fmt::format("Expected:\n\"{}\"\n\nActual:\n\"{}\"", ref, test));
}

int main()
{
  test_empty();
  test_single_state();
  test_basic();
  test_create();
}
