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
  nvbench::axes_metadata axes;
  axes.add_float64_axis("Radians", {3.14, 6.28});
  axes.add_int64_axis("VecSize", {2, 3, 4}, nvbench::int64_axis_flags::none);
  axes.add_int64_axis("NumInputs",
                      {10, 15, 20},
                      nvbench::int64_axis_flags::power_of_two);
  axes.add_string_axis("Strategy", {"Recursive", "Iterative"});

  const auto states = nvbench::detail::state_generator::create(axes);

  ASSERT_MSG(states.size() == (2 * 3 * 3 * 2),
             "Actual: {} Expected: {}",
             states.size(),
             2 * 3 * 3 * 2);

  fmt::memory_buffer buffer;
  for (const nvbench::state &state : states)
  {
    fmt::format_to(buffer,
                   "Radians: {:.2f} | "
                   "VecSize: {:1d} | "
                   "NumInputs: {:7d} | "
                   "Strategy: {}\n",
                   state.get_float64("Radians"),
                   state.get_int64("VecSize"),
                   state.get_int64("NumInputs"),
                   state.get_string("Strategy"));
  }

  const std::string ref =
    R"expected(Radians: 3.14 | VecSize: 2 | NumInputs:    1024 | Strategy: Recursive
Radians: 6.28 | VecSize: 2 | NumInputs:    1024 | Strategy: Recursive
Radians: 3.14 | VecSize: 3 | NumInputs:    1024 | Strategy: Recursive
Radians: 6.28 | VecSize: 3 | NumInputs:    1024 | Strategy: Recursive
Radians: 3.14 | VecSize: 4 | NumInputs:    1024 | Strategy: Recursive
Radians: 6.28 | VecSize: 4 | NumInputs:    1024 | Strategy: Recursive
Radians: 3.14 | VecSize: 2 | NumInputs:   32768 | Strategy: Recursive
Radians: 6.28 | VecSize: 2 | NumInputs:   32768 | Strategy: Recursive
Radians: 3.14 | VecSize: 3 | NumInputs:   32768 | Strategy: Recursive
Radians: 6.28 | VecSize: 3 | NumInputs:   32768 | Strategy: Recursive
Radians: 3.14 | VecSize: 4 | NumInputs:   32768 | Strategy: Recursive
Radians: 6.28 | VecSize: 4 | NumInputs:   32768 | Strategy: Recursive
Radians: 3.14 | VecSize: 2 | NumInputs: 1048576 | Strategy: Recursive
Radians: 6.28 | VecSize: 2 | NumInputs: 1048576 | Strategy: Recursive
Radians: 3.14 | VecSize: 3 | NumInputs: 1048576 | Strategy: Recursive
Radians: 6.28 | VecSize: 3 | NumInputs: 1048576 | Strategy: Recursive
Radians: 3.14 | VecSize: 4 | NumInputs: 1048576 | Strategy: Recursive
Radians: 6.28 | VecSize: 4 | NumInputs: 1048576 | Strategy: Recursive
Radians: 3.14 | VecSize: 2 | NumInputs:    1024 | Strategy: Iterative
Radians: 6.28 | VecSize: 2 | NumInputs:    1024 | Strategy: Iterative
Radians: 3.14 | VecSize: 3 | NumInputs:    1024 | Strategy: Iterative
Radians: 6.28 | VecSize: 3 | NumInputs:    1024 | Strategy: Iterative
Radians: 3.14 | VecSize: 4 | NumInputs:    1024 | Strategy: Iterative
Radians: 6.28 | VecSize: 4 | NumInputs:    1024 | Strategy: Iterative
Radians: 3.14 | VecSize: 2 | NumInputs:   32768 | Strategy: Iterative
Radians: 6.28 | VecSize: 2 | NumInputs:   32768 | Strategy: Iterative
Radians: 3.14 | VecSize: 3 | NumInputs:   32768 | Strategy: Iterative
Radians: 6.28 | VecSize: 3 | NumInputs:   32768 | Strategy: Iterative
Radians: 3.14 | VecSize: 4 | NumInputs:   32768 | Strategy: Iterative
Radians: 6.28 | VecSize: 4 | NumInputs:   32768 | Strategy: Iterative
Radians: 3.14 | VecSize: 2 | NumInputs: 1048576 | Strategy: Iterative
Radians: 6.28 | VecSize: 2 | NumInputs: 1048576 | Strategy: Iterative
Radians: 3.14 | VecSize: 3 | NumInputs: 1048576 | Strategy: Iterative
Radians: 6.28 | VecSize: 3 | NumInputs: 1048576 | Strategy: Iterative
Radians: 3.14 | VecSize: 4 | NumInputs: 1048576 | Strategy: Iterative
Radians: 6.28 | VecSize: 4 | NumInputs: 1048576 | Strategy: Iterative
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
