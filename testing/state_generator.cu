#include <nvbench/detail/state_generator.cuh>

#include <nvbench/axis_base.cuh>

#include "test_asserts.cuh"

#include <fmt/format.h>

void test_empty()
{
  // no axes = one state
  nvbench::detail::state_generator sg;
  ASSERT(sg.get_number_of_states() == 1);
  sg.init();
  ASSERT(sg.iter_valid());
  sg.next();
  ASSERT(!sg.iter_valid());
}

void test_single_state()
{
  // one single-value axis = one state
  nvbench::detail::state_generator sg;
  sg.add_axis("OnlyAxis", nvbench::axis_type::string, 1);
  ASSERT(sg.get_number_of_states() == 1);
  sg.init();
  ASSERT(sg.iter_valid());
  sg.next();
  ASSERT(!sg.iter_valid());
}

void test_basic()
{
  nvbench::detail::state_generator sg;
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

int main()
{
  test_empty();
  test_single_state();
  test_basic();
}
