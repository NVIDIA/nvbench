#include <nvbench/int64_axis.cuh>

#include "testing/test_asserts.cuh"

#include <fmt/format.h>

void test_basic()
{
  nvbench::int64_axis axis{"BasicAxis"};
  ASSERT(axis.get_name() == "BasicAxis");
  ASSERT(axis.get_type() == nvbench::axis_type::int64);
  ASSERT(!axis.is_power_of_two());

  axis.set_inputs({0, 1, 2, 3, 7, 6, 5, 4});
  ASSERT(axis.get_size() == 8);

  std::vector<nvbench::int64_t> ref{0, 1, 2, 3, 7, 6, 5, 4};
  ASSERT(axis.get_inputs() == ref);
  ASSERT(axis.get_values() == ref);

  for (size_t i = 0; i < 8; ++i)
  {
    ASSERT(axis.get_input_string(i) == fmt::to_string(ref[i]));
    ASSERT(axis.get_description(i).empty());
  }
}

void test_power_of_two()
{
  nvbench::int64_axis axis{"POTAxis", nvbench::int64_axis_flags::power_of_two};
  ASSERT(axis.get_name() == "POTAxis");
  ASSERT(axis.get_type() == nvbench::axis_type::int64);
  ASSERT(axis.is_power_of_two());

  axis.set_inputs({0, 1, 2, 3, 7, 6, 5, 4});
  ASSERT(axis.get_size() == 8);

  std::vector<nvbench::int64_t> ref_inputs{0, 1, 2, 3, 7, 6, 5, 4};
  std::vector<nvbench::int64_t> ref_values{1, 2, 4, 8, 128, 64, 32, 16};
  ASSERT(axis.get_inputs() == ref_inputs);
  ASSERT(axis.get_values() == ref_values);

  for (size_t i = 0; i < 8; ++i)
  {
//    fmt::print("{}: {}\n", i, axis.get_description(i));
    ASSERT(axis.get_input_string(i) == fmt::to_string(ref_inputs[i]));
    ASSERT(axis.get_description(i) ==
           fmt::format("2^{} = {}", ref_inputs[i], ref_values[i]));
  }
}

int main()
{
  test_basic();
  test_power_of_two();

  return EXIT_SUCCESS;
}
