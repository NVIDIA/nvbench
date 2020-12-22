#include <nvbench/float64_axis.cuh>

#include "test_asserts.cuh"

void test_empty()
{
  nvbench::float64_axis axis("Empty");
  ASSERT(axis.get_name() == "Empty");
  ASSERT(axis.get_type() == nvbench::axis_type::float64);
  ASSERT(axis.get_size() == 0);

  axis.set_inputs({});

  ASSERT(axis.get_size() == 0);
}

void test_basic()
{
  nvbench::float64_axis axis("Basic");
  ASSERT(axis.get_name() == "Basic");

  axis.set_inputs({-100.3, 0., 2064.15});

  ASSERT(axis.get_size() == 3);
  ASSERT(axis.get_value(0) == -100.3);
  ASSERT(axis.get_input_string(0) == "-100.3");
  ASSERT(axis.get_description(0) == "");
  ASSERT(axis.get_value(1) == 0.);
  ASSERT(axis.get_input_string(1) == "0");
  ASSERT(axis.get_description(1) == "");
  ASSERT(axis.get_value(2) == 2064.15);
  ASSERT(axis.get_input_string(2) == "2064.15");
  ASSERT(axis.get_description(2) == "");
}

int main()
{
  test_empty();
  test_basic();
}
