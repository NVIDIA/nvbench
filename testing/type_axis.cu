#include <nvbench/type_axis.cuh>

#include <nvbench/types.cuh>

#include "test_asserts.cuh"

#include <fmt/format.h>

void test_empty()
{
  nvbench::type_axis axis("Basic");
  ASSERT(axis.get_name() == "Basic");
  ASSERT(axis.get_type() == nvbench::axis_type::type);
  ASSERT(axis.get_size() == 0);

  axis.set_inputs<nvbench::type_list<>>();

  ASSERT(axis.get_size() == 0);
}

void test_single()
{
  nvbench::type_axis axis("Single");
  ASSERT(axis.get_name() == "Single");

  axis.set_inputs<nvbench::type_list<nvbench::int32_t>>();

  ASSERT(axis.get_size() == 1);
  ASSERT(axis.get_input_string(0) == "I32");
  ASSERT(axis.get_description(0) == "int32_t");
}

void test_several()
{
  nvbench::type_axis axis("Several");
  ASSERT(axis.get_name() == "Several");

  axis.set_inputs<
    nvbench::type_list<nvbench::int32_t, nvbench::float64_t, bool>>();

  ASSERT(axis.get_size() == 3);
  ASSERT(axis.get_input_string(0) == "I32");
  ASSERT(axis.get_description(0) == "int32_t");
  ASSERT(axis.get_input_string(1) == "F64");
  ASSERT(axis.get_description(1) == "double");
  ASSERT(axis.get_input_string(2) == "bool");
  ASSERT(axis.get_description(2) == "");
}

int main()
{
  test_empty();
  test_single();
  test_several();
}
