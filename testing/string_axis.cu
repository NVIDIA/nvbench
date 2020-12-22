#include <nvbench/string_axis.cuh>

#include "test_asserts.cuh"

void test_empty()
{
  nvbench::string_axis axis("Empty");
  ASSERT(axis.get_name() == "Empty");
  ASSERT(axis.get_type() == nvbench::axis_type::string);
  ASSERT(axis.get_size() == 0);

  axis.set_inputs({});

  ASSERT(axis.get_size() == 0);
}

void test_basic()
{
  nvbench::string_axis axis("Basic");
  ASSERT(axis.get_name() == "Basic");

  axis.set_inputs({"String 1", "String 2", "String 3"});

  ASSERT(axis.get_size() == 3);
  ASSERT(axis.get_value(0) == "String 1");
  ASSERT(axis.get_input_string(0) == "String 1");
  ASSERT(axis.get_description(0) == "");
  ASSERT(axis.get_value(1) == "String 2");
  ASSERT(axis.get_input_string(1) == "String 2");
  ASSERT(axis.get_description(1) == "");
  ASSERT(axis.get_value(2) == "String 3");
  ASSERT(axis.get_input_string(2) == "String 3");
  ASSERT(axis.get_description(2) == "");
}

int main()
{
  test_empty();
  test_basic();
}
