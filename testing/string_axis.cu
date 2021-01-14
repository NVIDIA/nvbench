#include <nvbench/string_axis.cuh>

#include "test_asserts.cuh"

void test_empty()
{
  nvbench::string_axis axis("Empty");
  axis.set_inputs({});

  ASSERT(axis.get_name() == "Empty");
  ASSERT(axis.get_type() == nvbench::axis_type::string);
  ASSERT(axis.get_size() == 0);
  ASSERT(axis.get_size() == 0);

  const auto clone_base = axis.clone();
  ASSERT(clone_base.get() != nullptr);
  const auto *clone =
    dynamic_cast<const nvbench::string_axis *>(clone_base.get());
  ASSERT(clone != nullptr);

  ASSERT(clone->get_name() == "Empty");
  ASSERT(clone->get_type() == nvbench::axis_type::string);
  ASSERT(clone->get_size() == 0);
  ASSERT(clone->get_size() == 0);
}

void test_basic()
{
  nvbench::string_axis axis("Basic");
  axis.set_inputs({"String 1", "String 2", "String 3"});

  ASSERT(axis.get_name() == "Basic");
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

  const auto clone_base = axis.clone();
  ASSERT(clone_base.get() != nullptr);
  const auto *clone =
    dynamic_cast<const nvbench::string_axis *>(clone_base.get());
  ASSERT(clone != nullptr);

  ASSERT(clone->get_name() == "Basic");
  ASSERT(clone->get_size() == 3);
  ASSERT(clone->get_value(0) == "String 1");
  ASSERT(clone->get_input_string(0) == "String 1");
  ASSERT(clone->get_description(0) == "");
  ASSERT(clone->get_value(1) == "String 2");
  ASSERT(clone->get_input_string(1) == "String 2");
  ASSERT(clone->get_description(1) == "");
  ASSERT(clone->get_value(2) == "String 3");
  ASSERT(clone->get_input_string(2) == "String 3");
  ASSERT(clone->get_description(2) == "");
}

int main()
{
  test_empty();
  test_basic();
}
