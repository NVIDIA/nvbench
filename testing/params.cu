#include <nvbench/params.cuh>

#include "test_asserts.cuh"

void test_basic()
{
  nvbench::params params;
  params.add_string_param("Axis 1", "Value 1");
  params.add_int64_param("Axis 2", 2);
  params.add_float64_param("Axis 3", 3.);
  params.add_string_param("Axis 4", "Value 4");
  params.add_int64_param("Axis 5", 5);
  params.add_float64_param("Axis 6", 6.);

  ASSERT(params.get_string_param("Axis 1") == "Value 1");
  ASSERT(params.get_int64_param("Axis 2") == 2);
  ASSERT(params.get_float64_param("Axis 3") == 3.);
  ASSERT(params.get_string_param("Axis 4") == "Value 4");
  ASSERT(params.get_int64_param("Axis 5") == 5);
  ASSERT(params.get_float64_param("Axis 6") == 6.);
}

int main()
{
  test_basic();
}
