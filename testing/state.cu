#include <nvbench/state.cuh>

#include <nvbench/types.cuh>

#include "test_asserts.cuh"

// Subclass to gain access to protected members for testing:
struct state_tester : public nvbench::state
{
  state_tester()
      : nvbench::state()
  {}

  template <typename T>
  void set_param(std::string name, T &&value)
  {
    this->state::m_axis_values.set_value(std::move(name),
                                         nvbench::named_values::value_type{
                                           std::forward<T>(value)});
  }
};

void test_params()
{
  // Build a state param by param
  state_tester state1;
  state1.set_param("TestInt", nvbench::int64_t{22});
  state1.set_param("TestFloat", nvbench::float64_t{3.14});
  state1.set_param("TestString", "A String!");

  ASSERT(state1.get_int64("TestInt") == nvbench::int64_t{22});
  ASSERT(state1.get_float64("TestFloat") == nvbench::float64_t{3.14});
  ASSERT(state1.get_string("TestString") == "A String!");
}

int main() { test_params(); }
