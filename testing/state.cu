#include <nvbench/state.cuh>

#include <nvbench/types.cuh>

#include "test_asserts.cuh"

// Subclass to gain access to protected members for testing:
struct state_tester : public nvbench::state
{
  using params_type = nvbench::state::params_type;

  state_tester()
      : nvbench::state()
  {}
  explicit state_tester(params_type params)
      : nvbench::state{std::move(params)}
  {}

  template <typename... Args>
  void set_param(Args &&...args)
  {
    this->state::set_param(std::forward<Args>(args)...);
  }

  const auto &get_params() const { return m_params; }
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

  // Construct a state from the parameter map built above:
  state_tester state2{state1.get_params()};

  ASSERT(state2.get_int64("TestInt") == nvbench::int64_t{22});
  ASSERT(state2.get_float64("TestFloat") == nvbench::float64_t{3.14});
  ASSERT(state2.get_string("TestString") == "A String!");
}

int main() { test_params(); }
