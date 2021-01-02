#include <nvbench/state.cuh>

#include <nvbench/summary.cuh>
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
  state_tester state;
  state.set_param("TestInt", nvbench::int64_t{22});
  state.set_param("TestFloat", nvbench::float64_t{3.14});
  state.set_param("TestString", "A String!");

  ASSERT(state.get_int64("TestInt") == nvbench::int64_t{22});
  ASSERT(state.get_float64("TestFloat") == nvbench::float64_t{3.14});
  ASSERT(state.get_string("TestString") == "A String!");
}

void test_summaries()
{
  state_tester state;
  ASSERT(state.get_summaries().size() == 0);

  {
    nvbench::summary& summary = state.add_summary("Test Summary1");
    summary.set_float64("Float", 3.14);
    summary.set_int64("Int", 128);
    summary.set_string("String", "str");
  }

  ASSERT(state.get_summaries().size() == 1);
  ASSERT(state.get_summary("Test Summary1").get_size() == 3);
  ASSERT(state.get_summary("Test Summary1").get_float64("Float") == 3.14);
  ASSERT(state.get_summary("Test Summary1").get_int64("Int") == 128);
  ASSERT(state.get_summary("Test Summary1").get_string("String") == "str");

  {
    nvbench::summary summary{"Test Summary2"};
    state.add_summary(std::move(summary));
  }

  ASSERT(state.get_summaries().size() == 2);
  ASSERT(state.get_summary("Test Summary1").get_size() == 3);
  ASSERT(state.get_summary("Test Summary1").get_float64("Float") == 3.14);
  ASSERT(state.get_summary("Test Summary1").get_int64("Int") == 128);
  ASSERT(state.get_summary("Test Summary1").get_string("String") == "str");
  ASSERT(state.get_summary("Test Summary2").get_size() == 0);
}

int main()
{
  test_params();
  test_summaries();
}
