#pragma once

#include <nvbench/axis_base.cuh>

#include <nvbench/type_list.cuh>
#include <nvbench/type_strings.cuh>

#include <string>
#include <vector>

namespace nvbench
{

struct type_axis final : public axis_base
{
  explicit type_axis(std::string name)
      : axis_base{std::move(name), axis_type::type}
      , m_input_strings{}
      , m_descriptions{}
  {}

  ~type_axis() final;

  template <typename TypeList>
  void set_inputs();

private:
  std::size_t do_get_size() const final { return m_input_strings.size(); }
  std::string do_get_input_string(std::size_t i) const final
  {
    return m_input_strings[i];
  }
  std::string do_get_description(std::size_t i) const final
  {
    return m_descriptions[i];
  }

  std::vector<std::string> m_input_strings;
  std::vector<std::string> m_descriptions;
};

template <typename TypeList>
void type_axis::set_inputs()
{
  // Need locals for lambda capture...
  auto &input_strings = m_input_strings;
  auto &descriptions  = m_descriptions;
  nvbench::tl::foreach<TypeList>(
    [&input_strings, &descriptions]([[maybe_unused]] auto wrapped_type) {
      using T       = typename decltype(wrapped_type)::type;
      using Strings = nvbench::type_strings<T>;
      input_strings.push_back(Strings::input_string());
      descriptions.push_back(Strings::description());
    });
}

} // namespace nvbench
