#pragma once

#include <nvbench/float64_axis.cuh>
#include <nvbench/int64_axis.cuh>
#include <nvbench/string_axis.cuh>
#include <nvbench/type_axis.cuh>

#include <memory>
#include <stdexcept>
#include <vector>

namespace nvbench
{

template <typename Kernel, typename TypeAxes>
struct benchmark
{
  using kernel_type  = Kernel;
  using type_axes    = TypeAxes;
  using type_configs = nvbench::tl::cartesian_product<type_axes>;

  static constexpr std::size_t num_type_configs =
    nvbench::tl::size<type_configs>{};

  void set_name(std::string name) { m_name = std::move(name); }
  const std::string &get_name() const { return m_name; }

  // Convenience API for a single type_axis.
  benchmark &set_type_axes_name(std::string name)
  {
    return this->set_type_axes_names({std::move(name)});
  }

  benchmark &set_type_axes_names(std::vector<std::string> names);

  benchmark &add_float64_axis(std::string name,
                              std::vector<nvbench::float64_t> data)
  {
    auto axis = std::make_unique<nvbench::float64_axis>(std::move(name));
    axis->set_inputs(std::move(data));
    m_float64_axes.push_back(std::move(axis));
    return *this;
  }

  benchmark &add_int64_axis(
    std::string name,
    std::vector<nvbench::int64_t> data,
    nvbench::int64_axis_flags flags = nvbench::int64_axis_flags::none)
  {
    auto axis = std::make_unique<nvbench::int64_axis>(std::move(name), flags);
    axis->set_inputs(std::move(data));
    m_int64_axes.push_back(std::move(axis));
    return *this;
  }

  benchmark &add_int64_power_of_two_axis(std::string name,
                                         std::vector<nvbench::int64_t> data)
  {
    return this->add_int64_axis(std::move(name),
                                std::move(data),
                                nvbench::int64_axis_flags::power_of_two);
  }

  benchmark &add_string_axis(std::string name, std::vector<std::string> data)
  {
    auto axis = std::make_unique<nvbench::string_axis>(std::move(name));
    axis->set_inputs(std::move(data));
    m_string_axes.push_back(std::move(axis));
    return *this;
  }

  [[nodiscard]] const auto &get_float64_axes() const { return m_float64_axes; }
  [[nodiscard]] const auto &get_int64_axes() const { return m_int64_axes; }
  [[nodiscard]] const auto &get_string_axes() const { return m_string_axes; }
  [[nodiscard]] const auto &get_type_axes() const { return m_type_axes; }

private:
  std::string m_name;

  std::vector<std::unique_ptr<nvbench::float64_axis>> m_float64_axes;
  std::vector<std::unique_ptr<nvbench::int64_axis>> m_int64_axes;
  std::vector<std::unique_ptr<nvbench::string_axis>> m_string_axes;
  std::vector<std::unique_ptr<nvbench::type_axis>> m_type_axes;
};

template <typename Kernel, typename TypeAxes>
benchmark<Kernel, TypeAxes> &
benchmark<Kernel, TypeAxes>::set_type_axes_names(std::vector<std::string> names)
{
  if (names.size() != nvbench::tl::size<type_axes>::value)
  { // TODO Find a way to get a better error message w/o bringing fmt
    // into this header.
    throw std::runtime_error("set_type_axes_names(): len(names) != "
                             "len(type_axes)");
  }
  auto names_iter = names.begin(); // contents will be moved from
  nvbench::tl::foreach<type_axes>([&axes = m_type_axes, &names_iter](
                                    [[maybe_unused]] auto wrapped_type) {
    // Note:
    // The word "type" appears 6 times in the next line.
    // Every. Single. Token.
    // Take a moment to just appreciate this beautiful language:
    typedef typename decltype(wrapped_type)::type type_list;
    auto axis = std::make_unique<nvbench::type_axis>(std::move(*names_iter++));
    axis->set_inputs<type_list>();
    axes.push_back(std::move(axis));
  });
  return *this;
}

} // namespace nvbench
