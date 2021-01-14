#include <nvbench/axes_metadata.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>

namespace nvbench
{

axes_metadata::axes_metadata(const axes_metadata &other)
{
  m_axes.reserve(other.get_axes().size());
  for (const auto &axis : other.get_axes())
  {
    m_axes.push_back(axis->clone());
  }
}

axes_metadata &axes_metadata::operator=(const axes_metadata &other)
{
  m_axes.clear();
  m_axes.reserve(other.get_axes().size());
  for (const auto &axis : other.get_axes())
  {
    m_axes.push_back(axis->clone());
  }
  return *this;
}

void axes_metadata::add_float64_axis(std::string name,
                                     std::vector<nvbench::float64_t> data)
{
  auto axis = std::make_unique<nvbench::float64_axis>(std::move(name));
  axis->set_inputs(std::move(data));
  m_axes.push_back(std::move(axis));
}

void axes_metadata::add_int64_axis(std::string name,
                                   std::vector<nvbench::int64_t> data,
                                   nvbench::int64_axis_flags flags)
{
  auto axis = std::make_unique<nvbench::int64_axis>(std::move(name), flags);
  axis->set_inputs(std::move(data));
  m_axes.push_back(std::move(axis));
}

void axes_metadata::add_string_axis(std::string name,
                                    std::vector<std::string> data)
{
  auto axis = std::make_unique<nvbench::string_axis>(std::move(name));
  axis->set_inputs(std::move(data));
  m_axes.push_back(std::move(axis));
}

const int64_axis &axes_metadata::get_int64_axis(std::string_view name) const
{
  const auto &axis = this->get_axis(name, nvbench::axis_type::int64);
  return static_cast<const nvbench::int64_axis &>(axis);
}

const float64_axis &axes_metadata::get_float64_axis(std::string_view name) const
{
  const auto &axis = this->get_axis(name, nvbench::axis_type::float64);
  return static_cast<const nvbench::float64_axis &>(axis);
}

const string_axis &axes_metadata::get_string_axis(std::string_view name) const
{
  const auto &axis = this->get_axis(name, nvbench::axis_type::string);
  return static_cast<const nvbench::string_axis &>(axis);
}

const type_axis &axes_metadata::get_type_axis(std::string_view name) const
{
  const auto &axis = this->get_axis(name, nvbench::axis_type::type);
  return static_cast<const nvbench::type_axis &>(axis);
}

const axis_base &axes_metadata::get_axis(std::string_view name) const
{
  auto iter =
    std::find_if(m_axes.cbegin(), m_axes.cend(), [&name](const auto &axis) {
      return axis->get_name() == name;
    });

  if (iter == m_axes.cend())
  {
    throw std::runtime_error(
      fmt::format("{}:{}: Axis '{}' not found.", __FILE__, __LINE__, name));
  }

  return **iter;
}

const axis_base &axes_metadata::get_axis(std::string_view name,
                                         nvbench::axis_type type) const
{
  const auto &axis = this->get_axis(name);
  if (axis.get_type() != type)
  {
    throw std::runtime_error(fmt::format("{}:{}: Axis '{}' type mismatch "
                                         "(expected {}, actual {}).",
                                         __FILE__,
                                         __LINE__,
                                         name,
                                         type,
                                         axis.get_type()));
  }
  return axis;
}

const nvbench::type_axis &axes_metadata::get_type_axis(std::size_t index) const
{
  for (const auto &axis : m_axes)
  {
    if (axis->get_type() == nvbench::axis_type::type)
    {
      const type_axis &t_axis = static_cast<const type_axis &>(*axis);
      if (t_axis.get_axis_index() == index)
      {
        return t_axis;
      }
    }
  }
  throw std::runtime_error(fmt::format("{}:{}: Invalid type axis index: {}.",
                                       __FILE__,
                                       __LINE__,
                                       index));
}

} // namespace nvbench
