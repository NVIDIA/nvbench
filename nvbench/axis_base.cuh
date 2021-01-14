#pragma once

#include <memory>
#include <string>
#include <utility>

namespace nvbench
{

enum class axis_type
{
  type,
  int64,
  float64,
  string
};

struct axis_base
{
  virtual ~axis_base();

  [[nodiscard]] std::unique_ptr<axis_base> clone() const;

  [[nodiscard]] const std::string &get_name() const { return m_name; }

  [[nodiscard]] axis_type get_type() const { return m_type; }

  [[nodiscard]] std::size_t get_size() const { return this->do_get_size(); }

  [[nodiscard]] std::string get_input_string(std::size_t i) const
  {
    return this->do_get_input_string(i);
  }

  [[nodiscard]] std::string get_description(std::size_t i) const
  {
    return this->do_get_description(i);
  }

protected:
  axis_base(std::string name, axis_type type)
      : m_name{std::move(name)}
      , m_type{type}
  {}

private:
  virtual std::unique_ptr<axis_base> do_clone() const          = 0;
  virtual std::size_t do_get_size() const                      = 0;
  virtual std::string do_get_input_string(std::size_t i) const = 0;
  virtual std::string do_get_description(std::size_t i) const  = 0;

  std::string m_name;
  axis_type m_type;
};

} // namespace nvbench
