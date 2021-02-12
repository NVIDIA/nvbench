#pragma once

#include <nvbench/types.cuh>

#include <string>
#include <variant>
#include <vector>

namespace nvbench
{

/**
 * Maintains a map of key / value pairs where the keys are names and the
 * values may be int64s, float64s, or strings.
 */
struct named_values
{
  using value_type =
    std::variant<nvbench::int64_t, nvbench::float64_t, std::string>;

  enum class type
  {
    int64,
    float64,
    string
  };

  void append(const named_values& other);

  [[nodiscard]] std::size_t get_size() const;
  [[nodiscard]] std::vector<std::string> get_names() const;

  void set_value(std::string name, value_type value);

  void set_int64(std::string name, nvbench::int64_t value);
  void set_float64(std::string name, nvbench::float64_t value);
  void set_string(std::string name, std::string value);

  [[nodiscard]] nvbench::int64_t get_int64(const std::string &name) const;
  [[nodiscard]] nvbench::float64_t get_float64(const std::string &name) const;
  [[nodiscard]] const std::string &get_string(const std::string &name) const;

  [[nodiscard]] type get_type(const std::string &name) const;
  [[nodiscard]] bool has_value(const std::string &name) const;
  [[nodiscard]] const value_type& get_value(const std::string &name) const;

  void clear();

  void remove_value(const std::string& name);

private:
  struct named_value
  {
    std::string name;
    value_type value;
  };
  // TODO this should just be an unordered_map<string, value_type>.
  using storage_type = std::vector<named_value>;

  storage_type m_storage;
};

} // namespace nvbench
