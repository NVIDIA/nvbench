#include <nvbench/detail/markdown_format.cuh>

#include <nvbench/benchmark_base.cuh>
#include <nvbench/benchmark_manager.cuh>
#include <nvbench/state.cuh>
#include <nvbench/summary.cuh>

#include <fmt/format.h>

#include <functional>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

struct table_builder
{
  void add_cell(std::size_t row, const std::string &header, std::string value)
  {
    auto iter = std::find_if(m_columns.begin(),
                             m_columns.end(),
                             [&header](const column &col) {
                               return col.header == header;
                             });

    auto &col = iter == m_columns.end()
                  ? m_columns.emplace_back(
                      column{header, std::vector<std::string>{}, header.size()})
                  : *iter;

    col.max_width = std::max(col.max_width, value.size());
    if (col.rows.size() <= row)
    {
      col.rows.resize(row + 1);
      col.rows[row] = std::move(value);
    }
  }

  std::string to_string()
  {
    fmt::memory_buffer buffer;
    this->fix_row_lengths();
    this->print_header(buffer);
    this->print_divider(buffer);
    this->print_rows(buffer);
    return fmt::to_string(buffer);
  }

private:
  struct column
  {
    std::string header;
    std::vector<std::string> rows;
    std::size_t max_width;
  };

  void fix_row_lengths()
  { // Ensure that each row is the same length:
    m_num_rows = std::transform_reduce(
      m_columns.cbegin(),
      m_columns.cend(),
      0ll,
      [](const auto &a, const auto &b) { return a > b ? a : b; },
      [](const column &col) { return col.rows.size(); });
    std::for_each(m_columns.begin(),
                  m_columns.end(),
                  [num_rows = m_num_rows](column &col) {
                    col.rows.resize(num_rows);
                  });
  }

  void print_header(fmt::memory_buffer &buffer)
  {
    fmt::format_to(buffer, "|");
    for (const column &col : m_columns)
    {
      fmt::format_to(buffer, " {:^{}} |", col.header, col.max_width);
    }
    fmt::format_to(buffer, "\n");
  }

  void print_divider(fmt::memory_buffer &buffer)
  {
    fmt::format_to(buffer, "|");
    for (const column &col : m_columns)
    { // fill=-, centered, empty string, width = max_width + 2
      fmt::format_to(buffer, "{:-^{}}|", "", col.max_width + 2);
    }
    fmt::format_to(buffer, "\n");
  }

  void print_rows(fmt::memory_buffer &buffer)
  {
    for (std::size_t row = 0; row < m_num_rows; ++row)
    {
      fmt::format_to(buffer, "|");
      for (const column &col : m_columns)
      { // fill=-, centered, empty string, width = max_width + 2
        fmt::format_to(buffer, " {:>{}} |", col.rows[row], col.max_width);
      }
      fmt::format_to(buffer, "\n");
    }
  }

  std::string m_row_format;
  std::string m_div;
  std::vector<column> m_columns;
  std::size_t m_num_rows;
};

} // namespace

namespace nvbench
{
namespace detail
{

void markdown_format::print()
{
  auto format_visitor = [](const auto &v) {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<T, nvbench::float64_t>)
    {
      return fmt::format("{:7.5g}", v);
    }
    else if constexpr (std::is_same_v<T, std::string>)
    {
      return v;
    }
    return fmt::format("{}", v);
  };

  auto format_duration = [](nvbench::float64_t seconds) {
    if (seconds >= 1.) // 1+ sec
    {
      return fmt::format("{:5.2f} s", seconds);
    }
    else if (seconds >= 1e-1) // 100+ ms.
    {
      return fmt::format("{:5.2f} ms", seconds * 1e3);
    }
    else if (seconds >= 1e-4) // 100+ us.
    {
      return fmt::format("{:5.2f} us", seconds * 1e6);
    }
    else
    {
      return fmt::format("{:5.2f} ns", seconds * 1e9);
    }
  };

  auto &mgr = nvbench::benchmark_manager::get();
  for (const auto &bench_ptr : mgr.get_benchmarks())
  {
    fmt::print("\n# {}\n\n", bench_ptr->get_name());

    std::size_t row = 0;
    table_builder table;

    for (const auto &inner_states : bench_ptr->get_states())
    {
      for (const nvbench::state &state : inner_states)
      {
        const auto &axis_values = state.get_axis_values();
        for (const auto &name : axis_values.get_names())
        {
          std::string value = std::visit(format_visitor,
                                         axis_values.get_value(name));
          table.add_cell(row, name, std::move(value));
        }

        for (const auto &summ : state.get_summaries())
        {
          const std::string &name = summ.has_value("short_name")
                                      ? summ.get_string("short_name")
                                      : summ.get_name();

          if (summ.has_value("hint") && summ.get_string("hint") == "duration")
          {
            table.add_cell(row,
                           name,
                           format_duration(summ.get_float64("value")));
          }
          else
          {
            table.add_cell(row,
                           name,
                           std::visit(format_visitor, summ.get_value("value")));
          }
        }
        row++;
      }
    }

    fmt::print(table.to_string());
  }
}

} // namespace detail
} // namespace nvbench
