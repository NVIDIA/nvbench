#pragma once

#include <nvbench/detail/transform_reduce.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace nvbench::internal
{

/*!
 * Implementation detail for markdown_format.
 *
 * Usage:
 *
 * ```
 * markdown_table table;
 * table.add_cell(...); // Insert all cells
 * std::string table_str = table.to_string();
 * ```
 */
struct markdown_table
{
  void add_cell(std::size_t row,
                const std::string &column_key,
                const std::string &header,
                std::string value)
  {
    auto iter = std::find_if(m_columns.begin(),
                             m_columns.end(),
                             [&column_key](const column &col) {
                               return col.key == column_key;
                             });

    auto &col = iter == m_columns.end()
                  ? m_columns.emplace_back(column{column_key,
                                                  header,
                                                  std::vector<std::string>{},
                                                  header.size()})
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
    if (m_columns.empty())
    {
      return {};
    }

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
    std::string key;
    std::string header;
    std::vector<std::string> rows;
    std::size_t max_width;
  };

  void fix_row_lengths()
  { // Ensure that each row is the same length:
    m_num_rows = nvbench::detail::transform_reduce(
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

  std::vector<column> m_columns;
  std::size_t m_num_rows{};
};

} // namespace nvbench::internal
