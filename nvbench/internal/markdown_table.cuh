#pragma once

#include <nvbench/detail/transform_reduce.cuh>

#include <nvbench/internal/table_builder.cuh>

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace nvbench::internal
{

/*!
 * Implementation detail for markdown_printer.
 *
 * Usage:
 *
 * ```
 * markdown_table table;
 * table.add_cell(...); // Insert all cells
 * std::string table_str = table.to_string();
 * ```
 */
struct markdown_table : private table_builder
{
  using table_builder::add_cell;

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
};

} // namespace nvbench::internal
