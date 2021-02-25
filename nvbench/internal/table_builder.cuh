#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace nvbench::internal
{

/*!
 * State for a text table (rows and columns of cells). Tracks column widths.
 */
struct table_builder
{
  struct column
  {
    std::string key;
    std::string header;
    std::vector<std::string> rows;
    std::size_t max_width;
  };

  std::vector<column> m_columns;
  std::size_t m_num_rows{};

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
};

} // namespace nvbench::internal
