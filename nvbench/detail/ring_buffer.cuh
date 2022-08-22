/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <nvbench/config.cuh>

#include <nvbench/detail/statistics.cuh>

#include <cassert>
#include <vector>

namespace nvbench::detail
{

/**
 * @brief A simple, dynamically sized ring buffer.
 */
template <typename T>
struct ring_buffer
{
  /**
   * Create a new ring buffer with the requested capacity.
   */
  explicit ring_buffer(std::size_t capacity)
      : m_buffer(capacity)
  {}

  /**
   * Iterators provide all values in the ring buffer in unspecified order.
   * @{
   */
  // clang-format off
  [[nodiscard]] auto begin()        { return m_buffer.begin(); }
  [[nodiscard]] auto begin() const  { return m_buffer.begin(); }
  [[nodiscard]] auto cbegin() const { return m_buffer.cbegin(); }
  [[nodiscard]] auto end()        { return m_buffer.begin()  + this->size(); }
  [[nodiscard]] auto end() const  { return m_buffer.begin()  + this->size(); }
  [[nodiscard]] auto cend() const { return m_buffer.cbegin() + this->size(); }
  // clang-format on
  /** @} */

  /**
   * The number of valid values in the ring buffer. Always <= capacity().
   */
  [[nodiscard]] std::size_t size() const { return m_full ? m_buffer.size() : m_index; }

  /**
   * The maximum size of the ring buffer.
   */
  [[nodiscard]] std::size_t capacity() const { return m_buffer.size(); }

  /**
   * @return True if the ring buffer is empty.
   */
  [[nodiscard]] bool empty() const { return m_index == 0 && !m_full; }

  /**
   * Remove all values from the buffer without modifying capacity.
   */
  void clear()
  {
    m_index = 0;
    m_full  = false;
  }

  /**
   * Add a new value to the ring buffer. If size() == capacity(), the oldest
   * element in the buffer is overwritten.
   */
  void push_back(T val)
  {
    assert(m_index < m_buffer.size());

    m_buffer[m_index] = val;

    m_index = (m_index + 1) % m_buffer.size();
    if (m_index == 0)
    { // buffer wrapped
      m_full = true;
    }
  }

  /**
   * Get the most recently added value.
   * @{
   */
  [[nodiscard]] auto back() const
  {
    assert(!this->empty());
    const auto back_index = m_index == 0 ? m_buffer.size() - 1 : m_index - 1;
    return m_buffer[back_index];
  }
  [[nodiscard]] auto back()
  {
    assert(!this->empty());
    const auto back_index = m_index == 0 ? m_buffer.size() - 1 : m_index - 1;
    return m_buffer[back_index];
  }
  /**@}*/

private:
  std::vector<T> m_buffer;
  std::size_t m_index{0};
  bool m_full{false};
};

} // namespace nvbench::detail
