// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <nvbench/config.cuh>

#if defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_GCC)
#pragma GCC system_header
#elif defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_CLANG)
#pragma clang system_header
#elif defined(NVBENCH_IMPLICIT_SYSTEM_HEADER_MSVC)
#pragma system_header
#endif

namespace nvbench::detail
{

template <typename Measure>
struct stream_cleanup_guard
{
  explicit stream_cleanup_guard(Measure &measure)
      : m_measure{measure}
  {
    m_sync_armed = true;
  }

  stream_cleanup_guard(const stream_cleanup_guard &)            = delete;
  stream_cleanup_guard(stream_cleanup_guard &&)                 = delete;
  stream_cleanup_guard &operator=(const stream_cleanup_guard &) = delete;
  stream_cleanup_guard &operator=(stream_cleanup_guard &&)      = delete;

  ~stream_cleanup_guard() noexcept
  {
    if (m_unblock_armed)
    {
      m_measure.unblock_stream_noexcept();
    }
    if (m_sync_armed)
    {
      (void)m_measure.sync_stream_noexcept();
    }
  }

  void block_stream()
  {
    // Arm cleanup before queueing the blocking kernel. If block_stream throws
    // after queueing work, the destructor must still unblock the stream.
    m_unblock_armed = true;
    m_measure.block_stream();
  }

  void unblock()
  {
    if (m_unblock_armed)
    {
      m_measure.unblock_stream();
      m_unblock_armed = false;
    }
  }

  void release() noexcept
  {
    m_unblock_armed = false;
    m_sync_armed    = false;
  }

private:
  Measure &m_measure;
  bool m_unblock_armed{false};
  bool m_sync_armed{false};
};

} // namespace nvbench::detail
