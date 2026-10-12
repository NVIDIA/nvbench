// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nvbench/cuda_call.cuh>
#include <nvbench/cupti_timer.cuh>
#include <nvbench/types.cuh>

#include <cupti.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// CUPTI workflow:
//
// 1. How CUPTI identifies CUDA calls:
//
// CUPTI gives every CUDA runtime and driver call a new number, called CorrelationId.
// CUPTI uses an identifier, called External correlation ID, to understand when the measurement is
// happening, excluding all previous and future CUDA calls. CUPTI keeps track of the records
// {externalId, correlationId} between push() and pop() APIs
//
// 2. How CUPTI retrieves the records:
//
// CUPTI uses a callback to retrieve the records. In particular, it needs a function to get a memory
// buffer to operate on, and another one for when the buffer is completed.
// Important: CUPTI starts its own worker thread. For this reason, the activity_store (collection of
// records) must be protected from concurrent access with a mutex.

namespace
{

[[noreturn]] void throw_cupti_error(const char *filename,
                                    nvbench::int64_t line,
                                    const char *command,
                                    CUptiResult error)
{
  const char *error_string{};
  cuptiGetResultString(error, &error_string);
  throw std::runtime_error(fmt::format("{}:{}: CUPTI API call returned error: {}\nCommand: '{}'",
                                       filename,
                                       line,
                                       error_string ? error_string : "unknown",
                                       command));
}

/// Throws a std::runtime_error if `call` doesn't return `CUPTI_SUCCESS`.
#define NVBENCH_CUPTI_CALL(call)                                                                   \
  do                                                                                               \
  {                                                                                                \
    const CUptiResult nvbench_cupti_call_error = call;                                             \
    if (nvbench_cupti_call_error != CUPTI_SUCCESS)                                                 \
    {                                                                                              \
      throw_cupti_error(__FILE__, __LINE__, #call, nvbench_cupti_call_error);                      \
    }                                                                                              \
  } while (false)

constexpr auto external_kind = CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0;

constexpr CUpti_ActivityKind activity_kinds[] = {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                                                 CUPTI_ACTIVITY_KIND_MEMCPY,
                                                 CUPTI_ACTIVITY_KIND_MEMSET,
                                                 CUPTI_ACTIVITY_KIND_RUNTIME,
                                                 CUPTI_ACTIVITY_KIND_DRIVER,
                                                 CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION};

struct activity_record
{
  nvbench::uint32_t correlation_id;
  nvbench::uint64_t start;
  nvbench::uint64_t end;
};

struct activity_store
{
  std::mutex mutex;
  std::vector<activity_record> records;

  // correlation id -> external id
  std::unordered_map<nvbench::uint32_t, nvbench::uint64_t> external_ids;

  // Only accessed by the thread using cupti_timer.
  nvbench::int64_t users    = 0;
  bool callbacks_registered = false;
  nvbench::uint64_t next_id = 0;
};

// CUPTI can still be running when the process is terminating.
// The activity_store lives until the process ends.
activity_store &get_store()
{
  static auto *store = new activity_store{};
  return *store;
}

// callback called by CUPTI when a memory buffer is requested
void CUPTIAPI buffer_requested(nvbench::uint8_t **buffer,
                               std::size_t *size,
                               std::size_t *max_records)
{
  // Size derived from CUPTI samples code
  constexpr auto buffer_size_bytes  = nvbench::uint64_t{4} << 20; // 4 MB
  constexpr auto buffer_size_dwords = buffer_size_bytes / sizeof(nvbench::uint64_t);
  *size                             = buffer_size_bytes;
  *max_records                      = 0; // no limit
  // 8-byte alignment is required by CUPTI
  auto buffer_ptr = new (std::nothrow) nvbench::uint64_t[buffer_size_dwords];
  *buffer         = reinterpret_cast<nvbench::uint8_t *>(buffer_ptr);
}

// callback called by CUPTI when a memory buffer is completed
void CUPTIAPI buffer_completed(CUcontext,
                               nvbench::uint32_t,
                               nvbench::uint8_t *buffer,
                               std::size_t,
                               std::size_t valid_size)
{
  auto &store = get_store();
  try
  {
    std::lock_guard lock{store.mutex};
    // start == 0 means that CUPTI cannot time the operation
    // end < start means the record is not finished (CUPTI_ACTIVITY_FLAG_FLUSH_FORCED)
    auto add_record =
      [&store](nvbench::uint32_t correlation_id, nvbench::uint64_t start, nvbench::uint64_t end) {
        if (start != 0 && end >= start)
        {
          store.records.push_back({correlation_id, start, end});
        }
      };

    // The start, end, and correlationId fields have the same layout in every
    // version of these records, so the oldest struct versions are used.
    CUpti_Activity *record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) == CUPTI_SUCCESS)
    {
      switch (record->kind)
      {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
          const auto *rec = reinterpret_cast<const CUpti_ActivityKernel4 *>(record);
          add_record(rec->correlationId, rec->start, rec->end);
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
          const auto *rec = reinterpret_cast<const CUpti_ActivityMemcpy *>(record);
          add_record(rec->correlationId, rec->start, rec->end);
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
          const auto *rec = reinterpret_cast<const CUpti_ActivityMemset *>(record);
          add_record(rec->correlationId, rec->start, rec->end);
          break;
        }
        case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
          const auto *r = reinterpret_cast<const CUpti_ActivityExternalCorrelation *>(record);
          if (r->externalKind == external_kind)
          {
            store.external_ids[r->correlationId] = r->externalId;
          }
          break;
        }
        default:
          break;
      }
    }
  }
  catch (...)
  { // Exceptions must not propagate into CUPTI. Records are dropped.
  }
  delete[] reinterpret_cast<nvbench::uint64_t *>(buffer);
}

void disable_activities() noexcept
{
  for (const auto kind : activity_kinds)
  {
    (void)cuptiActivityDisable(kind);
  }
}

void acquire_activities()
{
  auto &store = get_store();
  if (store.users == 0)
  {
    // initialize callbacks for memory buffer management
    if (!store.callbacks_registered)
    {
      NVBENCH_CUPTI_CALL(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
      store.callbacks_registered = true;
    }
    try
    {
      for (const auto kind : activity_kinds)
      {
        NVBENCH_CUPTI_CALL(cuptiActivityEnable(kind));
      }
    }
    catch (...)
    {
      disable_activities();
      throw;
    }
  }
  ++store.users;
}

void release_activities() noexcept
{
  auto &store = get_store();
  if (--store.users == 0)
  {
    (void)cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    disable_activities();

    std::lock_guard records_lock{store.mutex};
    store.records.clear();
    store.external_ids.clear();
  }
}

} // namespace

namespace nvbench
{

cupti_timer::cupti_timer() { acquire_activities(); }

cupti_timer::~cupti_timer()
{
  if (m_pushed)
  {
    nvbench::uint64_t id{};
    (void)cuptiActivityPopExternalCorrelationId(external_kind, &id);
  }
  release_activities();
}

void cupti_timer::start(cudaStream_t stream)
{
  this->stop(stream); // Close previous external correlation ID, if any
  m_duration.reset();
  m_id = ++get_store().next_id;
  NVBENCH_CUPTI_CALL(cuptiActivityPushExternalCorrelationId(external_kind, m_id));
  m_pushed = true;
}

void cupti_timer::stop(cudaStream_t)
{
  if (m_pushed)
  {
    nvbench::uint64_t id{};
    m_pushed = false;
    NVBENCH_CUPTI_CALL(cuptiActivityPopExternalCorrelationId(external_kind, &id));
  }
}

nvbench::float64_t cupti_timer::get_duration() const
{
  if (m_duration.has_value()) // a measurement is available
  {
    return *m_duration;
  }

  NVBENCH_CUDA_CALL(cudaDeviceSynchronize());
  NVBENCH_CUPTI_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));

  using interval_t = std::pair<nvbench::uint64_t, nvbench::uint64_t>;
  std::vector<interval_t> intervals;
  {
    auto &store = get_store();
    std::lock_guard lock{store.mutex};
    // for all records, find the ones that have the same external id as the current measurement
    for (const auto &rec : store.records)
    {
      const auto it = store.external_ids.find(rec.correlation_id);
      if (it != store.external_ids.end() && it->second == m_id)
      {
        intervals.emplace_back(rec.start, rec.end);
      }
    }
    // Everything issued until now has completed; older records are no longer needed.
    store.records.clear();
    store.external_ids.clear();
  }

  std::sort(intervals.begin(), intervals.end());
  nvbench::uint64_t busy_ns = 0;
  // For all intervals, calculate the total GPU busy time.
  // Note: intervals, which are sorted, can be overlapping
  if (!intervals.empty())
  {
    auto [current_start, current_end] = intervals.front();
    for (const auto &[start, end] : intervals)
    {
      if (start > current_end) // new interval
      {
        busy_ns += current_end - current_start;
        current_start = start;
      }
      current_end = std::max(current_end, end);
    }
    busy_ns += current_end - current_start;
  }

  m_duration = static_cast<nvbench::float64_t>(busy_ns) * 1e-9;
  return *m_duration;
}

} // namespace nvbench
