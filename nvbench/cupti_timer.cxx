// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nvbench/cuda_call.cuh>
#include <nvbench/cupti_timer.cuh>
#include <nvbench/types.cuh>

#include <cupti.h>

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
                                                 CUPTI_ACTIVITY_KIND_MEMCPY2,
                                                 CUPTI_ACTIVITY_KIND_MEMSET,
                                                 CUPTI_ACTIVITY_KIND_RUNTIME,
                                                 CUPTI_ACTIVITY_KIND_DRIVER,
                                                 CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION};

struct activity_record
{
  nvbench::uint32_t correlation_id;
  nvbench::uint64_t start;
  nvbench::uint64_t end;
  CUpti_ActivityKind kind;
};

struct activity_store
{
  // Callbacks may report failure without taking the mutex or allocating memory.
  std::atomic<nvbench::uint64_t> num_collection_failures{0};
  std::mutex mutex;
  std::vector<activity_record> records;

  // correlation id -> external id
  std::unordered_map<nvbench::uint32_t, nvbench::uint64_t> external_ids;

  // Only accessed by the thread using cupti_timer.
  std::unordered_set<nvbench::uint64_t> pending_measurements;
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

// CUPTI drops records if the host callback cannot provide a activity buffer large enough, or
// if the device-side CDP buffer is full. This would make the measurement incomplete, so the
// collection generation is incremented to invalidate any pending measurements
void check_dropped_records(CUcontext context, nvbench::uint32_t stream_id) noexcept
{
  std::size_t dropped = 0;
  if (cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped) != CUPTI_SUCCESS ||
      dropped != 0)
  {
    ++(get_store().num_collection_failures);
  }
}

// callback called by CUPTI when a memory buffer is requested
void CUPTIAPI buffer_requested_callback(nvbench::uint8_t **buffer,
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
  if (!buffer_ptr)
  {
    *size = 0;
    ++(get_store().num_collection_failures);
  }
}

// callback called by CUPTI when a memory buffer is completed
void CUPTIAPI buffer_completed_callback(CUcontext context,
                                        nvbench::uint32_t stream_id,
                                        nvbench::uint8_t *buffer,
                                        std::size_t,
                                        std::size_t valid_size)
{
  auto &store = get_store();
  try
  {
    std::lock_guard lock{store.mutex};
    // The start, end, and correlationId fields have the same layout in every
    // version of these records, so the oldest struct versions are used.
    CUpti_Activity *record = nullptr;
    CUptiResult status     = CUPTI_ERROR_MAX_LIMIT_REACHED;
    while (valid_size != 0 &&
           (status = cuptiActivityGetNextRecord(buffer, valid_size, &record)) == CUPTI_SUCCESS)
    {
      switch (record->kind)
      {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
          const auto *rec = reinterpret_cast<const CUpti_ActivityKernel4 *>(record);
          store.records.push_back({rec->correlationId, rec->start, rec->end, record->kind});
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
          const auto *rec = reinterpret_cast<const CUpti_ActivityMemcpy *>(record);
          store.records.push_back({rec->correlationId, rec->start, rec->end, record->kind});
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY2: {
          const auto *rec = reinterpret_cast<const CUpti_ActivityMemcpyPtoP *>(record);
          store.records.push_back({rec->correlationId, rec->start, rec->end, record->kind});
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
          const auto *rec = reinterpret_cast<const CUpti_ActivityMemset *>(record);
          store.records.push_back({rec->correlationId, rec->start, rec->end, record->kind});
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
    // CUPTI_ERROR_MAX_LIMIT_REACHED means the buffer is fully consumed. Other values indicate an
    // error
    if (status != CUPTI_ERROR_MAX_LIMIT_REACHED)
    {
      ++(store.num_collection_failures);
    }
  }
  catch (...)
  { // Exceptions must not propagate into CUPTI, but the collection is now incomplete
    ++(store.num_collection_failures);
  }
  // check if there are dropped records before reaching the buffer (e.g. no enough space)
  check_dropped_records(context, stream_id);
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
      NVBENCH_CUPTI_CALL(
        cuptiActivityRegisterCallbacks(buffer_requested_callback, buffer_completed_callback));
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
  get_store().pending_measurements.erase(m_id);
  release_activities();
}

void cupti_timer::start(cudaStream_t)
{
  this->pop_external_id(); // Close previous external correlation ID, if any
  auto &store = get_store();
  store.pending_measurements.erase(m_id);
  m_duration.reset();
  // keep track of the number of collection failures before start().
  // If the number of collection failures is different when we call get_duration(), CUPTI dropped
  // records in this timeframe and the measurement is incomplete.
  m_num_collection_failures = store.num_collection_failures.load();
  m_id                      = 0;
  const auto id             = ++store.next_id;
  store.pending_measurements.insert(id);
  try
  {
    NVBENCH_CUPTI_CALL(cuptiActivityPushExternalCorrelationId(external_kind, id));
  }
  catch (...)
  {
    store.pending_measurements.erase(id);
    throw;
  }
  m_id     = id;
  m_pushed = true;
}

void cupti_timer::stop(cudaStream_t) { this->pop_external_id(); }

void cupti_timer::pop_external_id()
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
  // Check for dropped records even when no callbacks have been called (e.g. allocation failure)
  // (nullptr, 0) means CUPTI’s global queue
  check_dropped_records(nullptr, 0);

  using interval_t = std::pair<nvbench::uint64_t, nvbench::uint64_t>;
  std::vector<interval_t> intervals;
  {
    auto &store = get_store();
    std::lock_guard lock{store.mutex};
    std::size_t invalid_kernels = 0;
    std::size_t invalid_memcpys = 0;
    std::size_t invalid_memsets = 0;
    // for all records, find the ones that have the same external id as the current measurement
    for (const auto &rec : store.records)
    {
      const auto it = store.external_ids.find(rec.correlation_id);
      if (it != store.external_ids.end() && it->second == m_id)
      {
        // Zero timestamps are unavailable; reversed intervals may be incomplete after a forced
        // flush. Keep their correlation IDs so only the affected measurement is rejected.
        if (rec.start == 0 || rec.end < rec.start)
        {
          switch (rec.kind)
          {
            case CUPTI_ACTIVITY_KIND_KERNEL:
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
              ++invalid_kernels;
              break;
            case CUPTI_ACTIVITY_KIND_MEMCPY:
            case CUPTI_ACTIVITY_KIND_MEMCPY2:
              ++invalid_memcpys;
              break;
            case CUPTI_ACTIVITY_KIND_MEMSET:
              ++invalid_memsets;
              break;
            default:
              break;
          }
          continue;
        }
        intervals.emplace_back(rec.start, rec.end);
      }
    }
    const auto invalid_records = invalid_kernels + invalid_memcpys + invalid_memsets;
    if (invalid_records != 0)
    {
      // Preserve the evidence so subsequent reads cannot report a successful zero duration.
      throw incomplete_cupti_measurement(fmt::format("CUPTI timer measurement is incomplete: {} "
                                                     "activity record(s) have invalid timestamps "
                                                     "(kernels: {}, memcpys: {}, memsets: {}).",
                                                     invalid_records,
                                                     invalid_kernels,
                                                     invalid_memcpys,
                                                     invalid_memsets));
    }
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

  // Reject incomplete collection before caching the duration
  // A record is in one of the following states:
  // - Consumed: get_duration() successfully completed.
  // - Abandoned: the timer starts a new measurement, invalidating the previous ones.
  // - No correlation mapping: events outsides timer start/stop range
  {
    auto &store = get_store();
    std::lock_guard lock{store.mutex};
    store.pending_measurements.erase(m_id);
    if (store.num_collection_failures.load() != m_num_collection_failures)
    {
      throw incomplete_cupti_measurement("CUPTI timer measurement is incomplete: activity records "
                                         "were dropped or collection failed.");
    }
    m_duration = static_cast<nvbench::float64_t>(busy_ns) * 1e-9;
    // Completed work may still belong to another timer whose result has not been read.
    // Deletes records related to consumed/invalid measurements and records without a
    // correlation mapping.
    auto remove_it =
      std::remove_if(store.records.begin(), store.records.end(), [&store](const auto &rec) {
        // correlation id -> external id
        const auto it = store.external_ids.find(rec.correlation_id);
        return it == store.external_ids.end() || store.pending_measurements.count(it->second) == 0;
      });
    store.records.erase(remove_it, store.records.end());

    // delete pending measurements that have no correlation mapping
    for (auto it = store.external_ids.begin(); it != store.external_ids.end();)
    {
      if (store.pending_measurements.count(it->second) == 0)
      {
        it = store.external_ids.erase(it);
      }
      else
      {
        ++it;
      }
    }
  }
  return *m_duration;
}

} // namespace nvbench
