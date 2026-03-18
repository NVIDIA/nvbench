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

#include <nvbench/cupti_profiler.cuh>
#include <nvbench/detail/throw.cuh>
#include <nvbench/device_info.cuh>

#include <cupti.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_range_profiler.h>
#include <cupti_target.h>

#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace nvbench::detail
{

namespace
{

constexpr const char *kRangeProfilerNullMsg = "CUPTI Range profiler object is null";
constexpr const char *kRangeName            = "nvbench";

void cupti_call_impl(const CUptiResult status, const char *prefix)
{
  if (status != CUPTI_SUCCESS)
  {
    const char *errstr{};
    cuptiGetResultString(status, &errstr);
    const std::string msg = errstr ? errstr : "unknown";
    NVBENCH_THROW(std::runtime_error, "{}{}", prefix, msg);
  }
}

void cupti_call(const CUptiResult status)
{
  cupti_call_impl(status, "CUPTI call returned error: ");
}

void cupti_host_call(const CUptiResult status)
{
  cupti_call_impl(status, "CUPTI Host API call returned error: ");
}

} // namespace

struct cupti_profiler::host_impl
{
  CUpti_Profiler_Host_Object *object{};
  bool initialized{};
  const char *supported_chip_name{};
  std::vector<const char *> metric_names;
  std::vector<std::uint8_t> config_image;
  CUpti_RangeProfiler_Object *range_profiler_object{};
};

class cupti_profiler::profiler_init_guard
{
public:
  profiler_init_guard()
      : m_active(true)
  {
    CUpti_Profiler_Initialize_Params profiler_params{};
    profiler_params.structSize = CUpti_Profiler_Initialize_Params_STRUCT_SIZE;
    cupti_call(cuptiProfilerInitialize(&profiler_params));
  }

  ~profiler_init_guard()
  {
    // Best-effort cleanup; cannot throw from destructor.
    (void)deinitialize();
  }

  CUptiResult deinitialize()
  {
    if (!m_active)
    {
      return CUPTI_SUCCESS;
    }
    m_active = false;
    CUpti_Profiler_DeInitialize_Params profiler_params{};
    profiler_params.structSize = CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE;
    return cuptiProfilerDeInitialize(&profiler_params);
  }

private:
  bool m_active;
};

cupti_profiler::cupti_profiler(nvbench::device_info device, std::vector<std::string> &&metric_names)
    : m_metric_names(metric_names)
    , m_device(device)
    , m_host(std::make_unique<host_impl>())
{
  initialize_profiler();
  initialize_chip_name();
  initialize_availability_image();
  initialize_config_image();
  initialize_counter_data_image();

  m_available = true;
}

cupti_profiler::cupti_profiler(cupti_profiler &&rhs) noexcept
    : m_device(rhs.m_device.get_id(), rhs.m_device.get_cuda_device_prop())
{
  (*this) = std::move(rhs);
}

cupti_profiler &cupti_profiler::operator=(cupti_profiler &&rhs) noexcept
{
  if (this == &rhs)
  {
    return *this;
  }

  disable_range_profiler();
  deinitialize_profiler_host();

  m_device               = rhs.m_device;
  m_available            = rhs.m_available;
  m_chip_name            = std::move(rhs.m_chip_name);
  m_metric_names         = std::move(rhs.m_metric_names);
  m_config_image         = std::move(rhs.m_config_image);
  m_data_image           = std::move(rhs.m_data_image);
  m_availability_image   = std::move(rhs.m_availability_image);
  m_all_passes_submitted = rhs.m_all_passes_submitted;
  m_profiler_guard       = std::move(rhs.m_profiler_guard);

  m_host = std::move(rhs.m_host);

  rhs.m_available = false;
  rhs.m_host.reset();
  rhs.m_all_passes_submitted = false;
  rhs.m_profiler_guard.reset();

  return *this;
}

void cupti_profiler::ensure_host()
{
  if (!m_host)
  {
    m_host = std::make_unique<host_impl>();
  }
}

void cupti_profiler::initialize_profiler()
{
  if (!m_device.is_cupti_supported())
  {
    NVBENCH_THROW(std::runtime_error, "{}", "Device isn't supported for CUPTI");
  }

  if (!m_profiler_guard)
  {
    m_profiler_guard = std::make_unique<profiler_init_guard>();
  }
}

void cupti_profiler::initialize_chip_name()
{
  CUpti_Device_GetChipName_Params params{};
  params.structSize  = CUpti_Device_GetChipName_Params_STRUCT_SIZE;
  params.deviceIndex = static_cast<size_t>(m_device.get_id());
  cupti_call(cuptiDeviceGetChipName(&params));

  if (params.pChipName == nullptr)
  {
    NVBENCH_THROW(std::runtime_error,
                  "CUPTI returned null chip name for device[{}]: {}",
                  m_device.get_id(),
                  m_device.get_name());
  }

  m_chip_name = params.pChipName;
}

void cupti_profiler::initialize_availability_image()
{
  CUpti_Profiler_GetCounterAvailability_Params params{};

  params.structSize                = CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE;
  params.ctx                       = m_device.get_context();
  params.bAllowDeviceLevelCounters = true;

  cupti_call(cuptiProfilerGetCounterAvailability(&params));

  m_availability_image.clear();
  m_availability_image.resize(params.counterAvailabilityImageSize);
  params.pCounterAvailabilityImage = m_availability_image.data();

  cupti_call(cuptiProfilerGetCounterAvailability(&params));
}

void cupti_profiler::initialize_profiler_host()
{
  if (m_host && m_host->initialized)
  {
    return;
  }

  ensure_host();
  m_host->supported_chip_name = nullptr;

  CUpti_Profiler_Host_GetSupportedChips_Params chips_params{};
  chips_params.structSize  = CUpti_Profiler_Host_GetSupportedChips_Params_STRUCT_SIZE;
  chips_params.pPriv       = nullptr;
  CUptiResult chips_status = cuptiProfilerHostGetSupportedChips(&chips_params);
  if (chips_status == CUPTI_SUCCESS && chips_params.numChips > 0)
  {
    for (size_t idx = 0; idx < chips_params.numChips; ++idx)
    {
      if (chips_params.ppChipNames[idx] != nullptr && m_chip_name == chips_params.ppChipNames[idx])
      {
        m_host->supported_chip_name = chips_params.ppChipNames[idx];
        break;
      }
    }

    if (m_host->supported_chip_name == nullptr)
    {
      m_host->supported_chip_name = chips_params.ppChipNames[0];
    }
  }

  if (m_host->supported_chip_name == nullptr && m_chip_name.empty())
  {
    NVBENCH_THROW(std::runtime_error, "{}", "CUPTI Host returned null supported chip name");
  }

  CUpti_Profiler_Host_Initialize_Params params{};
  params.structSize                = CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE;
  params.pPriv                     = nullptr;
  params.profilerType              = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
  params.pCounterAvailabilityImage = nullptr;

  params.pChipName        = m_host->supported_chip_name != nullptr ? m_host->supported_chip_name
                                                                   : m_chip_name.c_str();
  CUptiResult host_status = cuptiProfilerHostInitialize(&params);
  cupti_host_call(host_status);

  m_host->object      = params.pHostObject;
  m_host->initialized = true;
}

namespace
{
std::vector<const char *> collect_metric_ptrs(const std::vector<std::string> &names)
{
  std::vector<const char *> ptrs;
  ptrs.reserve(names.size());
  for (const auto &name : names)
  {
    ptrs.push_back(name.c_str());
  }
  return ptrs;
}

void append_base_metrics(CUpti_Profiler_Host_Object *host_object,
                         std::unordered_set<std::string> &supported_base_metrics,
                         CUpti_MetricType metric_type)
{
  CUpti_Profiler_Host_GetBaseMetrics_Params base_params{};
  base_params.structSize  = CUpti_Profiler_Host_GetBaseMetrics_Params_STRUCT_SIZE;
  base_params.pPriv       = nullptr;
  base_params.pHostObject = host_object;
  base_params.metricType  = metric_type;
  CUptiResult base_status = cuptiProfilerHostGetBaseMetrics(&base_params);
  if (base_status == CUPTI_SUCCESS && base_params.ppMetricNames != nullptr)
  {
    for (size_t idx = 0; idx < base_params.numMetrics; ++idx)
    {
      if (base_params.ppMetricNames[idx] != nullptr)
      {
        supported_base_metrics.emplace(base_params.ppMetricNames[idx]);
      }
    }
  }
}

void append_metric_names(CUpti_Profiler_Host_Object *host_object,
                         const std::vector<std::string> &requested_metrics,
                         const std::unordered_set<std::string> &supported_base_metrics,
                         std::vector<const char *> &host_metric_names,
                         std::vector<std::string> &config_metric_names)
{
  for (const auto &metric : requested_metrics)
  {
    if (metric.empty())
    {
      NVBENCH_THROW(std::runtime_error, "{}", "CUPTI Host metric name is empty");
    }

    const auto dot_pos      = metric.find('.');
    std::string base_metric = dot_pos != std::string::npos ? metric.substr(0, dot_pos) : metric;

    if (!supported_base_metrics.empty() &&
        supported_base_metrics.find(base_metric) == supported_base_metrics.end())
    {
      continue;
    }

    CUpti_Profiler_Host_GetMetricProperties_Params props_params{};
    props_params.structSize  = CUpti_Profiler_Host_GetMetricProperties_Params_STRUCT_SIZE;
    props_params.pPriv       = nullptr;
    props_params.pHostObject = host_object;
    props_params.pMetricName = base_metric.c_str();
    CUptiResult props_status = cuptiProfilerHostGetMetricProperties(&props_params);
    if (props_status != CUPTI_SUCCESS)
    {
      continue;
    }

    if (props_params.metricCollectionScope != CUPTI_METRIC_COLLECTION_SCOPE_CONTEXT)
    {
      continue;
    }

    host_metric_names.push_back(metric.c_str());
    config_metric_names.push_back(metric);
  }
}

std::vector<const char *> add_metrics_or_throw(const std::vector<std::string> &config_metric_names,
                                               CUpti_Profiler_Host_Object *host_object)
{
  std::vector<const char *> config_metric_ptrs = collect_metric_ptrs(config_metric_names);

  CUpti_Profiler_Host_ConfigAddMetrics_Params add_params{};
  add_params.structSize    = CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE;
  add_params.pPriv         = nullptr;
  add_params.pHostObject   = host_object;
  add_params.ppMetricNames = config_metric_ptrs.empty()
                               ? nullptr
                               : const_cast<const char **>(config_metric_ptrs.data());
  add_params.numMetrics    = config_metric_ptrs.size();
  cupti_host_call(cuptiProfilerHostConfigAddMetrics(&add_params));

  return config_metric_ptrs;
}
} // namespace

void cupti_profiler::deinitialize_profiler_host()
{
  if (!m_host || !m_host->initialized)
  {
    return;
  }

  m_host->metric_names.clear();
  m_host->config_image.clear();
  m_host->range_profiler_object = nullptr;

  CUpti_Profiler_Host_Deinitialize_Params params{};
  params.structSize  = CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE;
  params.pPriv       = nullptr;
  params.pHostObject = m_host->object;
  cupti_host_call(cuptiProfilerHostDeinitialize(&params));

  m_host->object      = nullptr;
  m_host->initialized = false;
}

void cupti_profiler::enable_range_profiler()
{
  if (!m_host)
  {
    m_host = std::make_unique<host_impl>();
  }

  if (m_host->range_profiler_object != nullptr)
  {
    return;
  }

  CUpti_RangeProfiler_Enable_Params params{};
  params.structSize = CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE;
  params.ctx        = m_device.get_context();
  cupti_call(cuptiRangeProfilerEnable(&params));
  m_host->range_profiler_object = params.pRangeProfilerObject;
}

void cupti_profiler::disable_range_profiler()
{
  if (!m_host || m_host->range_profiler_object == nullptr)
  {
    return;
  }

  CUpti_RangeProfiler_Disable_Params params{};
  params.structSize           = CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE;
  params.pRangeProfilerObject = m_host->range_profiler_object;
  cupti_call(cuptiRangeProfilerDisable(&params));
  m_host->range_profiler_object = nullptr;
}

void cupti_profiler::set_range_profiler_config()
{
  if (!m_host || m_host->range_profiler_object == nullptr)
  {
    NVBENCH_THROW(std::runtime_error, "{}", kRangeProfilerNullMsg);
  }

  CUpti_RangeProfiler_SetConfig_Params params{};
  params.structSize           = CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE;
  params.pRangeProfilerObject = m_host->range_profiler_object;
  params.pConfig              = m_config_image.data();
  params.configSize           = m_config_image.size();
  params.pCounterDataImage    = m_data_image.data();
  params.counterDataImageSize = m_data_image.size();
  params.maxRangesPerPass     = m_num_ranges;
  params.numNestingLevels     = 1;
  params.minNestingLevel      = 1;
  params.passIndex            = 0;
  params.targetNestingLevel   = 1;
  params.range                = CUPTI_UserRange;
  params.replayMode           = CUPTI_UserReplay;
  cupti_call(cuptiRangeProfilerSetConfig(&params));
}

void cupti_profiler::start_range_profiler()
{
  if (!m_host || m_host->range_profiler_object == nullptr)
  {
    NVBENCH_THROW(std::runtime_error, "{}", kRangeProfilerNullMsg);
  }

  CUpti_RangeProfiler_Start_Params params{};
  params.structSize           = CUpti_RangeProfiler_Start_Params_STRUCT_SIZE;
  params.pRangeProfilerObject = m_host->range_profiler_object;
  cupti_call(cuptiRangeProfilerStart(&params));
}

void cupti_profiler::stop_range_profiler()
{
  if (!m_host || m_host->range_profiler_object == nullptr)
  {
    NVBENCH_THROW(std::runtime_error, "{}", kRangeProfilerNullMsg);
  }

  CUpti_RangeProfiler_Stop_Params params{};
  params.structSize           = CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE;
  params.pRangeProfilerObject = m_host->range_profiler_object;
  cupti_call(cuptiRangeProfilerStop(&params));
  m_all_passes_submitted = params.isAllPassSubmitted;
}

void cupti_profiler::decode_counter_data()
{
  if (!m_host || m_host->range_profiler_object == nullptr)
  {
    NVBENCH_THROW(std::runtime_error, "{}", kRangeProfilerNullMsg);
  }

  CUpti_RangeProfiler_DecodeData_Params params{};
  params.structSize           = CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE;
  params.pRangeProfilerObject = m_host->range_profiler_object;
  cupti_call(cuptiRangeProfilerDecodeData(&params));
}

void cupti_profiler::initialize_config_image()
{
  initialize_config_image_host();
  m_config_image = m_host->config_image;
}

void cupti_profiler::initialize_config_image_host()
{
  initialize_profiler_host();

  if (m_metric_names.empty())
  {
    NVBENCH_THROW(std::runtime_error, "{}", "CUPTI Host config requires metrics, none provided");
  }

  m_host->metric_names.clear();
  m_host->metric_names.reserve(m_metric_names.size());
  std::vector<std::string> config_metric_names;
  config_metric_names.reserve(m_metric_names.size());
  std::unordered_set<std::string> supported_base_metrics;

  append_base_metrics(m_host->object, supported_base_metrics, CUPTI_METRIC_TYPE_COUNTER);
  append_base_metrics(m_host->object, supported_base_metrics, CUPTI_METRIC_TYPE_RATIO);
  append_base_metrics(m_host->object, supported_base_metrics, CUPTI_METRIC_TYPE_THROUGHPUT);
  append_metric_names(m_host->object,
                      m_metric_names,
                      supported_base_metrics,
                      m_host->metric_names,
                      config_metric_names);

  if (m_host->metric_names.empty())
  {
    NVBENCH_THROW(std::runtime_error, "{}", "CUPTI Host found no supported metrics");
  }
  std::vector<const char *> config_metric_ptrs = add_metrics_or_throw(config_metric_names,
                                                                      m_host->object);

  CUpti_Profiler_Host_GetConfigImageSize_Params size_params{};
  size_params.structSize  = CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE;
  size_params.pPriv       = nullptr;
  size_params.pHostObject = m_host->object;
  cupti_host_call(cuptiProfilerHostGetConfigImageSize(&size_params));

  if (m_host->config_image.size() != size_params.configImageSize)
  {
    m_host->config_image.resize(size_params.configImageSize);
  }

  CUpti_Profiler_Host_GetConfigImage_Params image_params{};
  image_params.structSize      = CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE;
  image_params.pPriv           = nullptr;
  image_params.pHostObject     = m_host->object;
  image_params.configImageSize = m_host->config_image.size();
  image_params.pConfigImage    = m_host->config_image.data();
  cupti_host_call(cuptiProfilerHostGetConfigImage(&image_params));

  m_config_image = m_host->config_image;
}

void cupti_profiler::initialize_counter_data_image()
{
  if (m_host->metric_names.empty())
  {
    initialize_config_image_host();
  }

  enable_range_profiler();
  if (!m_host || m_host->range_profiler_object == nullptr)
  {
    NVBENCH_THROW(std::runtime_error, "{}", kRangeProfilerNullMsg);
  }

  CUpti_RangeProfiler_GetCounterDataSize_Params size_params{};
  size_params.structSize           = CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE;
  size_params.pPriv                = nullptr;
  size_params.pRangeProfilerObject = m_host->range_profiler_object;
  size_params.pMetricNames         = m_host->metric_names.data();
  size_params.numMetrics           = m_host->metric_names.size();
  size_params.maxNumOfRanges       = m_num_ranges;
  size_params.maxNumRangeTreeNodes = m_num_ranges;
  cupti_call(cuptiRangeProfilerGetCounterDataSize(&size_params));

  m_data_image.assign(size_params.counterDataSize, 0);

  CUpti_RangeProfiler_CounterDataImage_Initialize_Params init_params{};
  init_params.structSize = CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE;
  init_params.pPriv      = nullptr;
  init_params.pRangeProfilerObject = m_host->range_profiler_object;
  init_params.pCounterData         = m_data_image.data();
  init_params.counterDataSize      = m_data_image.size();
  cupti_call(cuptiRangeProfilerCounterDataImageInitialize(&init_params));
}

cupti_profiler::~cupti_profiler()
{
  disable_range_profiler();
  deinitialize_profiler_host();
}

bool cupti_profiler::is_initialized() const { return m_available; }

void cupti_profiler::prepare_user_loop()
{
  enable_range_profiler();
  if (!m_host || m_host->range_profiler_object == nullptr)
  {
    NVBENCH_THROW(std::runtime_error, "{}", kRangeProfilerNullMsg);
  }

  set_range_profiler_config();
}

void cupti_profiler::start_user_loop()
{
  start_range_profiler();
  {
    CUpti_RangeProfiler_PushRange_Params params{};

    params.structSize           = CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE;
    params.pRangeProfilerObject = m_host->range_profiler_object;
    params.pRangeName           = kRangeName;

    cupti_call(cuptiRangeProfilerPushRange(&params));
  }
}

void cupti_profiler::stop_user_loop()
{
  {
    CUpti_RangeProfiler_PopRange_Params params{};
    params.structSize           = CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE;
    params.pRangeProfilerObject = m_host->range_profiler_object;
    cupti_call(cuptiRangeProfilerPopRange(&params));
  }

  stop_range_profiler();
}

bool cupti_profiler::is_replay_required() { return !m_all_passes_submitted; }

void cupti_profiler::process_user_loop() { decode_counter_data(); }

std::vector<double> cupti_profiler::get_counter_values()
{
  initialize_profiler_host();

  {
    CUpti_RangeProfiler_GetCounterDataInfo_Params params{};
    params.structSize           = CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE;
    params.pPriv                = nullptr;
    params.pCounterDataImage    = m_data_image.data();
    params.counterDataImageSize = m_data_image.size();
    cupti_call(cuptiRangeProfilerGetCounterDataInfo(&params));

    if (params.numTotalRanges != 1)
    {
      NVBENCH_THROW(std::runtime_error, "{}", "CUPTI expected one range in counter data");
    }
  }

  std::vector<double> result(m_metric_names.size());

  if (m_host->metric_names.empty())
  {
    initialize_config_image_host();
  }

  CUpti_Profiler_Host_EvaluateToGpuValues_Params params{};
  params.structSize           = CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE;
  params.pPriv                = nullptr;
  params.pHostObject          = m_host->object;
  params.pCounterDataImage    = m_data_image.data();
  params.counterDataImageSize = m_data_image.size();
  params.rangeIndex           = 0;
  params.ppMetricNames        = m_host->metric_names.data();
  params.numMetrics           = m_host->metric_names.size();
  params.pMetricValues        = result.data();

  cupti_host_call(cuptiProfilerHostEvaluateToGpuValues(&params));

  return result;
}

void cupti_profiler::finalize_profiler()
{
  if (!m_profiler_guard)
  {
    return;
  }

  cupti_call(m_profiler_guard->deinitialize());
  m_profiler_guard.reset();
}

} // namespace nvbench::detail
