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

#include <nvbench/axes_metadata.cuh>
#include <nvbench/device_info.cuh>
#include <nvbench/device_manager.cuh>
#include <nvbench/state.cuh>

#include <functional> // reference_wrapper, ref
#include <memory>
#include <optional>
#include <vector>

namespace nvbench
{

struct printer_base;
struct runner_base;

template <typename BenchmarkType>
struct runner;

/**
 * Hold runtime benchmark information and provides public customization API for
 * the `NVBENCH_BENCH` macros.
 *
 * Delegates responsibility to the following classes:
 * - nvbench::axes_metadata: Axis specifications.
 */
struct benchmark_base
{
  template <typename T>
  using optional_ref = std::optional<std::reference_wrapper<T>>;

  template <typename TypeAxes>
  explicit benchmark_base(TypeAxes type_axes)
      : m_axes(type_axes)
      , m_devices(nvbench::device_manager::get().get_devices())
  {}

  virtual ~benchmark_base();

  /**
   * Returns a pointer to a new instance of the concrete benchmark<...>
   * subclass.
   *
   * The result will have the same name and axes as the source benchmark.
   * The `get_states()` vector of the result will always be empty.
   */
  [[nodiscard]] std::unique_ptr<benchmark_base> clone() const;

  benchmark_base &set_name(std::string name)
  {
    m_name = std::move(name);
    return *this;
  }

  [[nodiscard]] const std::string &get_name() const { return m_name; }

  benchmark_base &set_type_axes_names(std::vector<std::string> names)
  {
    this->do_set_type_axes_names(std::move(names));
    return *this;
  }

  benchmark_base &add_float64_axis(std::string name,
                                   std::vector<nvbench::float64_t> data)
  {
    m_axes.add_float64_axis(std::move(name), std::move(data));
    return *this;
  }

  benchmark_base &add_int64_axis(
    std::string name,
    std::vector<nvbench::int64_t> data,
    nvbench::int64_axis_flags flags = nvbench::int64_axis_flags::none)
  {
    m_axes.add_int64_axis(std::move(name), std::move(data), flags);
    return *this;
  }

  benchmark_base &add_int64_power_of_two_axis(std::string name,
                                              std::vector<nvbench::int64_t> data)
  {
    return this->add_int64_axis(std::move(name),
                                std::move(data),
                                nvbench::int64_axis_flags::power_of_two);
  }

  benchmark_base &add_string_axis(std::string name,
                                  std::vector<std::string> data)
  {
    m_axes.add_string_axis(std::move(name), std::move(data));
    return *this;
  }

  /// Construct a zip iteration space from the provided value axes.
  ///
  /// When axes are zipped together they are iterated like a tuple
  /// of values instead of separate parameters. For example two
  /// value axes of 5 entries will generate 25 combinations, but
  /// when zipped will generate 5 combinations.
  ///
  /// @param[axes] a set of axis_base to be added to the benchmark
  /// and zipped together
  ///
  template<typename... Axes>
  benchmark_base &add_zip_axes(Axes&&... axes)
  {
    m_axes.add_zip_axes(std::forward<Axes>(axes)...);
    return *this;
  }
  /// @}

  /// Construct a user iteration space from the provided value axes.
  ///
  /// Instead of using the standard iteration over each axes, they
  /// are iterated using the custom user iterator that was provided.
  /// This allows for fancy iteration such as using every other
  /// value, random sampling, etc.
  ///
  /// @param[args] First argument is a `std::function<nvbench::make_user_space_signature>`
  /// which constructs the user iteration space, and the reseet are axis_base to be
  /// added to the benchmark and iterated using the user iteration space
  ///
  template<typename... ConstructorAndAxes>
  benchmark_base &add_user_iteration_axes(ConstructorAndAxes&&... args)
  {
    m_axes.add_user_iteration_axes(std::forward<ConstructorAndAxes>(args)...);
    return *this;
  }
  /// @}

  benchmark_base &set_devices(std::vector<int> device_ids);

  benchmark_base &set_devices(std::vector<nvbench::device_info> devices)
  {
    m_devices = std::move(devices);
    return *this;
  }

  benchmark_base &clear_devices()
  {
    m_devices.clear();
    return *this;
  }

  benchmark_base &add_device(int device_id);

  benchmark_base &add_device(nvbench::device_info device)
  {
    m_devices.push_back(std::move(device));
    return *this;
  }

  [[nodiscard]] const std::vector<nvbench::device_info> &get_devices() const
  {
    return m_devices;
  }

  [[nodiscard]] nvbench::axes_metadata &get_axes() { return m_axes; }

  [[nodiscard]] const nvbench::axes_metadata &get_axes() const
  {
    return m_axes;
  }

  // Computes the number of configs in the benchmark.
  // Unlike get_states().size(), this method may be used prior to calling run().
  [[nodiscard]] std::size_t get_config_count() const;

  // Is empty until run() is called.
  [[nodiscard]] const std::vector<nvbench::state> &get_states() const
  {
    return m_states;
  }
  [[nodiscard]] std::vector<nvbench::state> &get_states() { return m_states; }

  void run() { this->do_run(); }

  void set_printer(nvbench::printer_base &printer)
  {
    m_printer = std::ref(printer);
  }

  void clear_printer() { m_printer = std::nullopt; }

  [[nodiscard]] optional_ref<nvbench::printer_base> get_printer() const
  {
    return m_printer;
  }

  /// Execute at least this many trials per measurement. @{
  [[nodiscard]] nvbench::int64_t get_min_samples() const
  {
    return m_min_samples;
  }
  benchmark_base &set_min_samples(nvbench::int64_t min_samples)
  {
    m_min_samples = min_samples;
    return *this;
  }
  /// @}

  /// If true, the benchmark is only run once, skipping all warmup runs and only
  /// executing a single non-batched measurement. This is intended for use with
  /// external profiling tools. @{
  [[nodiscard]] bool get_run_once() const { return m_run_once; }
  benchmark_base &set_run_once(bool v)
  {
    m_run_once = v;
    return *this;
  }
  /// @}

  /// If true, the benchmark does not use the blocking_kernel. This is intended 
  /// for use with external profiling tools. @{
  [[nodiscard]] bool get_disable_blocking_kernel() const { return m_disable_blocking_kernel; }
  benchmark_base &set_disable_blocking_kernel(bool v)
  {
    m_disable_blocking_kernel = v;
    return *this;
  }
  /// @}

  /// Accumulate at least this many seconds of timing data per measurement. @{
  [[nodiscard]] nvbench::float64_t get_min_time() const { return m_min_time; }
  benchmark_base &set_min_time(nvbench::float64_t min_time)
  {
    m_min_time = min_time;
    return *this;
  }
  /// @}

  /// Specify the maximum amount of noise if a measurement supports noise.
  /// Noise is the relative standard deviation:
  /// `noise = stdev / mean_time`. @{
  [[nodiscard]] nvbench::float64_t get_max_noise() const { return m_max_noise; }
  benchmark_base &set_max_noise(nvbench::float64_t max_noise)
  {
    m_max_noise = max_noise;
    return *this;
  }
  /// @}

  /// If a warmup run finishes in less than `skip_time`, the measurement will
  /// be skipped.
  /// Extremely fast kernels (< 5000 ns) often timeout before they can
  /// accumulate `min_time` measurements, and are often uninteresting. Setting
  /// this value can help improve performance by skipping time consuming
  /// measurement that don't provide much information.
  /// Default value is -1., which disables the feature.
  /// @{
  [[nodiscard]] nvbench::float64_t get_skip_time() const { return m_skip_time; }
  benchmark_base &set_skip_time(nvbench::float64_t skip_time)
  {
    m_skip_time = skip_time;
    return *this;
  }
  /// @}

  /// If a measurement take more than `timeout` seconds to complete, stop the
  /// measurement early. A warning should be printed if this happens.
  /// This setting overrides all other termination criteria.
  /// Note that this is measured in CPU walltime, not sample time.
  /// @{
  [[nodiscard]] nvbench::float64_t get_timeout() const { return m_timeout; }
  benchmark_base &set_timeout(nvbench::float64_t timeout)
  {
    m_timeout = timeout;
    return *this;
  }
  /// @}

protected:

  /// Move existing Axis to being part of zip axis iteration space.
  /// This will remove any existing iteration spaces that the named axis
  /// are part of, while restoring all other axis in those spaces to
  /// the default linear space
  ///
  /// This is meant to be used only by the option_parser
  ///  @{
  benchmark_base &zip_axes(std::vector<std::string> names)
  {
    m_axes.zip_axes(std::move(names));
    return *this;
  }
  /// @}


  /// Move existing Axis to being part of user axis iteration space.
  /// This will remove any existing iteration spaces that the named axis
  /// are part of, while restoring all other axis in those spaces to
  /// the default linear space
  ///
  /// This is meant to be used only by the option_parser
  ///  @{
  benchmark_base &
  user_iteration_axes(std::function<nvbench::make_user_space_signature> make,
                      std::vector<std::string> names)
  {
    m_axes.user_iteration_axes(std::move(make), std::move(names));
    return *this;
  }
  /// @}

  friend struct nvbench::runner_base;

  template <typename BenchmarkType>
  friend struct nvbench::runner;

  std::string m_name;
  nvbench::axes_metadata m_axes;
  std::vector<nvbench::device_info> m_devices;
  std::vector<nvbench::state> m_states;

  optional_ref<nvbench::printer_base> m_printer;

  bool m_run_once{false};
  bool m_disable_blocking_kernel{false};

  nvbench::int64_t m_min_samples{10};
  nvbench::float64_t m_min_time{0.5};
  nvbench::float64_t m_max_noise{0.005}; // 0.5% relative standard deviation

  nvbench::float64_t m_skip_time{-1.};
  nvbench::float64_t m_timeout{15.};

private:
  // route these through virtuals so the templated subclass can inject type info
  virtual std::unique_ptr<benchmark_base> do_clone() const            = 0;
  virtual void do_set_type_axes_names(std::vector<std::string> names) = 0;
  virtual void do_run()                                               = 0;
};

} // namespace nvbench
