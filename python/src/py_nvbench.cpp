/*
 *  Copyright 2025 NVIDIA Corporation
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

// clang-format off
// Include Pybind11 headers first thing
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// clang-format on

#include <nvbench/nvbench.cuh>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

struct PyObjectDeleter
{
  void operator()(py::object *p)
  {
    const bool initialized = Py_IsInitialized();

#if PY_VERSION_HEX < 0x30d0000
    const bool finalizing = _Py_IsFinalizing();
#else
    const bool finalizing = Py_IsFinalizing();
#endif
    const bool guard = initialized && !finalizing;

    // deleter only call ~object if interpreter is active and
    // not shutting down, let OS clean up resources after
    // interpreter tear-down
    if (guard)
    {
      delete p;
    }
  }
};

struct benchmark_wrapper_t
{

  benchmark_wrapper_t() = default;

  explicit benchmark_wrapper_t(py::object o)
      : m_fn{std::shared_ptr<py::object>(new py::object(std::move(o)), PyObjectDeleter{})}
  {
    if (!PyCallable_Check(m_fn->ptr()))
    {
      throw py::value_error("Argument must be a callable");
    }
  }

  // Only copy constructor is used, delete copy-assign, and moves
  benchmark_wrapper_t(const benchmark_wrapper_t &other)            = default;
  benchmark_wrapper_t &operator=(const benchmark_wrapper_t &other) = delete;
  benchmark_wrapper_t(benchmark_wrapper_t &&) noexcept             = delete;
  benchmark_wrapper_t &operator=(benchmark_wrapper_t &&) noexcept  = delete;

  void operator()(nvbench::state &state, nvbench::type_list<>)
  {
    if (!m_fn)
    {
      throw std::runtime_error("No function to execute");
    }
    // box as Python object, using reference semantics
    auto arg = py::cast(std::ref(state), py::return_value_policy::reference);

    // Execute Python callable
    try
    {
      (*m_fn)(arg);
    }
    catch (const py::error_already_set &e)
    {
      if (e.matches(PyExc_KeyboardInterrupt))
      {
        // interrupt execution of outstanding instances
        throw nvbench::stop_runner_loop(e.what());
      }
      else
      {
        // re-raise
        throw;
      }
    }
  }

private:
  // Important to use shared pointer here rather than py::object directly,
  // since copy constructor must be const (consequence of benchmark::do_clone
  // being const member method)
  std::shared_ptr<py::object> m_fn;
};

// Use struct to ensure public inheritance
struct nvbench_run_error : std::runtime_error
{
  // ask compiler to generate all constructor signatures
  // that are defined for the base class
  using std::runtime_error::runtime_error;
};

PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> exc_storage;

void run_interruptible(nvbench::option_parser &parser)
{
  auto &printer    = parser.get_printer();
  auto &benchmarks = parser.get_benchmarks();

  std::size_t total_states = 0;
  for (auto &bench_ptr : benchmarks)
  {
    total_states += bench_ptr->get_config_count();
  }

  printer.set_completed_state_count(0);
  printer.set_total_state_count(total_states);

  bool skip_remaining_flag = false;
  for (auto &bench_ptr : benchmarks)
  {
    bench_ptr->set_printer(printer);
    bench_ptr->run_or_skip(skip_remaining_flag);
    bench_ptr->clear_printer();
  }
}

class GlobalBenchmarkRegistry
{
  bool m_finalized;

public:
  GlobalBenchmarkRegistry()
      : m_finalized(false) {};

  GlobalBenchmarkRegistry(const GlobalBenchmarkRegistry &)            = delete;
  GlobalBenchmarkRegistry &operator=(const GlobalBenchmarkRegistry &) = delete;

  GlobalBenchmarkRegistry(GlobalBenchmarkRegistry &&)            = delete;
  GlobalBenchmarkRegistry &operator=(GlobalBenchmarkRegistry &&) = delete;

  bool is_finalized() const { return m_finalized; }

  nvbench::benchmark_base &add_bench(py::object fn)
  {
    if (m_finalized)
    {
      throw std::runtime_error("Can not register more benchmarks after benchmark was run");
    }
    if (!PyCallable_Check(fn.ptr()))
    {
      throw py::value_error("Benchmark should be a callable object");
    }
    std::string name;
    if (py::hasattr(fn, "__name__"))
    {
      py::str py_name = fn.attr("__name__");
      name            = py::cast<std::string>(py_name);
    }
    else
    {
      py::str py_name = py::repr(fn);
      name            = py::cast<std::string>(py_name);
    }
    benchmark_wrapper_t executor(fn);

    return nvbench::benchmark_manager::get()
      .add(std::make_unique<nvbench::benchmark<benchmark_wrapper_t>>(executor))
      .set_name(std::move(name));
  }

  void run(const std::vector<std::string> &argv)
  {
    if (nvbench::benchmark_manager::get().get_benchmarks().empty())
    {
      throw std::runtime_error("No benchmarks had been registered yet");
    }
    if (m_finalized)
    {
      throw std::runtime_error("Benchmarks were already executed");
    }
    m_finalized = true;

    try
    {
      // This line is mandatory for correctness to populate
      // benchmark with devices requested by user via CLI
      nvbench::benchmark_manager::get().initialize();
      {
        nvbench::option_parser parser{};
        parser.parse(argv);

        NVBENCH_MAIN_PRINT_PREAMBLE(parser);
        run_interruptible(parser);
        NVBENCH_MAIN_PRINT_EPILOGUE(parser);

        NVBENCH_MAIN_PRINT_RESULTS(parser);
      } /* Tear down parser before finalization */
    }
    catch (py::error_already_set &e)
    {
      py::raise_from(e, exc_storage.get_stored().ptr(), "Python error raised ");
      throw py::error_already_set();
    }
    catch (const std::exception &e)
    {
      const std::string &exc_message = e.what();
      py::set_error(exc_storage.get_stored(), exc_message.c_str());
      throw py::error_already_set();
    }
    catch (...)
    {
      py::set_error(exc_storage.get_stored(), "Caught unknown exception in nvbench_main");
      throw py::error_already_set();
    }
  }
};

py::dict py_get_axis_values(const nvbench::state &state)
{
  auto named_values = state.get_axis_values();

  auto names = named_values.get_names();
  py::dict res;

  for (const auto &name : names)
  {
    if (named_values.has_value(name))
    {
      auto v            = named_values.get_value(name);
      res[name.c_str()] = py::cast(v);
    }
  }

  return res;
}

// essentially a global variable, but allocated on the heap during module initialization
std::unique_ptr<GlobalBenchmarkRegistry, py::nodelete> global_registry{};

// Definitions of Python API
static void def_class_CudaStream(py::module_ m)
{
  // Define CudaStream class
  //    ATTN: nvbench::cuda_stream is move-only class
  //    Methods:
  //       Constructors, based on device, or on existing stream
  //       nvbench::cuda_stream::get_stream

  static constexpr const char *class_CudaStream_doc = R"XXX(
Represents CUDA stream

    Note
    ----
        The class is not user-constructible.
)XXX";

  auto py_cuda_stream_cls = py::class_<nvbench::cuda_stream>(m, "CudaStream", class_CudaStream_doc);

  auto method__cuda_stream__impl =
    [](const nvbench::cuda_stream &s) -> std::pair<std::size_t, std::size_t> {
    return std::make_pair(std::size_t{0}, reinterpret_cast<std::size_t>(s.get_stream()));
  };
  static constexpr const char *method__cuda_stream__doc = R"XXX(
        Special method implement CUDA stream protocol
        from `cuda.core`. Returns a pair of integers:
        (protocol_version, integral_value_of_cudaStream_t pointer)

        Example
        -------
            import cuda.core.experimental as core
            import cuda.bench as bench

            def bench(state: bench.State):
                dev = core.Device(state.get_device())
                dev.set_current()
                # converts CudaString to core.Stream
                # using __cuda_stream__ protocol
                dev.create_stream(state.get_stream())
)XXX";
  py_cuda_stream_cls.def("__cuda_stream__", method__cuda_stream__impl, method__cuda_stream__doc);

  auto method_addressof_impl = [](const nvbench::cuda_stream &s) -> std::size_t {
    return reinterpret_cast<std::size_t>(s.get_stream());
  };
  static constexpr const char *method_addressof_doc =
    R"XXXX(Integral value of address of driver's CUDA stream struct")XXXX";
  py_cuda_stream_cls.def("addressof", method_addressof_impl, method_addressof_doc);
}

void def_class_Launch(py::module_ m)
{
  // Define Launch class
  //    ATTN: nvbench::launch is move-only class
  //    Methods:
  //        nvbench::launch::get_stream -> nvbench::cuda_stream

  static constexpr const char *class_Launch_doc = R"XXXX(
Configuration object for function launch.

    Note
    ----
        The class is not user-constructible.
)XXXX";
  auto py_launch_cls = py::class_<nvbench::launch>(m, "Launch", class_Launch_doc);

  auto method_get_stream_impl = [](nvbench::launch &launch) {
    return std::ref(launch.get_stream());
  };
  static constexpr const char *method_get_stream_doc =
    R"XXXX(Get CUDA stream of this configuration)XXXX";
  py_launch_cls.def("get_stream",
                    method_get_stream_impl,
                    method_get_stream_doc,
                    py::return_value_policy::reference);
}

static void def_class_Benchmark(py::module_ m)
{
  // Define Benchmark class
  //    ATTN: nvbench::benchmark_base is move-only class
  //    Methods:
  //        nvbench::benchmark_base::get_name
  //        nvbench::benchmark_base::add_int64_axis
  //        nvbench::benchmark_base::add_int64_power_of_two_axis
  //        nvbench::benchmark_base::add_float64_axis
  //        nvbench::benchmark_base::add_string_axis
  //        nvbench::benchmark_base::set_name
  //        nvbench::benchmark_base::set_is_cpu_only
  //        nvbench::benchmark_base::set_skip_time
  //        nvbench::benchmark_base::set_timeout
  //        nvbench::benchmark_base::set_throttle_threshold
  //        nvbench::benchmark_base::set_throttle_recovery_delay
  //        nvbench::benchmark_base::set_stopping_criterion
  //        nvbench::benchmark_base::set_criterion_param_int64
  //        nvbench::benchmark_base::set_criterion_param_float64
  //        nvbench::benchmark_base::set_criterion_param_string
  //        nvbench::benchmark_base::set_min_samples

  static constexpr const char *class_Benchmark_doc = R"XXXX(
Represents NVBench benchmark.

    Note
    ----
        The class is not user-constructible.

        Use `~register` function to create Benchmark and register
        it with NVBench.
)XXXX";
  auto py_benchmark_cls = py::class_<nvbench::benchmark_base>(m, "Benchmark", class_Benchmark_doc);

  // method Benchmark.get_name
  auto method_get_name_impl                        = &nvbench::benchmark_base::get_name;
  static constexpr const char *method_get_name_doc = R"XXXX(Get benchmark name)XXXX";
  py_benchmark_cls.def("get_name", method_get_name_impl, method_get_name_doc);

  // method Benchmark.add_int64_axis
  auto method_add_int64_axis_impl =
    [](nvbench::benchmark_base &self, std::string name, std::vector<nvbench::int64_t> data) {
      self.add_int64_axis(std::move(name), std::move(data));
      return std::ref(self);
    };
  static constexpr const char *method_add_int64_axis_doc = R"XXXX(
Add integral type parameter axis with given name and values to sweep over
)XXXX";
  py_benchmark_cls.def("add_int64_axis",
                       method_add_int64_axis_impl,
                       method_add_int64_axis_doc,
                       py::return_value_policy::reference,
                       py::arg("name"),
                       py::arg("values"));

  // method Benchmark.add_int64_power_of_two_axis
  auto method_add_int64_power_of_two_axis_impl = [](nvbench::benchmark_base &self,
                                                    std::string name,
                                                    std::vector<nvbench::int64_t> data) {
    self.add_int64_axis(std::move(name), std::move(data), nvbench::int64_axis_flags::power_of_two);
    return std::ref(self);
  };
  static constexpr const char *method_add_int64_power_of_two_axis_doc = R"XXXX(
Add integral type parameter axis with given name and power of two values to sweep over
)XXXX";
  py_benchmark_cls.def("add_int64_power_of_two_axis",
                       method_add_int64_power_of_two_axis_impl,
                       method_add_int64_power_of_two_axis_doc,
                       py::return_value_policy::reference,
                       py::arg("name"),
                       py::arg("values"));

  // method Benchmark.add_float64_axis
  auto method_add_float64_axis_impl =
    [](nvbench::benchmark_base &self, std::string name, std::vector<nvbench::float64_t> data) {
      self.add_float64_axis(std::move(name), std::move(data));
      return std::ref(self);
    };
  static constexpr const char *method_add_float64_axis_doc = R"XXXX(
Add floating-point type parameter axis with given name and values to sweep over
)XXXX";
  py_benchmark_cls.def("add_float64_axis",
                       method_add_float64_axis_impl,
                       method_add_float64_axis_doc,
                       py::return_value_policy::reference,
                       py::arg("name"),
                       py::arg("values"));

  // method Benchmark.add_string_axis
  auto method_add_string_axis_impl =
    [](nvbench::benchmark_base &self, std::string name, std::vector<std::string> data) {
      self.add_string_axis(std::move(name), std::move(data));
      return std::ref(self);
    };
  static constexpr const char *method_add_string_axis_doc = R"XXXX(
Add string type parameter axis with given name and values to sweep over
)XXXX";
  py_benchmark_cls.def("add_string_axis",
                       method_add_string_axis_impl,
                       method_add_string_axis_doc,
                       py::return_value_policy::reference,
                       py::arg("name"),
                       py::arg("values"));

  // method Benchmark.set_name
  auto method_set_name_impl = [](nvbench::benchmark_base &self, std::string name) {
    self.set_name(std::move(name));
    return std::ref(self);
  };
  static constexpr const char *method_set_name_doc = R"XXXX(Set benchmark name)XXXX";
  py_benchmark_cls.def("set_name",
                       method_set_name_impl,
                       method_set_name_doc,
                       py::return_value_policy::reference,
                       py::arg("name"));

  // method Benchmark.set_is_cpu_only
  auto method_set_is_cpu_only_impl = [](nvbench::benchmark_base &self, bool is_cpu_only) {
    self.set_is_cpu_only(is_cpu_only);
    return std::ref(self);
  };
  static constexpr const char *method_set_is_cpu_only_doc =
    R"XXXX(Set whether this benchmark only executes on CPU)XXXX";
  py_benchmark_cls.def("set_is_cpu_only",
                       method_set_is_cpu_only_impl,
                       method_set_is_cpu_only_doc,
                       py::return_value_policy::reference,
                       py::arg("is_cpu_only"));

  // method Benchmark.set_run_once
  auto method_set_run_once_impl = [](nvbench::benchmark_base &self, bool run_once) {
    self.set_run_once(run_once);
    return std::ref(self);
  };
  static constexpr const char *method_set_run_once_doc = R"XXXX(
Set whether all benchmark configurations are executed only once
)XXXX";
  // TODO: should this be exposed?
  py_benchmark_cls.def("set_run_once",
                       method_set_run_once_impl,
                       method_set_run_once_doc,
                       py::return_value_policy::reference,
                       py::arg("run_once"));

  // method Benchmark.set_skip_time
  auto method_set_skip_time_impl = [](nvbench::benchmark_base &self,
                                      nvbench::float64_t skip_duration_seconds) {
    self.set_skip_time(skip_duration_seconds);
    return std::ref(self);
  };
  static constexpr const char *method_set_skip_time_doc = R"XXXX(
Set value, in seconds, such that runs with duration shorter than this are skipped
)XXXX";
  py_benchmark_cls.def("set_skip_time",
                       method_set_skip_time_impl,
                       method_set_skip_time_doc,
                       py::return_value_policy::reference,
                       py::arg("duration_seconds"));

  // method Benchmark.set_timeout
  auto method_set_timeout_impl = [](nvbench::benchmark_base &self,
                                    nvbench::float64_t duration_seconds) {
    self.set_timeout(duration_seconds);
    return std::ref(self);
  };
  static constexpr const char *method_set_timeout_doc = R"XXXX(
Set benchmark run duration timeout value, in seconds
)XXXX";
  py_benchmark_cls.def("set_timeout",
                       method_set_timeout_impl,
                       method_set_timeout_doc,
                       py::return_value_policy::reference,
                       py::arg("duration_seconds"));

  // method Benchmark.set_throttle_threshold
  auto method_set_throttle_threshold_impl = [](nvbench::benchmark_base &self,
                                               nvbench::float32_t threshold) {
    self.set_throttle_threshold(threshold);
    return std::ref(self);
  };
  static constexpr const char *method_set_throttle_threshold_doc = R"XXXX(
Set throttle threshold, as a fraction of maximal GPU frequency, in percents
)XXXX";
  py_benchmark_cls.def("set_throttle_threshold",
                       method_set_throttle_threshold_impl,
                       method_set_throttle_threshold_doc,
                       py::return_value_policy::reference,
                       py::arg("threshold"));

  // method Benchmark.set_throttle_recovery_delay
  auto method_set_throttle_recovery_delay_impl = [](nvbench::benchmark_base &self,
                                                    nvbench::float32_t delay) {
    self.set_throttle_recovery_delay(delay);
    return std::ref(self);
  };
  static constexpr const char *method_set_throttle_recovery_delay_doc = R"XXXX(
Set throttle recovery delay, in seconds
)XXXX";
  py_benchmark_cls.def("set_throttle_recovery_delay",
                       method_set_throttle_recovery_delay_impl,
                       method_set_throttle_recovery_delay_doc,
                       py::return_value_policy::reference,
                       py::arg("delay_seconds"));

  // method Benchmark.set_stopping_criterion
  auto method_set_stopping_criterion_impl = [](nvbench::benchmark_base &self,
                                               std::string criterion) {
    self.set_stopping_criterion(std::move(criterion));
    return std::ref(self);
  };
  static constexpr const char *method_set_stopping_criterion_doc = R"XXXX(
Set stopping criterion to be used
)XXXX";
  py_benchmark_cls.def("set_stopping_criterion",
                       method_set_stopping_criterion_impl,
                       method_set_stopping_criterion_doc,
                       py::return_value_policy::reference,
                       py::arg("criterion"));

  // method Benchmark.set_criterion_param_int64
  auto method_set_criterion_param_int64_impl =
    [](nvbench::benchmark_base &self, std::string name, nvbench::int64_t value) {
      self.set_criterion_param_int64(std::move(name), value);
      return std::ref(self);
    };
  static constexpr const char *method_set_criterion_param_int64_doc = R"XXXX(
Set stopping criterion integer parameter value
)XXXX";
  py_benchmark_cls.def("set_criterion_param_int64",
                       method_set_criterion_param_int64_impl,
                       method_set_criterion_param_int64_doc,
                       py::return_value_policy::reference,
                       py::arg("name"),
                       py::arg("value"));

  // method Benchmark.set_criterion_param_float64
  auto method_set_criterion_param_float64_impl =
    [](nvbench::benchmark_base &self, std::string name, nvbench::float64_t value) {
      self.set_criterion_param_float64(std::move(name), value);
      return std::ref(self);
    };
  static constexpr const char *method_set_criterion_param_float64_doc = R"XXXX(
Set stopping criterion floating point parameter value"
)XXXX";
  py_benchmark_cls.def("set_criterion_param_float64",
                       method_set_criterion_param_float64_impl,
                       method_set_criterion_param_float64_doc,
                       py::return_value_policy::reference,
                       py::arg("name"),
                       py::arg("value"));

  // method Benchmark.set_criterion_param_string
  auto method_set_criterion_param_string_impl =
    [](nvbench::benchmark_base &self, std::string name, std::string value) {
      self.set_criterion_param_string(std::move(name), std::move(value));
      return std::ref(self);
    };
  static constexpr const char *method_set_criterion_param_string_doc = R"XXXX(
Set stopping criterion string parameter value
)XXXX";
  py_benchmark_cls.def("set_criterion_param_string",
                       method_set_criterion_param_string_impl,
                       method_set_criterion_param_string_doc,
                       py::return_value_policy::reference,
                       py::arg("name"),
                       py::arg("value"));

  // method Benchmark.set_min_samples
  auto method_set_min_samples_impl = [](nvbench::benchmark_base &self, nvbench::int64_t count) {
    self.set_min_samples(count);
    return std::ref(self);
  };
  static constexpr const char *method_set_min_samples_doc = R"XXXX(
Set minimal samples count before stopping criterion applies
)XXXX";
  py_benchmark_cls.def("set_min_samples",
                       method_set_min_samples_impl,
                       method_set_min_samples_doc,
                       py::return_value_policy::reference,
                       py::arg("min_samples_count"));
}

void def_class_State(py::module_ m)
{
  // Define PyState class
  //    ATTN: nvbench::state is move-only class
  //    Methods:
  //        nvbench::state::get_cuda_stream
  //        nvbench::state::get_cuda_stream_optional
  //        nvbench::state::set_cuda_stream
  //        nvbench::state::get_device
  //        nvbench::state::get_is_cpu_only
  //        nvbench::state::get_type_config_index
  //        nvbench::state::get_int64
  //        nvbench::state::get_int64_or_default
  //        nvbench::state::get_float64
  //        nvbench::state::get_float64_or_default
  //        nvbench::state::get_string
  //        nvbench::state::get_string_or_default
  //        nvbench::state::add_element_count
  //        nvbench::state::set_element_count
  //        nvbench::state::get_element_count
  //        nvbench::state::add_global_memory_reads
  //        nvbench::state::add_global_memory_writes
  //        nvbench::state::add_buffer_size
  //        nvbench::state::set_global_memory_rw_bytes
  //        nvbench::state::get_global_memory_rw_bytes
  //        nvbench::state::skip
  //        nvbench::state::is_skipped
  //        nvbench::state::get_skip_reason
  //        nvbench::state::get_min_samples
  //        nvbench::state::set_min_samples
  //        nvbench::state::get_criterion_params
  //        nvbench::state::get_stopping_criterion
  //        nvbench::state::get_run_once
  //        nvbench::state::set_run_once
  //        nvbench::state::get_disable_blocking_kernel
  //        nvbench::state::set_disable_blocking_kernel
  //        nvbench::state::set_skip_time
  //        nvbench::state::get_skip_time
  //        nvbench::state::set_timeout
  //        nvbench::state::get_timeout
  //        nvbench::state::set_throttle_threshold
  //        nvbench::state::get_throttle_threshold
  //        nvbench::state::set_throttle_recovery_delay
  //        nvbench::state::get_throttle_recovery_delay
  //        nvbench::state::get_blocking_kernel_timeout
  //        nvbench::state::set_blocking_kernel_timeout
  //        nvbench::state::get_axis_values
  //        nvbench::state::get_axis_values_as_string
  //        nvbench::state::get_benchmark
  //        nvbench::state::collect_l1_hit_rates
  //        nvbench::state::collect_l2_hit_rates
  //        nvbench::state::collect_stores_efficiency
  //        nvbench::state::collect_loads_efficiency
  //        nvbench::state::collect_dram_throughput
  //        nvbench::state::collect_cupti_metrics
  //        nvbench::state::is_l1_hit_rate_collected
  //        nvbench::state::is_l2_hit_rate_collected
  //        nvbench::state::is_stores_efficiency_collected
  //        nvbench::state::is_loads_efficiency_collected
  //        nvbench::state::is_dram_throughput_collected
  //        nvbench::state::is_cupti_required
  //        nvbench::state::add_summary
  //        nvbench::state::get_summary
  //        nvbench::state::get_summaries
  //        nvbench::state::get_short_description
  //        nvbench::state::exec
  // NOTE:
  //    State wraps std::reference_wrapper<nvbench::state>

  using state_ref_t                            = std::reference_wrapper<nvbench::state>;
  static constexpr const char *class_State_doc = R"XXXX(
Represent benchmark configuration state.

    Note
    ----
        The class is not user-constructible.
)XXXX";
  auto pystate_cls = py::class_<nvbench::state>(m, "State", class_State_doc);

  // method State.has_device
  auto method_has_device_impl = [](const nvbench::state &state) -> bool {
    return static_cast<bool>(state.get_device());
  };
  static constexpr const char *method_has_device_doc = R"XXXX(
Returns True if configuration has a device
)XXXX";
  pystate_cls.def("has_device", method_has_device_impl, method_has_device_doc);

  // method State.has_printers
  auto method_has_printers_impl = [](const nvbench::state &state) -> bool {
    return state.get_benchmark().get_printer().has_value();
  };
  static constexpr const char *method_has_printers_doc = R"XXXX(
Returns True if configuration has a printer"
)XXXX";
  pystate_cls.def("has_printers", method_has_printers_impl, method_has_printers_doc);

  // method State.get_device
  auto method_get_device_impl = [](const nvbench::state &state) {
    auto dev = state.get_device();
    if (dev.has_value())
    {
      return py::cast(dev.value().get_id());
    }
    return py::object(py::none());
  };
  static constexpr const char *method_get_device_doc = R"XXXX(
Get device_id of the device from this configuration
)XXXX";
  pystate_cls.def("get_device", method_get_device_impl, method_get_device_doc);

  // method State.get_stream
  auto method_get_stream_impl = [](nvbench::state &state) {
    return std::ref(state.get_cuda_stream());
  };
  static constexpr const char *method_get_stream_doc = R"XXXX(
Get `~CudaStream` object from this configuration"
)XXXX";
  pystate_cls.def("get_stream",
                  method_get_stream_impl,
                  method_get_stream_doc,
                  py::return_value_policy::reference);

  // method State.get_int64
  auto method_get_int64_impl                        = &nvbench::state::get_int64;
  static constexpr const char *method_get_int64_doc = R"XXXX(
Get value for given Int64 axis from this configuration
)XXXX";
  pystate_cls.def("get_int64", method_get_int64_impl, method_get_int64_doc, py::arg("name"));

  // method State.get_int64_or_default
  auto method_get_int64_or_default_impl = &nvbench::state::get_int64_or_default;
  static constexpr const char *method_get_int64_or_default_doc = method_get_int64_doc;
  pystate_cls.def("get_int64_or_default",
                  method_get_int64_or_default_impl,
                  method_get_int64_or_default_doc,
                  py::arg("name"),
                  py::pos_only{},
                  py::arg("default_value"));

  // method State.get_float64
  auto method_get_float64_impl                        = &nvbench::state::get_float64;
  static constexpr const char *method_get_float64_doc = R"XXXX(
Get value for given Float64 axis from this configuration
)XXXX";
  pystate_cls.def("get_float64", method_get_float64_impl, method_get_float64_doc, py::arg("name"));

  // method State.get_float64_or_default
  static constexpr const char *method_get_float64_or_default_doc = method_get_float64_doc;
  pystate_cls.def("get_float64_or_default",
                  &nvbench::state::get_float64_or_default,
                  method_get_float64_or_default_doc,
                  py::arg("name"),
                  py::pos_only{},
                  py::arg("default_value"));

  // method State.get_string
  static constexpr const char *method_get_string_doc = R"XXXX(
Get value for given String axis from this configuration
)XXXX";
  pystate_cls.def("get_string", &nvbench::state::get_string, method_get_string_doc, py::arg("name"));

  // method State.get_string_or_default
  static constexpr const char *method_get_string_or_default_doc = method_get_string_doc;
  pystate_cls.def("get_string_or_default",
                  &nvbench::state::get_string_or_default,
                  method_get_string_or_default_doc,
                  py::arg("name"),
                  py::pos_only{},
                  py::arg("default_value"));

  // method State.get_element_count
  static constexpr const char *method_add_element_count_doc = R"XXXX(
Add element count"
)XXXX";
  pystate_cls.def("add_element_count",
                  &nvbench::state::add_element_count,
                  method_add_element_count_doc,
                  py::arg("count"),
                  py::arg("column_name") = py::str(""));

  // method State.set_element_count
  static constexpr const char *method_set_element_count_doc = R"XXXX(
Set element count
)XXXX";
  pystate_cls.def("set_element_count",
                  &nvbench::state::set_element_count,
                  method_set_element_count_doc,
                  py::arg("count"));

  // method State.get_element_count
  static constexpr const char *method_get_element_count = R"XXXX(
Get element count
)XXXX";
  pystate_cls.def("get_element_count",
                  &nvbench::state::get_element_count,
                  method_get_element_count);

  // method State.skip
  static constexpr const char *method_skip_doc = "Skip this configuration";
  pystate_cls.def("skip", &nvbench::state::skip, py::arg("reason"));

  // method State.is_skipped
  static constexpr const char *method_is_skipped_doc = R"XXXX(
Returns True if this configuration is being skipped";
)XXXX";
  pystate_cls.def("is_skipped", &nvbench::state::is_skipped, method_is_skipped_doc);

  // method State.get_skip_reason
  static constexpr const char *method_get_skip_reason_doc = R"XXXX(
Get reason provided for skipping this configuration
)XXXX";
  pystate_cls.def("get_skip_reason", &nvbench::state::get_skip_reason, method_get_skip_reason_doc);

  // method State.add_global_memory_reads
  auto method_add_global_memory_reads_impl =
    [](nvbench::state &state, std::size_t nbytes, const std::string &column_name) -> void {
    state.add_global_memory_reads(nbytes, column_name);
  };
  static constexpr const char *method_add_global_memory_reads_doc = R"XXXX(
Inform NVBench that given amount of bytes is being read by the benchmark from global memory
)XXXX";
  pystate_cls.def("add_global_memory_reads",
                  method_add_global_memory_reads_impl,
                  method_add_global_memory_reads_doc,
                  py::arg("nbytes"),
                  py::pos_only{},
                  py::arg("column_name") = py::str(""));

  // method State.add_global_memory_writes
  auto method_add_global_memory_writes_impl =
    [](nvbench::state &state, std::size_t nbytes, const std::string &column_name) -> void {
    state.add_global_memory_writes(nbytes, column_name);
  };
  static constexpr const char *method_add_global_memory_writes_doc = R"XXXX(
Inform NVBench that given amount of bytes is being written by the benchmark into global memory
)XXXX";
  pystate_cls.def("add_global_memory_writes",
                  method_add_global_memory_writes_impl,
                  method_add_global_memory_writes_doc,
                  py::arg("nbytes"),
                  py::pos_only{},
                  py::arg("column_name") = py::str(""));

  // method State.get_benchmark
  auto method_get_benchmark_impl = [](const nvbench::state &state) {
    return std::ref(state.get_benchmark());
  };
  static constexpr const char *method_get_benchmark_doc = R"XXXX(
Get Benchmark this configuration is a part of
)XXXX";
  pystate_cls.def("get_benchmark",
                  method_get_benchmark_impl,
                  method_get_benchmark_doc,
                  py::return_value_policy::reference);

  // method State.get_throttle_threshold
  static constexpr const char *method_get_throttle_threshold_doc = R"XXXX(
Get throttle threshold value, as fraction of maximal frequency.

Note
----
    A valid threshold value is between 0 and 1.
)XXXX";
  pystate_cls.def("get_throttle_threshold",
                  &nvbench::state::get_throttle_threshold,
                  method_get_throttle_threshold_doc);

  // method State.set_throttle_threshold
  static constexpr const char *method_set_throttle_threshold_doc = R"XXXX(
Set throttle threshold fraction to the specified value, expected to be between 0 and 1"
)XXXX";
  pystate_cls.def("set_throttle_threshold",
                  &nvbench::state::set_throttle_threshold,
                  method_set_throttle_threshold_doc,
                  py::arg("throttle_fraction"));

  // method State.get_min_samples
  static constexpr const char *method_get_min_samples_doc = R"XXXX(
Get the number of benchmark timings NVBench performs before stopping criterion begins being used
)XXXX";
  pystate_cls.def("get_min_samples", &nvbench::state::get_min_samples, method_get_min_samples_doc);

  // method State.set_min_samples
  static constexpr const char *method_set_min_samples_doc = R"XXXX(
Set the number of benchmark timings for NVBench to perform before stopping criterion begins being used
)XXXX";
  pystate_cls.def("set_min_samples",
                  &nvbench::state::set_min_samples,
                  method_set_min_samples_doc,
                  py::arg("min_samples_count"));

  // method State.get_disable_blocking_kernel
  static constexpr const char *method_get_disable_blocking_kernel_doc = R"XXXX(
Return True if use of blocking kernel by NVBench is disabled, False otherwise
)XXXX";
  pystate_cls.def("get_disable_blocking_kernel",
                  &nvbench::state::get_disable_blocking_kernel,
                  method_get_disable_blocking_kernel_doc);

  // method State.set_disable_blocking_kernel
  static constexpr const char *method_set_disable_blocking_kernel_doc = R"XXXX(
Use argument True to disable use of blocking kernel by NVBench"
)XXXX";
  pystate_cls.def("set_disable_blocking_kernel",
                  &nvbench::state::set_disable_blocking_kernel,
                  method_set_disable_blocking_kernel_doc,
                  py::arg("disable_blocking_kernel"));

  // method State.get_run_once
  static constexpr const char *method_get_run_once_doc =
    R"XXXX(Boolean flag indicating whether configuration should only run once)XXXX";
  pystate_cls.def("get_run_once", &nvbench::state::get_run_once, method_get_run_once_doc);

  // method State.set_run_once
  static constexpr const char *method_set_run_once_doc =
    R"XXXX(Set run-once flag for this configuration)XXXX";
  pystate_cls.def("set_run_once",
                  &nvbench::state::set_run_once,
                  method_set_run_once_doc,
                  py::arg("run_once"));

  // method State.get_timeout
  static constexpr const char *method_get_timeout_doc =
    R"XXXX(Get time-out value for benchmark execution of this configuration, in seconds)XXXX";
  pystate_cls.def("get_timeout", &nvbench::state::get_timeout, method_get_timeout_doc);

  // method State.set_timeout
  static constexpr const char *method_set_timeout_doc =
    R"XXXX(Set time-out value for benchmark execution of this configuration, in seconds)XXXX";
  pystate_cls.def("set_timeout",
                  &nvbench::state::set_timeout,
                  method_set_timeout_doc,
                  py::arg("duration_seconds"));

  // method State.get_blocking_kernel_timeout
  static constexpr const char *method_get_blocking_kernel_timeout_doc =
    R"XXXX(Get time-out value for execution of blocking kernel, in seconds)XXXX";
  pystate_cls.def("get_blocking_kernel_timeout",
                  &nvbench::state::get_blocking_kernel_timeout,
                  method_get_blocking_kernel_timeout_doc);

  // method State.set_blocking_kernel_timeout
  static constexpr const char *method_set_blocking_kernel_timeout_doc =
    R"XXXX(Set time-out value for execution of blocking kernel, in seconds)XXXX";
  pystate_cls.def("set_blocking_kernel_timeout",
                  &nvbench::state::set_blocking_kernel_timeout,
                  method_set_blocking_kernel_timeout_doc,
                  py::arg("duration_seconds"));

  // method State.collect_cupti_metrics
  static constexpr const char *method_collect_cupti_metrics_doc =
    R"XXXX(Request NVBench to record CUPTI metrics while running benchmark for this configuration)XXXX";
  pystate_cls.def("collect_cupti_metrics",
                  &nvbench::state::collect_cupti_metrics,
                  method_collect_cupti_metrics_doc);

  // method State.is_cupti_required
  static constexpr const char *method_is_cupti_required_doc =
    R"XXXX(True if (some) CUPTI metrics are being collected)XXXX";
  pystate_cls.def("is_cupti_required",
                  &nvbench::state::is_cupti_required,
                  method_is_cupti_required_doc);

  // method State.exec
  auto method_exec_impl =
    [](nvbench::state &state, py::object py_launcher_fn, bool batched, bool sync) -> void {
    if (!PyCallable_Check(py_launcher_fn.ptr()))
    {
      throw py::type_error("Argument of exec method must be a callable object");
    }

    // wrapper to invoke Python callable
    auto cpp_launcher_fn = [py_launcher_fn](nvbench::launch &launch_descr) -> void {
      // cast C++ object to python object
      auto launch_pyarg = py::cast(std::ref(launch_descr), py::return_value_policy::reference);
      // call Python callable
      py_launcher_fn(launch_pyarg);
    };

    if (sync)
    {
      if (batched)
      {
        constexpr auto tag = nvbench::exec_tag::sync;
        state.exec(tag, cpp_launcher_fn);
      }
      else
      {
        constexpr auto tag = nvbench::exec_tag::sync | nvbench::exec_tag::no_batch;
        state.exec(tag, cpp_launcher_fn);
      }
    }
    else
    {
      if (batched)
      {
        constexpr auto tag = nvbench::exec_tag::none;
        state.exec(tag, cpp_launcher_fn);
      }
      else
      {
        constexpr auto tag = nvbench::exec_tag::no_batch;
        state.exec(tag, cpp_launcher_fn);
      }
    }
  };
  static constexpr const char *method_exec_doc = R"XXXX(
Execute callable running the benchmark.

    The callable may be executed multiple times. The callable
    will be passed `~Launch` object argument.

    Parameters
    ----------
        fn: Callable
            Python callable with signature fn(Launch) -> None that executes the benchmark.
        batched: bool, optional
            If `True`, no cache flushing is performed between callable invocations.
            Default: `True`.
        sync: bool, optional
            True value indicates that callable performs device synchronization.
            NVBench disables use of blocking kernel in this case.
            Default: `False`.

)XXXX";
  pystate_cls.def("exec",
                  method_exec_impl,
                  method_exec_doc,
                  py::arg("launcher_fn"),
                  py::pos_only{},
                  py::arg("batched") = true,
                  py::arg("sync")    = false);

  // method State.get_short_description
  static constexpr const char *method_get_short_description_doc = R"XXXX(
Get short description for this configuration
)XXXX";
  pystate_cls.def("get_short_description",
                  &nvbench::state::get_short_description,
                  method_get_short_description_doc);

  // method State.add_summary
  auto method_add_summary_string_value_impl =
    [](nvbench::state &state, std::string column_name, std::string value) {
      auto &summ = state.add_summary("nv/python/" + column_name);
      summ.set_string("description", "User tag: " + column_name);
      summ.set_string("name", std::move(column_name));
      summ.set_string("value", std::move(value));
    };
  static constexpr const char *method_add_summary_doc = R"XXXX(
Add summary column with given name and value
)XXXX";
  pystate_cls.def("add_summary",
                  method_add_summary_string_value_impl,
                  method_add_summary_doc,
                  py::arg("name"),
                  py::arg("value"));

  auto method_add_summary_int64_value_impl =
    [](nvbench::state &state, std::string column_name, nvbench::int64_t value) {
      auto &summ = state.add_summary("nv/python/" + column_name);
      summ.set_string("description", "User tag: " + column_name);
      summ.set_string("name", std::move(column_name));
      summ.set_int64("value", value);
    };
  pystate_cls.def("add_summary",
                  method_add_summary_int64_value_impl,
                  method_add_summary_doc,
                  py::arg("name"),
                  py::arg("value"));

  auto method_add_summary_float64_value_impl =
    [](nvbench::state &state, std::string column_name, nvbench::float64_t value) {
      auto &summ = state.add_summary("nv/python/" + column_name);
      summ.set_string("description", "User tag: " + column_name);
      summ.set_string("name", std::move(column_name));
      summ.set_float64("value", value);
    };
  pystate_cls.def("add_summary",
                  method_add_summary_float64_value_impl,
                  method_add_summary_doc,
                  py::arg("name"),
                  py::arg("value"));

  // method State.get_axis_values_as_string
  static constexpr const char *method_get_axis_values_as_string_doc = R"XXXX(
Get string of space-separated name=value pairs for this configuration
)XXXX";
  pystate_cls.def("get_axis_values_as_string",
                  &nvbench::state::get_axis_values_as_string,
                  method_get_axis_values_as_string_doc);

  // method State.get_axis_values
  static constexpr const char *method_get_axis_values_doc = R"XXXX(
Get dictionary with axis values for this configuration
)XXXX";
  pystate_cls.def("get_axis_values", &py_get_axis_values, method_get_axis_values_doc);

  // method State.get_stopping_criterion
  static constexpr const char *method_get_stopping_criterion_doc = R"XXXX(
Get string name of the stopping criterion used
)XXXX";
  pystate_cls.def("get_stopping_criterion",
                  &nvbench::state::get_stopping_criterion,
                  method_get_stopping_criterion_doc);
}

} // namespace

// ==========================================
// PLEASE KEEP IN SYNC WITH __init__.pyi FILE
// ==========================================
// If you modify these bindings, please be sure to update the
// corresponding type hints in ``../cuda/nvbench/__init__.pyi``

#ifndef PYBIND11_MODULE_NAME
#define PYBIND11_MODULE_NAME _nvbench
#endif

PYBIND11_MODULE(PYBIND11_MODULE_NAME, m)
{
  NVBENCH_DRIVER_API_CALL(cuInit(0));

  // This line ensures that benchmark_manager has been created during module init
  // It is reinitialized before running all benchmarks to set devices to use
  nvbench::benchmark_manager::get().initialize();

  def_class_CudaStream(m);

  def_class_Launch(m);

  def_class_Benchmark(m);

  def_class_State(m);

  // Use handle to take a memory leak here, since this object's destructor may be called after
  // interpreter has shut down
  static constexpr const char *exception_nvbench_runtime_error_doc = R"XXXX(
An exception raised if running benchmarks encounters an error
)XXXX";
  exc_storage.call_once_and_store_result([&]() {
    py::object benchmark_exc_ =
      py::exception<nvbench_run_error>(m, "NVBenchRuntimeError", PyExc_RuntimeError);
    benchmark_exc_.attr("__doc__") = exception_nvbench_runtime_error_doc;
    return benchmark_exc_;
  });

  // ATTN: nvbench::benchmark_manager is a singleton, it is exposed through
  // GlobalBenchmarkRegistry class
  global_registry =
    std::unique_ptr<GlobalBenchmarkRegistry, py::nodelete>(new GlobalBenchmarkRegistry(),
                                                           py::nodelete{});

  // function register
  auto func_register_impl = [](py::object fn) { return std::ref(global_registry->add_bench(fn)); };
  static constexpr const char *func_register_doc = R"XXXX(
Register benchmark function of type Callable[[nvbench.State], None]
)XXXX";
  m.def("register",
        func_register_impl,
        func_register_doc,
        py::return_value_policy::reference,
        py::arg("benchmark_fn"));

  // function run_all_benchmarks
  auto func_run_all_benchmarks_impl = [&](py::object argv) -> void {
    if (!py::isinstance<py::list>(argv))
    {
      throw py::type_error("run_all_benchmarks expects a list of command-line arguments");
    }
    std::vector<std::string> args = py::cast<std::vector<std::string>>(argv);
    global_registry->run(args);
  };
  static constexpr const char *func_run_all_benchmarks_doc = R"XXXX(
    Run all benchmarks registered with NVBench.

    Parameters
    ----------
    argv: List[str]
        Sequence of CLI arguments controlling NVBench. Usually, it is `sys.argv`.
)XXXX";
  m.def("run_all_benchmarks",
        func_run_all_benchmarks_impl,
        func_run_all_benchmarks_doc,
        py::arg("argv") = py::list());

  // Testing utilities
  m.def("test_cpp_exception", []() { throw nvbench_run_error("Test"); });
  m.def("test_py_exception", []() {
    py::set_error(exc_storage.get_stored(), "Test");
    throw py::error_already_set();
  });
}
