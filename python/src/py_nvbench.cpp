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
    (*m_fn)(arg);
  }

private:
  // Important to use shared pointer here rather than py::object directly,
  // since copy constructor must be const (benchmark::do_clone is const member method)
  std::shared_ptr<py::object> m_fn;
};

// Use struct to ensure public inheritance
struct nvbench_run_error : std::runtime_error
{
  // ask compiler to generate all constructor signatures
  // that are defined for the base class
  using std::runtime_error::runtime_error;
};
py::handle benchmark_exc{};

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
        NVBENCH_MAIN_RUN_BENCHMARKS(parser);
        NVBENCH_MAIN_PRINT_EPILOGUE(parser);

        NVBENCH_MAIN_PRINT_RESULTS(parser);
      } /* Tear down parser before finalization */
    }
    catch (py::error_already_set &e)
    {
      py::raise_from(e, benchmark_exc.ptr(), "Python error raised ");
      throw py::error_already_set();
    }
    catch (const std::exception &e)
    {
      const std::string &exc_message = e.what();
      py::set_error(benchmark_exc, exc_message.c_str());
      throw py::error_already_set();
    }
    catch (...)
    {
      py::set_error(benchmark_exc, "Caught unknown exception in nvbench_main");
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

} // end of anonymous namespace

// ==========================================
// PLEASE KEEP IN SYNC WITH __init__.pyi FILE
// ==========================================
// If you modify these bindings, please be sure to update the
// corresponding type hints in ``../cuda/nvbench/__init__.pyi``

PYBIND11_MODULE(_nvbench, m)
{
  // == STEP 1
  // Set environment variable CUDA_MODULE_LOADING=EAGER

  NVBENCH_DRIVER_API_CALL(cuInit(0));

  // This line ensures that benchmark_manager has been created during module init
  // It is reinitialized before running all benchmarks to set devices to use
  nvbench::benchmark_manager::get().initialize();

  // == STEP 2
  // Define CudaStream class
  //    ATTN: nvbench::cuda_stream is move-only class
  //    Methods:
  //       Constructors, based on device, or on existing stream
  //       nvbench::cuda_stream::get_stream

  auto py_cuda_stream_cls = py::class_<nvbench::cuda_stream>(m, "CudaStream");

  py_cuda_stream_cls.def("__cuda_stream__",
                         [](const nvbench::cuda_stream &s) -> std::pair<std::size_t, std::size_t> {
                           return std::make_pair(std::size_t{0},
                                                 reinterpret_cast<std::size_t>(s.get_stream()));
                         });
  py_cuda_stream_cls.def("addressof", [](const nvbench::cuda_stream &s) -> std::size_t {
    return reinterpret_cast<std::size_t>(s.get_stream());
  });

  // == STEP 3
  // Define Launch class
  //    ATTN: nvbench::launch is move-only class
  //    Methods:
  //        nvbench::launch::get_stream -> nvbench::cuda_stream

  auto py_launch_cls = py::class_<nvbench::launch>(m, "Launch");

  py_launch_cls.def(
    "get_stream",
    [](nvbench::launch &launch) { return std::ref(launch.get_stream()); },
    py::return_value_policy::reference);

  // == STEP 4
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

  auto py_benchmark_cls = py::class_<nvbench::benchmark_base>(m, "Benchmark");
  py_benchmark_cls.def("get_name", &nvbench::benchmark_base::get_name);
  py_benchmark_cls.def(
    "add_int64_axis",
    [](nvbench::benchmark_base &self, std::string name, std::vector<nvbench::int64_t> data) {
      self.add_int64_axis(std::move(name), std::move(data));
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("name"),
    py::arg("values"));
  py_benchmark_cls.def(
    "add_int64_power_of_two_axis",
    [](nvbench::benchmark_base &self, std::string name, std::vector<nvbench::int64_t> data) {
      self.add_int64_axis(std::move(name),
                          std::move(data),
                          nvbench::int64_axis_flags::power_of_two);
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("name"),
    py::arg("values"));
  py_benchmark_cls.def(
    "add_float64_axis",
    [](nvbench::benchmark_base &self, std::string name, std::vector<nvbench::float64_t> data) {
      self.add_float64_axis(std::move(name), std::move(data));
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("name"),
    py::arg("values"));
  py_benchmark_cls.def(
    "add_string_axis",
    [](nvbench::benchmark_base &self, std::string name, std::vector<std::string> data) {
      self.add_string_axis(std::move(name), std::move(data));
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("name"),
    py::arg("values"));
  py_benchmark_cls.def(
    "set_name",
    [](nvbench::benchmark_base &self, std::string name) {
      self.set_name(std::move(name));
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("name"));
  py_benchmark_cls.def(
    "set_is_cpu_only",
    [](nvbench::benchmark_base &self, bool is_cpu_only) {
      self.set_is_cpu_only(is_cpu_only);
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("is_cpu_only"));
  // TODO: should this be exposed?
  py_benchmark_cls.def(
    "set_run_once",
    [](nvbench::benchmark_base &self, bool run_once) {
      self.set_run_once(run_once);
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("run_once"));
  py_benchmark_cls.def(
    "set_skip_time",
    [](nvbench::benchmark_base &self, nvbench::float64_t skip_duration_seconds) {
      self.set_skip_time(skip_duration_seconds);
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("duration_seconds"));
  py_benchmark_cls.def(
    "set_timeout",
    [](nvbench::benchmark_base &self, nvbench::float64_t duration_seconds) {
      self.set_timeout(duration_seconds);
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("duration_seconds"));
  py_benchmark_cls.def(
    "set_throttle_threshold",
    [](nvbench::benchmark_base &self, nvbench::float32_t threshold) {
      self.set_throttle_threshold(threshold);
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("threshold"));
  py_benchmark_cls.def(
    "set_throttle_recovery_delay",
    [](nvbench::benchmark_base &self, nvbench::float32_t delay) {
      self.set_throttle_recovery_delay(delay);
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("delay_seconds"));
  py_benchmark_cls.def(
    "set_stopping_criterion",
    [](nvbench::benchmark_base &self, std::string criterion) {
      self.set_stopping_criterion(std::move(criterion));
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("criterion"));
  py_benchmark_cls.def(
    "set_criterion_param_int64",
    [](nvbench::benchmark_base &self, std::string name, nvbench::int64_t value) {
      self.set_criterion_param_int64(std::move(name), value);
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("name"),
    py::arg("value"));
  py_benchmark_cls.def(
    "set_criterion_param_float64",
    [](nvbench::benchmark_base &self, std::string name, nvbench::float64_t value) {
      self.set_criterion_param_float64(std::move(name), value);
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("name"),
    py::arg("value"));
  py_benchmark_cls.def(
    "set_criterion_param_string",
    [](nvbench::benchmark_base &self, std::string name, std::string value) {
      self.set_criterion_param_string(std::move(name), std::move(value));
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("name"),
    py::arg("value"));
  py_benchmark_cls.def(
    "set_min_samples",
    [](nvbench::benchmark_base &self, nvbench::int64_t count) {
      self.set_min_samples(count);
      return std::ref(self);
    },
    py::return_value_policy::reference,
    py::arg("min_samples_count"));

  // == STEP 5
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

  using state_ref_t = std::reference_wrapper<nvbench::state>;
  auto pystate_cls  = py::class_<nvbench::state>(m, "State");

  pystate_cls.def("has_device", [](const nvbench::state &state) -> bool {
    return static_cast<bool>(state.get_device());
  });
  pystate_cls.def("has_printers", [](const nvbench::state &state) -> bool {
    return state.get_benchmark().get_printer().has_value();
  });
  pystate_cls.def("get_device", [](const nvbench::state &state) {
    auto dev = state.get_device();
    if (dev.has_value())
    {
      return py::cast(dev.value().get_id());
    }
    return py::object(py::none());
  });

  pystate_cls.def(
    "get_stream",
    [](nvbench::state &state) { return std::ref(state.get_cuda_stream()); },
    py::return_value_policy::reference);

  pystate_cls.def("get_int64", &nvbench::state::get_int64, py::arg("name"));
  pystate_cls.def("get_int64_or_default",
                  &nvbench::state::get_int64_or_default,
                  py::arg("name"),
                  py::pos_only{},
                  py::arg("default_value"));

  pystate_cls.def("get_float64", &nvbench::state::get_float64, py::arg("name"));
  pystate_cls.def("get_float64_or_default",
                  &nvbench::state::get_float64_or_default,
                  py::arg("name"),
                  py::pos_only{},
                  py::arg("default_value"));

  pystate_cls.def("get_string", &nvbench::state::get_string, py::arg("name"));
  pystate_cls.def("get_string_or_default",
                  &nvbench::state::get_string_or_default,
                  py::arg("name"),
                  py::pos_only{},
                  py::arg("default_value"));

  pystate_cls.def("add_element_count",
                  &nvbench::state::add_element_count,
                  py::arg("count"),
                  py::arg("column_name") = py::str(""));
  pystate_cls.def("set_element_count", &nvbench::state::set_element_count, py::arg("count"));
  pystate_cls.def("get_element_count", &nvbench::state::get_element_count);

  pystate_cls.def("skip", &nvbench::state::skip, py::arg("reason"));
  pystate_cls.def("is_skipped", &nvbench::state::is_skipped);
  pystate_cls.def("get_skip_reason", &nvbench::state::get_skip_reason);

  pystate_cls.def(
    "add_global_memory_reads",
    [](nvbench::state &state, std::size_t nbytes, const std::string &column_name) -> void {
      state.add_global_memory_reads(nbytes, column_name);
    },
    "Add size, in bytes, of global memory reads",
    py::arg("nbytes"),
    py::pos_only{},
    py::arg("column_name") = py::str(""));
  pystate_cls.def(
    "add_global_memory_writes",
    [](nvbench::state &state, std::size_t nbytes, const std::string &column_name) -> void {
      state.add_global_memory_writes(nbytes, column_name);
    },
    "Add size, in bytes, of global memory writes",
    py::arg("nbytes"),
    py::pos_only{},
    py::arg("column_name") = py::str(""));
  pystate_cls.def(
    "get_benchmark",
    [](const nvbench::state &state) { return std::ref(state.get_benchmark()); },
    py::return_value_policy::reference);
  pystate_cls.def("get_throttle_threshold", &nvbench::state::get_throttle_threshold);
  pystate_cls.def("set_throttle_threshold",
                  &nvbench::state::set_throttle_threshold,
                  py::arg("throttle_fraction"));

  pystate_cls.def("get_min_samples", &nvbench::state::get_min_samples);
  pystate_cls.def("set_min_samples",
                  &nvbench::state::set_min_samples,
                  py::arg("min_samples_count"));

  pystate_cls.def("get_disable_blocking_kernel", &nvbench::state::get_disable_blocking_kernel);
  pystate_cls.def("set_disable_blocking_kernel",
                  &nvbench::state::set_disable_blocking_kernel,
                  py::arg("disable_blocking_kernel"));

  pystate_cls.def("get_run_once", &nvbench::state::get_run_once);
  pystate_cls.def("set_run_once", &nvbench::state::set_run_once, py::arg("run_once"));

  pystate_cls.def("get_timeout", &nvbench::state::get_timeout);
  pystate_cls.def("set_timeout", &nvbench::state::set_timeout, py::arg("duration"));

  pystate_cls.def("get_blocking_kernel_timeout", &nvbench::state::get_blocking_kernel_timeout);
  pystate_cls.def("set_blocking_kernel_timeout",
                  &nvbench::state::set_blocking_kernel_timeout,
                  py::arg("duration"));

  pystate_cls.def("collect_cupti_metrics", &nvbench::state::collect_cupti_metrics);
  pystate_cls.def("is_cupti_required", &nvbench::state::is_cupti_required);

  pystate_cls.def(
    "exec",
    [](nvbench::state &state, py::object py_launcher_fn, bool batched, bool sync) {
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
    },
    "Executor for given launcher callable fn(state : Launch)",
    py::arg("launcher_fn"),
    py::pos_only{},
    py::arg("batched") = true,
    py::arg("sync")    = false);

  pystate_cls.def("get_short_description",
                  [](const nvbench::state &state) { return state.get_short_description(); });

  pystate_cls.def(
    "add_summary",
    [](nvbench::state &state, std::string column_name, std::string value) {
      auto &summ = state.add_summary("nv/python/" + column_name);
      summ.set_string("description", "User tag: " + column_name);
      summ.set_string("name", std::move(column_name));
      summ.set_string("value", std::move(value));
    },
    py::arg("name"),
    py::arg("value"));
  pystate_cls.def(
    "add_summary",
    [](nvbench::state &state, std::string column_name, std::int64_t value) {
      auto &summ = state.add_summary("nv/python/" + column_name);
      summ.set_string("description", "User tag: " + column_name);
      summ.set_string("name", std::move(column_name));
      summ.set_int64("value", value);
    },
    py::arg("name"),
    py::arg("value"));
  pystate_cls.def(
    "add_summary",
    [](nvbench::state &state, std::string column_name, double value) {
      auto &summ = state.add_summary("nv/python/" + column_name);
      summ.set_string("description", "User tag: " + column_name);
      summ.set_string("name", std::move(column_name));
      summ.set_float64("value", value);
    },
    py::arg("name"),
    py::arg("value"));
  pystate_cls.def("get_axis_values_as_string",
                  [](const nvbench::state &state) { return state.get_axis_values_as_string(); });
  pystate_cls.def("get_axis_values", &py_get_axis_values);
  pystate_cls.def("get_stopping_criterion", &nvbench::state::get_stopping_criterion);

  // Use handle to take a memory leak here, since this object's destructor may be called after
  // interpreter has shut down
  benchmark_exc =
    py::exception<nvbench_run_error>(m, "NVBenchRuntimeError", PyExc_RuntimeError).release();
  // == STEP 6
  //    ATTN: nvbench::benchmark_manager is a singleton

  global_registry =
    std::unique_ptr<GlobalBenchmarkRegistry, py::nodelete>(new GlobalBenchmarkRegistry(),
                                                           py::nodelete{});

  m.def(
    "register",
    [&](py::object fn) { return std::ref(global_registry->add_bench(fn)); },
    "Register benchmark function of type Callable[[nvbench.State], None]",
    py::return_value_policy::reference,
    py::arg("benchmark_fn"));

  m.def(
    "run_all_benchmarks",
    [&](py::object argv) -> void {
      if (!py::isinstance<py::list>(argv))
      {
        throw py::type_error("run_all_benchmarks expects a list of command-line arguments");
      }
      std::vector<std::string> args = py::cast<std::vector<std::string>>(argv);
      global_registry->run(args);
    },
    "Run all registered benchmarks",
    py::arg("argv") = py::list());

  m.def("test_cpp_exception", []() { throw nvbench_run_error("Test"); });
  m.def("test_py_exception", []() {
    py::set_error(benchmark_exc, "Test");
    throw py::error_already_set();
  });
}
