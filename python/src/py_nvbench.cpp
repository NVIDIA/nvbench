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

inline void set_env(const char *name, const char *value)
{
#ifdef _MSC_VER
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

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

  benchmark_wrapper_t()
      : m_fn() {};
  explicit benchmark_wrapper_t(py::object o)
      : m_fn{std::shared_ptr<py::object>(new py::object(o), PyObjectDeleter{})}
  {}

  benchmark_wrapper_t(const benchmark_wrapper_t &other)
      : m_fn{other.m_fn}
  {}
  benchmark_wrapper_t &operator=(const benchmark_wrapper_t &other) = delete;
  benchmark_wrapper_t(benchmark_wrapper_t &&) noexcept             = delete;
  benchmark_wrapper_t &operator=(benchmark_wrapper_t &&) noexcept  = delete;

  void operator()(nvbench::state &state, nvbench::type_list<>)
  {
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

class nvbench_run_error : std::runtime_error
{};
constinit py::handle benchmark_exc{};

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
      std::stringstream ss;
      ss << "Caught exception while running benchmarks: ";
      ss << e.what();

      const std::string &exc_message = ss.str();
      py::set_error(benchmark_exc, exc_message.c_str());
    }
    catch (...)
    {
      py::set_error(benchmark_exc, "Caught unknown exception in nvbench_main");
    }
  }
};

// essentially a global variable, but allocated on the heap during module initialization
constinit std::unique_ptr<GlobalBenchmarkRegistry, py::nodelete> global_registry{};

} // end of anonymous namespace

PYBIND11_MODULE(_nvbench, m)
{
  // == STEP 1
  // Set environment variable CUDA_MODULE_LOADING=EAGER

  // See NVIDIA/NVBench#136 for CUDA_MODULE_LOADING
  set_env("CUDA_MODULE_LOADING", "EAGER");

  NVBENCH_DRIVER_API_CALL(cuInit(0));

  // This line ensures that benchmark_manager has been created during module init
  // It is reinitialized before running all benchmarks to set devices to use
  nvbench::benchmark_manager::get().initialize();

  // TODO: Use cuModuleGetLoadingMode(&mode) to confirm that (mode == CU_MODULE_EAGER_LOADING)
  // and issue warning otherwise

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
    "getStream",
    [](nvbench::launch &launch) { return std::ref(launch.get_stream()); },
    py::return_value_policy::reference);

  // == STEP 4
  // Define Benchmark class

  auto py_benchmark_cls = py::class_<nvbench::benchmark_base>(m, "Benchmark");
  py_benchmark_cls.def("getName", &nvbench::benchmark_base::get_name);
  py_benchmark_cls.def(
    "addInt64Axis",
    [](nvbench::benchmark_base &self, std::string name, const std::vector<nvbench::int64_t> &data) {
      self.add_int64_axis(std::move(name), data);
      return std::ref(self);
    },
    py::return_value_policy::reference);
  py_benchmark_cls.def(
    "addFloat64Axis",
    [](nvbench::benchmark_base &self,
       std::string name,
       const std::vector<nvbench::float64_t> &data) {
      self.add_float64_axis(std::move(name), data);
      return std::ref(self);
    },
    py::return_value_policy::reference);
  py_benchmark_cls.def(
    "addStringAxis",
    [](nvbench::benchmark_base &self, std::string name, const std::vector<std::string> &data) {
      self.add_string_axis(std::move(name), data);
      return std::ref(self);
    },
    py::return_value_policy::reference);
  py_benchmark_cls.def(
    "setName",
    [](nvbench::benchmark_base &self, std::string name) {
      self.set_name(std::move(name));
      return std::ref(self);
    },
    py::return_value_policy::reference);

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

  pystate_cls.def("hasDevice", [](const nvbench::state &state) -> bool {
    return static_cast<bool>(state.get_device());
  });
  pystate_cls.def("hasPrinters", [](const nvbench::state &state) -> bool {
    return state.get_benchmark().get_printer().has_value();
  });
  pystate_cls.def("getDevice", [](const nvbench::state &state) {
    auto dev = state.get_device();
    if (dev.has_value())
    {
      return py::cast(dev.value().get_id());
    }
    return py::object(py::none());
  });

  pystate_cls.def(
    "getStream",
    [](nvbench::state &state) { return std::ref(state.get_cuda_stream()); },
    py::return_value_policy::reference);

  pystate_cls.def("getInt64", &nvbench::state::get_int64);
  pystate_cls.def("getInt64", &nvbench::state::get_int64_or_default);

  pystate_cls.def("getFloat64", &nvbench::state::get_float64);
  pystate_cls.def("getFloat64", &nvbench::state::get_float64_or_default);

  pystate_cls.def("getString", &nvbench::state::get_string);
  pystate_cls.def("getString", &nvbench::state::get_string_or_default);

  pystate_cls.def("addElementCount",
                  &nvbench::state::add_element_count,
                  py::arg("count"),
                  py::arg("column_name") = py::str(""));
  pystate_cls.def("setElementCount", &nvbench::state::set_element_count);
  pystate_cls.def("getElementCount", &nvbench::state::get_element_count);

  pystate_cls.def("skip", &nvbench::state::skip);
  pystate_cls.def("isSkipped", &nvbench::state::is_skipped);
  pystate_cls.def("getSkipReason", &nvbench::state::get_skip_reason);

  pystate_cls.def(
    "addGlobalMemoryReads",
    [](nvbench::state &state, std::size_t nbytes, const std::string &column_name) -> void {
      state.add_global_memory_reads(nbytes, column_name);
    },
    "Add size, in bytes, of global memory reads",
    py::arg("nbytes"),
    py::pos_only{},
    py::arg("column_name") = py::str(""));
  pystate_cls.def(
    "addGlobalMemoryWrites",
    [](nvbench::state &state, std::size_t nbytes, const std::string &column_name) -> void {
      state.add_global_memory_writes(nbytes, column_name);
    },
    "Add size, in bytes, of global memory writes",
    py::arg("nbytes"),
    py::pos_only{},
    py::arg("column_name") = py::str(""));
  pystate_cls.def(
    "getBenchmark",
    [](const nvbench::state &state) { return std::ref(state.get_benchmark()); },
    py::return_value_policy::reference);
  pystate_cls.def("getThrottleThreshold", &nvbench::state::get_throttle_threshold);

  pystate_cls.def("getMinSamples", &nvbench::state::get_min_samples);
  pystate_cls.def("setMinSamples", &nvbench::state::set_min_samples);

  pystate_cls.def("getDisableBlockingKernel", &nvbench::state::get_disable_blocking_kernel);
  pystate_cls.def("setDisableBlockingKernel", &nvbench::state::set_disable_blocking_kernel);

  pystate_cls.def("getRunOnce", &nvbench::state::get_run_once);
  pystate_cls.def("setRunOnce", &nvbench::state::set_run_once);

  pystate_cls.def("getTimeout", &nvbench::state::get_timeout);
  pystate_cls.def("setTimeout", &nvbench::state::set_timeout);

  pystate_cls.def("getBlockingKernelTimeout", &nvbench::state::get_blocking_kernel_timeout);
  pystate_cls.def("setBlockingKernelTimeout", &nvbench::state::set_blocking_kernel_timeout);

  pystate_cls.def("collectCUPTIMetrics", &nvbench::state::collect_cupti_metrics);
  pystate_cls.def("isCUPTIRequired", &nvbench::state::is_cupti_required);

  pystate_cls.def(
    "exec",
    [](nvbench::state &state, py::object fn, bool batched, bool sync) {
      auto launcher_fn = [fn](nvbench::launch &launch_descr) -> void {
        fn(py::cast(std::ref(launch_descr), py::return_value_policy::reference));
      };

      if (sync)
      {
        if (batched)
        {
          state.exec(nvbench::exec_tag::sync, launcher_fn);
        }
        else
        {
          state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::no_batch, launcher_fn);
        }
      }
      else
      {
        if (batched)
        {
          state.exec(nvbench::exec_tag::none, launcher_fn);
        }
        else
        {
          state.exec(nvbench::exec_tag::no_batch, launcher_fn);
        }
      }
    },
    "Executor for given callable fn(state : Launch)",
    py::arg("fn"),
    py::pos_only{},
    py::arg("batched") = true,
    py::arg("sync")    = false);

  pystate_cls.def("get_short_description",
                  [](const nvbench::state &state) { return state.get_short_description(); });

  pystate_cls.def("add_summary",
                  [](nvbench::state &state, std::string column_name, std::string value) {
                    auto &summ = state.add_summary("nv/python/" + column_name);
                    summ.set_string("description", "User tag: " + column_name);
                    summ.set_string("name", std::move(column_name));
                    summ.set_string("value", std::move(value));
                  });
  pystate_cls.def("add_summary",
                  [](nvbench::state &state, std::string column_name, std::int64_t value) {
                    auto &summ = state.add_summary("nv/python/" + column_name);
                    summ.set_string("description", "User tag: " + column_name);
                    summ.set_string("name", std::move(column_name));
                    summ.set_int64("value", value);
                  });
  pystate_cls.def("add_summary", [](nvbench::state &state, std::string column_name, double value) {
    auto &summ = state.add_summary("nv/python/" + column_name);
    summ.set_string("description", "User tag: " + column_name);
    summ.set_string("name", std::move(column_name));
    summ.set_float64("value", value);
  });

  // Use handle to take a memory leak here, since this object's destructor may be called after
  // interpreter has shut down
  benchmark_exc =
    py::exception<nvbench_run_error>(m, "NVBenchRuntimeException", PyExc_RuntimeError).release();
  // == STEP 6
  //    ATTN: nvbench::benchmark_manager is a singleton

  global_registry =
    std::unique_ptr<GlobalBenchmarkRegistry, py::nodelete>(new GlobalBenchmarkRegistry(),
                                                           py::nodelete{});

  m.def(
    "register",
    [&](py::object fn) { return std::ref(global_registry->add_bench(fn)); },
    py::return_value_policy::reference);

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
    "Run all benchmarks",
    py::arg("argv") = py::list());
}
