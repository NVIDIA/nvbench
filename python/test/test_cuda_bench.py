# Copyright 2026 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 with the LLVM exception
#  (the "License"); you may not use this file except in compliance with
#  the License.
#
#  You may obtain a copy of the License at
#
#      http://llvm.org/foundation/relicensing/LICENSE.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
from typing import Union

import cuda.bench as bench
import pytest


def test_cpp_exception():
    with pytest.raises(RuntimeError, match="Test"):
        bench._nvbench._test_cpp_exception()


def test_py_exception():
    with pytest.raises(bench.NVBenchRuntimeError, match="Test"):
        bench._nvbench._test_py_exception()


@pytest.mark.parametrize(
    "cls", [bench.CudaStream, bench.State, bench.Launch, bench.Benchmark]
)
def test_api_ctor(cls):
    with pytest.raises(TypeError, match="No constructor defined!"):
        cls()


def t_bench(state: bench.State):
    s = {"a": 1, "b": 0.5, "c": "test", "d": {"a": 1}}

    def launcher(launch: bench.Launch):
        for _ in range(10000):
            _ = json.dumps(s)

    state.exec(launcher)


def test_cpu_only():
    b = bench.register(t_bench)
    b.set_is_cpu_only(True)

    bench.run_all_benchmarks(["-q", "--profile"])


def docstring_check(doc_str: Union[str, None]) -> None:
    assert isinstance(doc_str, str)
    assert len(doc_str) > 0


def obj_has_docstring_check(o: object) -> None:
    docstring_check(o.__doc__)


def test_module_doc():
    obj_has_docstring_check(bench)


def test_register_doc():
    obj_has_docstring_check(bench.register)


def test_decorator_docstrings():
    obj_has_docstring_check(bench.axis)
    obj_has_docstring_check(bench.axis.int64)
    obj_has_docstring_check(bench.axis.add_int64_axis)
    obj_has_docstring_check(bench.axis.int64_power_of_two)
    obj_has_docstring_check(bench.axis.power_of_two)
    obj_has_docstring_check(bench.axis.add_int64_power_of_two_axis)
    obj_has_docstring_check(bench.axis.float64)
    obj_has_docstring_check(bench.axis.add_float64_axis)
    obj_has_docstring_check(bench.axis.string)
    obj_has_docstring_check(bench.axis.add_string_axis)

    obj_has_docstring_check(bench.option)
    obj_has_docstring_check(bench.option.name)
    obj_has_docstring_check(bench.option.set_name)
    obj_has_docstring_check(bench.option.run_once)
    obj_has_docstring_check(bench.option.set_run_once)
    obj_has_docstring_check(bench.option.skip_time)
    obj_has_docstring_check(bench.option.set_skip_time)
    obj_has_docstring_check(bench.option.throttle_recovery_delay)
    obj_has_docstring_check(bench.option.set_throttle_recovery_delay)
    obj_has_docstring_check(bench.option.throttle_threshold)
    obj_has_docstring_check(bench.option.set_throttle_threshold)
    obj_has_docstring_check(bench.option.timeout)
    obj_has_docstring_check(bench.option.set_timeout)
    obj_has_docstring_check(bench.option.stopping_criterion)
    obj_has_docstring_check(bench.option.set_stopping_criterion)
    obj_has_docstring_check(bench.option.criterion_param_float64)
    obj_has_docstring_check(bench.option.set_criterion_param_float64)
    obj_has_docstring_check(bench.option.criterion_param_int64)
    obj_has_docstring_check(bench.option.set_criterion_param_int64)
    obj_has_docstring_check(bench.option.criterion_param_string)
    obj_has_docstring_check(bench.option.set_criterion_param_string)
    obj_has_docstring_check(bench.option.min_samples)
    obj_has_docstring_check(bench.option.set_min_samples)
    obj_has_docstring_check(bench.option.is_cpu_only)
    obj_has_docstring_check(bench.option.set_is_cpu_only)


def test_register_decorator_preserves_function_and_applies_options(monkeypatch):
    class FakeBenchmark:
        def __init__(self):
            self.calls = []

        def add_int64_axis(self, name, values):
            self.calls.append(("int64", name, list(values)))
            return self

        def set_min_samples(self, count):
            self.calls.append(("min_samples", count))
            return self

    fake_benchmark = FakeBenchmark()
    registered_functions = []

    def fake_register(fn):
        registered_functions.append(fn)
        return fake_benchmark

    monkeypatch.setattr(bench, "_register", fake_register)

    @bench.register()
    @bench.axis.int64("Elements", [1, 2, 3])
    @bench.option.min_samples(11)
    def decorated(state: bench.State):
        pass

    assert registered_functions == [decorated]
    assert fake_benchmark.calls == [
        ("int64", "Elements", [1, 2, 3]),
        ("min_samples", 11),
    ]
    assert callable(decorated)


def test_register_function_form_applies_decorated_options(monkeypatch):
    class FakeBenchmark:
        def __init__(self):
            self.calls = []

        def add_float64_axis(self, name, values):
            self.calls.append(("float64", name, list(values)))
            return self

    fake_benchmark = FakeBenchmark()

    def fake_register(fn):
        return fake_benchmark

    monkeypatch.setattr(bench, "_register", fake_register)

    @bench.axis.float64("Duration", [0.1, 0.2])
    def decorated(state: bench.State):
        pass

    assert bench.register(decorated) is fake_benchmark
    assert fake_benchmark.calls == [("float64", "Duration", [0.1, 0.2])]


def test_option_decorators_reject_wrong_order(monkeypatch):
    class FakeBenchmark:
        pass

    def fake_register(fn):
        return FakeBenchmark()

    monkeypatch.setattr(bench, "_register", fake_register)

    @bench.register()
    def decorated(state: bench.State):
        pass

    with pytest.raises(RuntimeError, match="must be placed below"):
        bench.option.min_samples(3)(decorated)


def test_axis_decorators_reject_wrong_order(monkeypatch):
    class FakeBenchmark:
        pass

    def fake_register(fn):
        return FakeBenchmark()

    monkeypatch.setattr(bench, "_register", fake_register)

    @bench.register()
    def decorated(state: bench.State):
        pass

    with pytest.raises(RuntimeError, match="must be placed below"):
        bench.axis.int64("Elements", [1, 2, 3])(decorated)


def test_run_all_benchmarks_doc():
    obj_has_docstring_check(bench.run_all_benchmarks)


def test_State_doc():
    cl = bench.State
    obj_has_docstring_check(cl)
    obj_has_docstring_check(cl.exec)
    obj_has_docstring_check(cl.get_int64)
    obj_has_docstring_check(cl.get_float64)
    obj_has_docstring_check(cl.get_string)
    obj_has_docstring_check(cl.skip)


def test_Launch_doc():
    cl = bench.Launch
    obj_has_docstring_check(cl)
    obj_has_docstring_check(cl.get_stream)


def test_CudaStream_doc():
    cl = bench.CudaStream
    obj_has_docstring_check(cl)


def test_Benchmark_doc():
    cl = bench.Benchmark
    obj_has_docstring_check(cl)
    obj_has_docstring_check(cl.add_int64_axis)
    obj_has_docstring_check(cl.add_int64_power_of_two_axis)
    obj_has_docstring_check(cl.add_float64_axis)
    obj_has_docstring_check(cl.add_string_axis)
