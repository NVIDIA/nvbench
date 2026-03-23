import json

import cuda.bench as bench
import pytest


def test_cpp_exception():
    with pytest.raises(RuntimeError, match="Test"):
        bench._nvbench.test_cpp_exception()


def test_py_exception():
    with pytest.raises(bench.NVBenchRuntimeError, match="Test"):
        bench._nvbench.test_py_exception()


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


def docstring_check(doc_str: str) -> None:
    assert isinstance(doc_str, str)
    assert len(doc_str) > 0


def obj_has_docstring_check(o: object) -> None:
    docstring_check(o.__doc__)


def test_module_doc():
    obj_has_docstring_check(bench)


def test_register_doc():
    obj_has_docstring_check(bench.register)


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
