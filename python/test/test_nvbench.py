import json

import cuda.nvbench as nvbench
import pytest


def test_cpp_exception():
    with pytest.raises(RuntimeError, match="Test"):
        nvbench._nvbench.test_cpp_exception()


def test_py_exception():
    with pytest.raises(nvbench.NVBenchRuntimeError, match="Test"):
        nvbench._nvbench.test_py_exception()


@pytest.mark.parametrize(
    "cls", [nvbench.CudaStream, nvbench.State, nvbench.Launch, nvbench.Benchmark]
)
def test_api_ctor(cls):
    with pytest.raises(TypeError, match="No constructor defined!"):
        cls()


def t_bench(state: nvbench.State):
    s = {"a": 1, "b": 0.5, "c": "test", "d": {"a": 1}}

    def launcher(launch: nvbench.Launch):
        for _ in range(10000):
            _ = json.dumps(s)

    state.exec(launcher)


def test_cpu_only():
    b = nvbench.register(t_bench)
    b.set_is_cpu_only(True)

    nvbench.run_all_benchmarks(["-q", "--profile"])
