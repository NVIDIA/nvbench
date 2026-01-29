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
