import importlib.metadata

from cuda.bindings.path_finder import (  # type: ignore[import-not-found]
    _load_nvidia_dynamic_library,
)

try:
    __version__ = importlib.metadata.version("pynvbench")
except Exception:
    __version__ = "0.0.0dev"

for libname in ("cupti", "nvperf_target", "nvperf_host"):
    _load_nvidia_dynamic_library(libname)

from ._nvbench import *  # noqa: E402, F403
from ._nvbench import register, run_all_benchmarks  # noqa: E402

__all__ = ["register", "run_all_benchmarks"]
