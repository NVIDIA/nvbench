# Copyright 2025 NVIDIA Corporation
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

import importlib.metadata
import warnings

from cuda.pathfinder import (  # type: ignore[import-not-found]
    load_nvidia_dynamic_lib,
)

try:
    __version__ = importlib.metadata.version("pynvbench")
except Exception as e:
    __version__ = "0.0.0dev"
    warnings.warn(
        "Could not retrieve version of pynvbench package dynamically from its metadata. "
        f"Exception {e} was raised. "
        f"Version is set to fall-back value '{__version__}' instead."
    )

for libname in ("cupti", "nvperf_target", "nvperf_host"):
    load_nvidia_dynamic_lib(libname)

from cuda.bench._nvbench import (  # noqa: E402
    Benchmark as Benchmark,
)
from cuda.bench._nvbench import (  # noqa: E402
    CudaStream as CudaStream,
)
from cuda.bench._nvbench import (  # noqa: E402
    Launch as Launch,
)
from cuda.bench._nvbench import (  # noqa: E402
    NVBenchRuntimeError as NVBenchRuntimeError,
)
from cuda.bench._nvbench import (  # noqa: E402
    State as State,
)
from cuda.bench._nvbench import (  # noqa: E402
    register as register,
)
from cuda.bench._nvbench import (  # noqa: E402
    run_all_benchmarks as run_all_benchmarks,
)

del load_nvidia_dynamic_lib
