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

from ._nvbench import (  # noqa: E402
    Benchmark as Benchmark,
)
from ._nvbench import (  # noqa: E402
    CudaStream as CudaStream,
)
from ._nvbench import (  # noqa: E402
    Launch as Launch,
)
from ._nvbench import (  # noqa: E402
    State as State,
)
from ._nvbench import (  # noqa: E402
    register as register,
)
from ._nvbench import (  # noqa: E402
    run_all_benchmarks as run_all_benchmarks,
)

del load_nvidia_dynamic_lib
