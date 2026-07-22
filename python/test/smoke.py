# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

import cuda.bench as bench


@bench.register()
@bench.option.is_cpu_only()
@bench.option.stopping_criterion("sample-count")
@bench.option.criterion_param_int64("target-samples", 10)
def smoke(state: bench.State) -> None:
    state.exec(lambda launch: None)


if __name__ == "__main__":
    bench.run_all_benchmarks(sys.argv)
