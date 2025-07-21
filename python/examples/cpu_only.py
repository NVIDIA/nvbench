import sys
import time

import cuda.nvbench as nvbench


def throughput_bench(state: nvbench.State) -> None:
    def launcher(launch: nvbench.Launch):
        time.sleep(1)

    state.exec(launcher)


if __name__ == "__main__":
    b = nvbench.register(throughput_bench)
    b.setIsCPUOnly(True)

    nvbench.run_all_benchmarks(sys.argv)
