import sys
import time

import cuda.nvbench as nvbench


def sleep_bench(state: nvbench.State) -> None:
    def launcher(launch: nvbench.Launch):
        time.sleep(1)

    state.exec(launcher)


def sleep_bench_sync(state: nvbench.State) -> None:
    sync = state.get_string("Sync")
    sync_flag = sync == "Do sync"

    def launcher(launch: nvbench.Launch):
        time.sleep(1)

    state.exec(launcher, sync=sync_flag)


if __name__ == "__main__":
    # time function sleeping on the host
    # using CPU timer only
    b = nvbench.register(sleep_bench)
    b.set_is_cpu_only(True)

    # time the same function using both CPU/GPU timers
    b2 = nvbench.register(sleep_bench_sync)
    b2.add_string_axis("Sync", ["Do not sync", "Do sync"])

    nvbench.run_all_benchmarks(sys.argv)
