import sys

import cuda.bench as bench
import torch


def as_torch_cuda_Stream(
    cs: bench.CudaStream, dev: int | None
) -> torch.cuda.ExternalStream:
    return torch.cuda.ExternalStream(
        stream_ptr=cs.addressof(), device=torch.cuda.device(dev)
    )


def torch_bench(state: bench.State) -> None:
    state.set_throttle_threshold(0.25)

    dev_id = state.get_device()
    tc_s = as_torch_cuda_Stream(state.get_stream(), dev_id)

    dt = torch.float32
    scalar_shape: tuple = tuple()
    n = 2**28
    with torch.cuda.stream(tc_s):
        a3 = torch.randn(scalar_shape, dtype=dt)
        a2 = torch.randn(scalar_shape, dtype=dt)
        a1 = torch.randn(scalar_shape, dtype=dt)
        a0 = torch.randn(scalar_shape, dtype=dt)
        x = torch.linspace(-3, 3, n, dtype=dt)
        y = torch.sin(x)

    learning_rate = 1e-4

    def launcher(launch: bench.Launch) -> None:
        tc_s = as_torch_cuda_Stream(launch.get_stream(), dev_id)
        with torch.cuda.stream(tc_s):
            x2 = torch.square(x)
            y_pred = (a3 + x2 * a1) + x * (a2 + a0 * x2)

            _ = torch.square(y_pred - y).sum()
            grad_y_pred = 2 * (y_pred - y)
            grad_a3 = grad_y_pred.sum()
            grad_a2 = (grad_y_pred * x).sum()
            grad_a1 = (grad_y_pred * x2).sum()
            grad_a0 = (grad_y_pred * x2 * x).sum()

            _ = a3 - grad_a3 * learning_rate
            _ = a2 - grad_a2 * learning_rate
            _ = a1 - grad_a1 * learning_rate
            _ = a0 - grad_a0 * learning_rate

    state.exec(launcher, sync=True)


if __name__ == "__main__":
    bench.register(torch_bench)

    bench.run_all_benchmarks(sys.argv)
