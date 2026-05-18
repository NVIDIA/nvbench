# Copyright 2026 NVIDIA Corporation
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

"""Function decorators for registering and configuring NVBench benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])
_Benchmark = Any
_RawRegister = Callable[[Callable[..., Any]], _Benchmark]
_RegisterGetter = Callable[[], _RawRegister]
_BenchmarkAction = Callable[[_Benchmark], None]
_BENCHMARK_ACTIONS_ATTR = "__cuda_bench_actions__"
_BENCHMARK_REGISTERED_ATTR = "__cuda_bench_registered__"


def _append_benchmark_action(action: _BenchmarkAction) -> Callable[[_F], _F]:
    """Return a function-preserving decorator that records a benchmark action."""

    def decorator(fn: _F) -> _F:
        """Attach a delayed benchmark action to a benchmark function."""
        if getattr(fn, _BENCHMARK_REGISTERED_ATTR, False):
            raise RuntimeError(
                "NVBench axis and option decorators must be placed below "
                "@bench.register()"
            )

        actions = getattr(fn, _BENCHMARK_ACTIONS_ATTR, None)
        if actions is None:
            actions = []
            try:
                setattr(fn, _BENCHMARK_ACTIONS_ATTR, actions)
            except AttributeError as e:
                raise TypeError(
                    "NVBench benchmark decorators require an object that "
                    "supports attribute assignment"
                ) from e

        actions.append(action)
        return fn

    return decorator


def _apply_benchmark_actions(
    benchmark: _Benchmark, fn: Callable[..., Any]
) -> _Benchmark:
    """Apply delayed benchmark actions to a registered benchmark.

    Python invokes stacked decorators from the function outward, so actions are
    recorded opposite their source order. Apply them as written below
    ``@bench.register()``.
    """
    for action in reversed(getattr(fn, _BENCHMARK_ACTIONS_ATTR, ())):
        action(benchmark)

    return benchmark


def _mark_registered(fn: Callable[..., Any]) -> None:
    """Mark a callable as registered when it supports attribute assignment."""
    try:
        setattr(fn, _BENCHMARK_REGISTERED_ATTR, True)
    except AttributeError:
        pass


def make_register(get_register: _RegisterGetter) -> Callable[..., Any]:
    """Create the public ``register`` function around a raw register function."""

    def register(fn=None, /):
        """Register a Python benchmark function with NVBench.

        Called as ``bench.register(fn)``, this returns the registered
        ``Benchmark``. Called as ``@bench.register()``, this returns a decorator
        that registers the function and leaves the decorated symbol unchanged.

        When used as a decorator, ``@bench.register()`` must be the top-most
        decorator. Axis and option decorators are lazy; they only record how to
        configure the benchmark. ``@bench.register()`` consumes those recorded
        actions and registers the function with NVBench.

        Example
        -------

        .. code-block:: python

            @bench.register()
            @bench.axis.int64("Elements", [1024, 2048])
            @bench.option.min_samples(10)
            def copy(state: bench.State):
                ...
        """
        if fn is None:

            def decorator(benchmark_fn):
                benchmark = get_register()(benchmark_fn)
                _apply_benchmark_actions(benchmark, benchmark_fn)
                _mark_registered(benchmark_fn)
                return benchmark_fn

            return decorator

        benchmark = get_register()(fn)
        _apply_benchmark_actions(benchmark, fn)
        _mark_registered(fn)
        return benchmark

    register.__name__ = "register"
    register.__qualname__ = "register"
    return register


class _AxisDecorators:
    """Namespace for decorators that add axes to a benchmark."""

    def int64(self, name: str, values: Sequence[int]) -> Callable[[_F], _F]:
        """Add an ``int64`` axis to the decorated benchmark."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.add_int64_axis(name, values)
        )

    def add_int64_axis(self, name: str, values: Sequence[int]) -> Callable[[_F], _F]:
        """Alias for :meth:`int64`."""
        return self.int64(name, values)

    def int64_power_of_two(
        self, name: str, values: Sequence[int]
    ) -> Callable[[_F], _F]:
        """Add a power-of-two ``int64`` axis to the decorated benchmark."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.add_int64_power_of_two_axis(name, values)
        )

    def power_of_two(self, name: str, values: Sequence[int]) -> Callable[[_F], _F]:
        """Alias for :meth:`int64_power_of_two`."""
        return self.int64_power_of_two(name, values)

    def add_int64_power_of_two_axis(
        self, name: str, values: Sequence[int]
    ) -> Callable[[_F], _F]:
        """Alias for :meth:`int64_power_of_two`."""
        return self.int64_power_of_two(name, values)

    def float64(self, name: str, values: Sequence[float]) -> Callable[[_F], _F]:
        """Add a ``float64`` axis to the decorated benchmark."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.add_float64_axis(name, values)
        )

    def add_float64_axis(
        self, name: str, values: Sequence[float]
    ) -> Callable[[_F], _F]:
        """Alias for :meth:`float64`."""
        return self.float64(name, values)

    def string(self, name: str, values: Sequence[str]) -> Callable[[_F], _F]:
        """Add a string axis to the decorated benchmark."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.add_string_axis(name, values)
        )

    def add_string_axis(self, name: str, values: Sequence[str]) -> Callable[[_F], _F]:
        """Alias for :meth:`string`."""
        return self.string(name, values)


class _OptionDecorators:
    """Namespace for decorators that set benchmark options."""

    def name(self, value: str) -> Callable[[_F], _F]:
        """Set the benchmark name."""
        return self.set_name(value)

    def set_name(self, value: str) -> Callable[[_F], _F]:
        """Set the benchmark name."""
        return _append_benchmark_action(lambda benchmark: benchmark.set_name(value))

    def run_once(self, value: bool = True) -> Callable[[_F], _F]:
        """Set whether each benchmark configuration runs only once."""
        return self.set_run_once(value)

    def set_run_once(self, value: bool) -> Callable[[_F], _F]:
        """Set whether each benchmark configuration runs only once."""
        return _append_benchmark_action(lambda benchmark: benchmark.set_run_once(value))

    def skip_time(self, duration_seconds: float) -> Callable[[_F], _F]:
        """Set the threshold below which benchmark runs are skipped."""
        return self.set_skip_time(duration_seconds)

    def set_skip_time(self, duration_seconds: float) -> Callable[[_F], _F]:
        """Set the threshold below which benchmark runs are skipped."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.set_skip_time(duration_seconds)
        )

    def throttle_recovery_delay(self, delay_seconds: float) -> Callable[[_F], _F]:
        """Set the delay after GPU clock throttling is detected."""
        return self.set_throttle_recovery_delay(delay_seconds)

    def set_throttle_recovery_delay(self, delay_seconds: float) -> Callable[[_F], _F]:
        """Set the delay after GPU clock throttling is detected."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.set_throttle_recovery_delay(delay_seconds)
        )

    def throttle_threshold(self, threshold: float) -> Callable[[_F], _F]:
        """Set the GPU clock throttle threshold."""
        return self.set_throttle_threshold(threshold)

    def set_throttle_threshold(self, threshold: float) -> Callable[[_F], _F]:
        """Set the GPU clock throttle threshold."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.set_throttle_threshold(threshold)
        )

    def timeout(self, duration_seconds: float) -> Callable[[_F], _F]:
        """Set the benchmark timeout in seconds."""
        return self.set_timeout(duration_seconds)

    def set_timeout(self, duration_seconds: float) -> Callable[[_F], _F]:
        """Set the benchmark timeout in seconds."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.set_timeout(duration_seconds)
        )

    def stopping_criterion(self, criterion: str) -> Callable[[_F], _F]:
        """Set the benchmark stopping criterion."""
        return self.set_stopping_criterion(criterion)

    def set_stopping_criterion(self, criterion: str) -> Callable[[_F], _F]:
        """Set the benchmark stopping criterion."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.set_stopping_criterion(criterion)
        )

    def criterion_param_float64(self, name: str, value: float) -> Callable[[_F], _F]:
        """Set a floating-point parameter for the stopping criterion."""
        return self.set_criterion_param_float64(name, value)

    def set_criterion_param_float64(
        self, name: str, value: float
    ) -> Callable[[_F], _F]:
        """Set a floating-point parameter for the stopping criterion."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.set_criterion_param_float64(name, value)
        )

    def criterion_param_int64(self, name: str, value: int) -> Callable[[_F], _F]:
        """Set an integer parameter for the stopping criterion."""
        return self.set_criterion_param_int64(name, value)

    def set_criterion_param_int64(self, name: str, value: int) -> Callable[[_F], _F]:
        """Set an integer parameter for the stopping criterion."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.set_criterion_param_int64(name, value)
        )

    def criterion_param_string(self, name: str, value: str) -> Callable[[_F], _F]:
        """Set a string parameter for the stopping criterion."""
        return self.set_criterion_param_string(name, value)

    def set_criterion_param_string(self, name: str, value: str) -> Callable[[_F], _F]:
        """Set a string parameter for the stopping criterion."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.set_criterion_param_string(name, value)
        )

    def min_samples(self, count: int) -> Callable[[_F], _F]:
        """Set the minimum number of samples to collect."""
        return self.set_min_samples(count)

    def set_min_samples(self, count: int) -> Callable[[_F], _F]:
        """Set the minimum number of samples to collect."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.set_min_samples(count)
        )

    def cold_warmup_runs(self, count: int) -> Callable[[_F], _F]:
        """Set the number of cold measurement warmup runs."""
        return self.set_cold_warmup_runs(count)

    def set_cold_warmup_runs(self, count: int) -> Callable[[_F], _F]:
        """Set the number of cold measurement warmup runs."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.set_cold_warmup_runs(count)
        )

    def cold_max_warmup_walltime(self, duration_seconds: float) -> Callable[[_F], _F]:
        """Set the maximum walltime spent on cold measurement warmup runs."""
        return self.set_cold_max_warmup_walltime(duration_seconds)

    def set_cold_max_warmup_walltime(
        self, duration_seconds: float
    ) -> Callable[[_F], _F]:
        """Set the maximum walltime spent on cold measurement warmup runs."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.set_cold_max_warmup_walltime(duration_seconds)
        )

    def is_cpu_only(self, value: bool = True) -> Callable[[_F], _F]:
        """Set whether the benchmark only performs CPU work."""
        return self.set_is_cpu_only(value)

    def set_is_cpu_only(self, value: bool) -> Callable[[_F], _F]:
        """Set whether the benchmark only performs CPU work."""
        return _append_benchmark_action(
            lambda benchmark: benchmark.set_is_cpu_only(value)
        )


axis = _AxisDecorators()
option = _OptionDecorators()
