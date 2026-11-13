# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Identify the effective GPU timer before comparing timing results."""


def _is_gpu_timing(summary):
    tag = summary.get("tag", "")
    return tag.startswith(("nv/cold/time/gpu/", "nv/batch/time/gpu/")) or tag == (
        "nv/json/bin:nv/cold/sample_times"
    )


def get_gpu_timer(state):
    """Require explicit timer metadata rather than interpreting descriptions."""
    if "gpu_timer" not in state:
        raise ValueError(
            "missing required gpu_timer metadata; regenerate results with NVBench JSON 1.1.0 or newer"
        )
    timer = state["gpu_timer"]
    if timer is not None and timer not in ("cuda_events", "cupti"):
        raise ValueError(f"unsupported gpu_timer value: {timer!r}")
    if timer is None and any(
        _is_gpu_timing(summary) for summary in state.get("summaries") or []
    ):
        raise ValueError("gpu_timer is null but the state contains GPU timing data")
    return timer


def require_matching_gpu_timers(ref_state, cmp_state, benchmark_name):
    """Reject comparisons of different or unidentified GPU timing methodologies."""
    context = f"benchmark {benchmark_name!r} state {cmp_state.get('name')!r}"
    try:
        ref_timer = get_gpu_timer(ref_state)
        cmp_timer = get_gpu_timer(cmp_state)
    except ValueError as exc:
        raise ValueError(f"{context}: {exc}") from exc
    if ref_timer != cmp_timer:
        raise ValueError(
            f"{context}: incompatible GPU timing methodologies: "
            f"reference={ref_timer!r}, comparison={cmp_timer!r}"
        )
