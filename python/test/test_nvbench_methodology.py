# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def methodology(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("nvbench_json.methodology")


def state(**metadata):
    return {
        "name": "state",
        "summaries": [{"tag": "nv/cold/time/gpu/mean"}],
        **metadata,
    }


@pytest.mark.parametrize("timer", ["cuda_events", "cupti"])
def test_matching_structured_timers(methodology, timer):
    methodology.require_matching_gpu_timers(
        state(gpu_timer=timer), state(gpu_timer=timer), "bench"
    )


def test_mismatched_timers_are_rejected(methodology):
    with pytest.raises(ValueError, match="incompatible GPU timing methodologies"):
        methodology.require_matching_gpu_timers(
            state(gpu_timer="cuda_events"), state(gpu_timer="cupti"), "bench"
        )


@pytest.mark.parametrize(
    "description",
    [
        None,
        "Mean isolated time (measured with CUDA events)",
        "Mean isolated time (measured with CUPTI)",
    ],
)
@pytest.mark.parametrize("missing_from", ["reference", "comparison"])
def test_missing_timer_metadata_is_rejected(methodology, description, missing_from):
    reference = state(gpu_timer="cuda_events")
    comparison = state(gpu_timer="cuda_events")
    missing = reference if missing_from == "reference" else comparison
    del missing["gpu_timer"]
    if description is not None:
        missing["summaries"][0]["description"] = description
    with pytest.raises(ValueError, match="missing required gpu_timer metadata"):
        methodology.require_matching_gpu_timers(reference, comparison, "bench")


@pytest.mark.parametrize("timer", ["future_timer", False, [], {}])
def test_unsupported_timer_metadata_is_rejected(methodology, timer):
    with pytest.raises(ValueError, match="unsupported gpu_timer"):
        methodology.require_matching_gpu_timers(
            state(gpu_timer=timer), state(gpu_timer="cuda_events"), "bench"
        )


def test_null_timer_with_gpu_data_is_rejected(methodology):
    with pytest.raises(ValueError, match="gpu_timer is null"):
        methodology.get_gpu_timer(state(gpu_timer=None))


def test_structured_timer_takes_precedence_over_description(methodology):
    reference = state(gpu_timer="cuda_events")
    reference["summaries"][0]["description"] = "measured with CUPTI"
    methodology.require_matching_gpu_timers(
        reference, state(gpu_timer="cuda_events"), "bench"
    )


def test_missing_timing_data_keeps_existing_comparison_handling(methodology):
    methodology.require_matching_gpu_timers(
        state(gpu_timer="cupti", summaries=None), state(gpu_timer="cupti"), "bench"
    )


def test_cpu_only_state_has_no_gpu_timer(methodology):
    assert methodology.get_gpu_timer(state(gpu_timer=None, summaries=[])) is None
