# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def ensure_optional_script_dependency(name, module):
    try:
        importlib.import_module(name)
    except ImportError:
        sys.modules[name] = module


def load_nvbench_compare():
    ensure_optional_script_dependency(
        "jsondiff", types.SimpleNamespace(diff=lambda *args, **kwargs: {})
    )
    ensure_optional_script_dependency(
        "tabulate",
        types.SimpleNamespace(
            __version__="0.8.10", tabulate=lambda *args, **kwargs: ""
        ),
    )
    ensure_optional_script_dependency(
        "colorama",
        types.SimpleNamespace(
            Fore=types.SimpleNamespace(
                BLUE="",
                GREEN="",
                RED="",
                RESET="",
                YELLOW="",
            )
        ),
    )

    module_path = Path(__file__).resolve().parents[1] / "scripts" / "nvbench_compare.py"
    spec = importlib.util.spec_from_file_location("nvbench_compare", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nvbench_compare = load_nvbench_compare()


def make_state(name):
    return {
        "name": name,
        "device": 0,
        "axis_values": [],
        "summaries": [
            {
                "tag": nvbench_compare.GPU_TIME_MEAN_TAG,
                "data": [{"name": "value", "type": "float64", "value": "1.0"}],
            },
            {
                "tag": nvbench_compare.GPU_TIME_STDEV_RELATIVE_TAG,
                "data": [{"name": "value", "type": "float64", "value": "0.01"}],
            },
        ],
    }


def make_benchmark(states):
    return {
        "name": "bench",
        "devices": [0],
        "axes": [],
        "states": states,
    }


def test_compare_benches_rejects_swapped_duplicate_state_counts():
    ref_benches = [
        make_benchmark(
            [
                make_state("state1"),
                make_state("state1"),
                make_state("state1"),
                make_state("state2"),
                make_state("state2"),
            ]
        )
    ]
    cmp_benches = [
        make_benchmark(
            [
                make_state("state1"),
                make_state("state1"),
                make_state("state2"),
                make_state("state2"),
                make_state("state2"),
            ]
        )
    ]

    with pytest.raises(ValueError, match="mismatched state occurrences"):
        nvbench_compare.compare_benches(
            ref_benches,
            cmp_benches,
            threshold=0.0,
            plot_along=None,
            plot=False,
            dark=False,
            axis_filters=[],
            benchmark_filters=[],
            no_color=True,
        )
