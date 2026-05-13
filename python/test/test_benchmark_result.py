# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import struct
from dataclasses import dataclass

import cuda.bench
import cuda.bench.results as results
import pytest


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def block_size_axis(*values):
    return {
        "name": "BlockSize",
        "type": "int64",
        "flags": "pow2",
        "values": [
            {
                "input_string": str(value),
                "description": f"2^{value} = {2**value}",
                "value": 2**value,
            }
            for value in values
        ],
    }


def sample_file_summary(tag, filename, size):
    return {
        "tag": tag,
        "data": [
            {
                "name": "filename",
                "type": "string",
                "value": filename,
            },
            {
                "name": "size",
                "type": "int64",
                "value": str(size),
            },
        ],
    }


def sample_times_summary(filename, size):
    return sample_file_summary(
        "nv/json/bin:nv/cold/sample_times",
        filename,
        size,
    )


def sample_frequencies_summary(filename, size):
    return sample_file_summary(
        "nv/json/freqs-bin:nv/cold/sample_freqs",
        filename,
        size,
    )


def bwutil_summary(value):
    return {
        "tag": "nv/cold/bw/global/utilization",
        "name": "BWUtil",
        "hint": "percentage",
        "description": "Global memory utilization",
        "data": [
            {
                "name": "value",
                "type": "float64",
                "value": str(value),
            }
        ],
    }


@pytest.fixture
def sample_result_path(tmp_path):
    bin_dir = tmp_path / "result.json-bin"
    bin_dir.mkdir()
    (bin_dir / "0.bin").write_bytes(struct.pack("<3f", 1.0, 2.0, 4.0))
    freq_bin_dir = tmp_path / "result.json-freqs-bin"
    freq_bin_dir.mkdir()
    (freq_bin_dir / "0.bin").write_bytes(struct.pack("<3f", 100.0, 200.0, 400.0))

    json_fn = tmp_path / "result.json"
    write_json(
        json_fn,
        {
            "benchmarks": [
                {
                    "name": "copy",
                    "axes": [block_size_axis(8)],
                    "states": [
                        {
                            "name": "Device=0 BlockSize=2^8",
                            "axis_values": [
                                {
                                    "name": "BlockSize",
                                    "type": "int64",
                                    "value": "256",
                                }
                            ],
                            "summaries": [
                                sample_times_summary("result.json-bin/0.bin", 3),
                                bwutil_summary(0.75),
                                sample_frequencies_summary(
                                    "result.json-freqs-bin/0.bin",
                                    3,
                                ),
                            ],
                            "is_skipped": False,
                        }
                    ],
                }
            ]
        },
    )
    return json_fn


@pytest.fixture
def sample_result(sample_result_path):
    return results.BenchmarkResult.from_json(sample_result_path)


@pytest.fixture
def sample_subbenchmark(sample_result):
    return sample_result["copy"]


@pytest.fixture
def sample_state(sample_subbenchmark):
    return sample_subbenchmark[0]


def test_result_classes_are_exposed_from_results_namespace():
    assert results.BenchmarkResult.__module__ == results.__name__
    assert results.BenchmarkResultSummary.__module__ == results.__name__
    assert not hasattr(cuda.bench, "BenchmarkResult")


def test_from_json_preserves_optional_metadata(sample_result_path):
    metadata = {"returncode": 0, "elapsed_seconds": 0.25}

    default_result = results.BenchmarkResult.from_json(sample_result_path)
    result = results.BenchmarkResult.from_json(sample_result_path, metadata=metadata)

    assert default_result.metadata is None
    assert result.metadata is metadata


def test_benchmark_result_implements_mapping_protocol(sample_result):
    subbenchmark = sample_result["copy"]

    assert len(sample_result) == 1
    assert list(sample_result) == ["copy"]
    assert list(sample_result.keys()) == ["copy"]
    assert list(sample_result.values()) == [subbenchmark]
    assert list(sample_result.items()) == [("copy", subbenchmark)]
    assert "copy" in sample_result
    assert "missing" not in sample_result
    assert subbenchmark is sample_result.subbenches["copy"]
    with pytest.raises(KeyError):
        sample_result["missing"]


def test_subbenchmark_result_implements_sequence_protocol(sample_subbenchmark):
    state = sample_subbenchmark[0]

    assert len(sample_subbenchmark) == 1
    assert sample_subbenchmark[-1] is state
    assert sample_subbenchmark[:] == sample_subbenchmark.states
    assert list(sample_subbenchmark) == sample_subbenchmark.states
    with pytest.raises(IndexError):
        sample_subbenchmark[1]


def test_state_parses_axis_name_and_bandwidth(sample_state):
    assert sample_state.name() == "BlockSize[pow2]=8"
    assert sample_state.bw == 0.75


def test_state_stores_rich_summary_metadata(sample_state):
    bw_summary = sample_state.summaries["nv/cold/bw/global/utilization"]

    assert bw_summary.tag == "nv/cold/bw/global/utilization"
    assert bw_summary.name == "BWUtil"
    assert bw_summary.hint == "percentage"
    assert bw_summary.hide is None
    assert bw_summary.description == "Global memory utilization"
    assert bw_summary.value == pytest.approx(0.75)
    assert bw_summary["value"] == pytest.approx(0.75)
    assert sample_state.summaries["nv/json/bin:nv/cold/sample_times"].data == {
        "filename": "result.json-bin/0.bin",
        "size": 3,
    }
    assert sample_state.summaries["nv/json/freqs-bin:nv/cold/sample_freqs"].data == {
        "filename": "result.json-freqs-bin/0.bin",
        "size": 3,
    }


def test_state_preserves_null_summary_values(tmp_path):
    json_fn = tmp_path / "result.json"
    write_json(
        json_fn,
        {
            "benchmarks": [
                {
                    "name": "copy",
                    "axes": [],
                    "states": [
                        {
                            "name": "Device=0",
                            "axis_values": [],
                            "summaries": [
                                {
                                    "tag": "nv/cold/time/gpu/stdev/relative",
                                    "name": "Noise",
                                    "hint": "percentage",
                                    "data": [
                                        {
                                            "name": "value",
                                            "type": "float64",
                                            "value": None,
                                        }
                                    ],
                                }
                            ],
                            "is_skipped": False,
                        }
                    ],
                }
            ]
        },
    )

    summary = results.BenchmarkResult.from_json(json_fn)["copy"][0].summaries[
        "nv/cold/time/gpu/stdev/relative"
    ]

    assert summary.value is None
    assert summary["value"] is None


def test_state_reports_malformed_numeric_summary_values(tmp_path):
    json_fn = tmp_path / "result.json"
    write_json(
        json_fn,
        {
            "benchmarks": [
                {
                    "name": "copy",
                    "axes": [],
                    "states": [
                        {
                            "name": "Device=0",
                            "axis_values": [],
                            "summaries": [
                                {
                                    "tag": "nv/cold/time/gpu/mean",
                                    "name": "GPU Time",
                                    "hint": "duration",
                                    "data": [
                                        {
                                            "name": "value",
                                            "type": "float64",
                                            "value": "not-a-number",
                                        }
                                    ],
                                }
                            ],
                            "is_skipped": False,
                        }
                    ],
                }
            ]
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "summary 'nv/cold/time/gpu/mean' field 'value' "
            "value 'not-a-number' is not a float64"
        ),
    ):
        results.BenchmarkResult.from_json(json_fn)


def test_state_loads_samples_and_frequencies(sample_state):
    assert sample_state.samples is not None
    assert list(sample_state.samples) == pytest.approx([1.0, 2.0, 4.0])
    assert sample_state.frequencies is not None
    assert list(sample_state.frequencies) == pytest.approx([100.0, 200.0, 400.0])


def test_centers_apply_estimators_to_samples(sample_result):
    centers = sample_result.centers(lambda samples: sum(samples) / len(samples))

    assert centers == {"copy": {"BlockSize[pow2]=8": pytest.approx(7.0 / 3.0)}}


def test_centers_with_frequencies_apply_estimators(sample_result, sample_subbenchmark):
    def weighted_mean(samples, frequencies):
        return sum(
            sample * frequency for sample, frequency in zip(samples, frequencies)
        ) / sum(frequencies)

    weighted_centers = sample_result.centers_with_frequencies(weighted_mean)

    assert weighted_centers == {"copy": {"BlockSize[pow2]=8": pytest.approx(3.0)}}
    assert (
        sample_subbenchmark.centers_with_frequencies(weighted_mean)
        == weighted_centers["copy"]
    )


def test_benchmark_result_constructor_is_private():
    with pytest.raises(TypeError, match="from_json\\(\\).*empty\\(\\)"):
        results.BenchmarkResult()
    with pytest.raises(TypeError, match="from_json\\(\\).*empty\\(\\)"):
        results.BenchmarkResult("result.json")
    with pytest.raises(TypeError):
        results.BenchmarkResult(metadata=None)
    with pytest.raises(TypeError):
        results.BenchmarkResult(json_path="result.json", parse=False)


def test_benchmark_result_empty_does_not_read_json(tmp_path):
    @dataclass
    class RunMetadata:
        returncode: int
        elapsed_seconds: float

    metadata = RunMetadata(returncode=1, elapsed_seconds=0.25)
    missing_json = tmp_path / "missing.json"

    result = results.BenchmarkResult.empty(metadata=metadata)

    assert result.metadata is metadata
    assert result.subbenches == {}

    with pytest.raises(FileNotFoundError):
        results.BenchmarkResult.from_json(missing_json, metadata=metadata)
    with pytest.raises(FileNotFoundError):
        results.BenchmarkResult.from_json(json_path=missing_json, metadata=metadata)


def test_benchmark_result_accepts_no_axis_benchmark_with_recorded_binary_path(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "temp_data"
    data_dir.mkdir()
    bin_dir = data_dir / "axes_run1.json-bin"
    bin_dir.mkdir()
    (bin_dir / "0.bin").write_bytes(struct.pack("<2f", 1.0, 4.0))
    freq_bin_dir = data_dir / "axes_run1.json-freqs-bin"
    freq_bin_dir.mkdir()
    (freq_bin_dir / "0.bin").write_bytes(struct.pack("<2f", 100.0, 400.0))

    json_fn = data_dir / "axes_run1.json"
    write_json(
        json_fn,
        {
            "benchmarks": [
                {
                    "name": "simple",
                    "axes": None,
                    "states": [
                        {
                            "name": "Device=0",
                            "axis_values": None,
                            "summaries": [
                                sample_times_summary(
                                    "temp_data/axes_run1.json-bin/0.bin",
                                    2,
                                ),
                                sample_frequencies_summary(
                                    "temp_data/axes_run1.json-freqs-bin/0.bin",
                                    2,
                                ),
                            ],
                            "is_skipped": False,
                        }
                    ],
                }
            ]
        },
    )

    monkeypatch.chdir(tmp_path)

    result = results.BenchmarkResult.from_json("temp_data/axes_run1.json")

    state = result.subbenches["simple"].states[0]
    assert state.name() == "Device=0"
    assert state.point == {}
    assert state.samples is not None
    assert list(state.samples) == pytest.approx([1.0, 4.0])
    assert state.frequencies is not None
    assert list(state.frequencies) == pytest.approx([100.0, 400.0])


def test_benchmark_result_accepts_axis_value_input_string():
    result = results.SubBenchmarkResult(
        {
            "name": "single_float64_axis",
            "axes": [
                {
                    "name": "Duration",
                    "type": "float64",
                    "flags": "",
                    "values": [
                        {
                            "input_string": "0",
                            "description": "",
                            "value": 0.0,
                        }
                    ],
                }
            ],
            "states": [
                {
                    "name": "Device=0 Duration=0",
                    "axis_values": [
                        {
                            "name": "Duration",
                            "type": "float64",
                            "value": "0",
                        }
                    ],
                    "summaries": [],
                    "is_skipped": False,
                }
            ],
        },
        "",
    )

    state = result.states[0]
    assert state.name() == "Duration=0"
    assert state.point == {"Duration": "0"}


def test_benchmark_result_normalizes_axis_value_lookup_key():
    result = results.SubBenchmarkResult(
        {
            "name": "num_blocks",
            "axes": [
                {
                    "name": "NumBlocks",
                    "type": "int64",
                    "flags": "",
                    "values": [
                        {
                            "input_string": "64",
                            "description": "",
                            "value": 64,
                        },
                        {
                            "input_string": "default",
                            "description": "",
                            "value": None,
                        },
                    ],
                }
            ],
            "states": [
                {
                    "name": "Device=0 NumBlocks=64",
                    "axis_values": [
                        {
                            "name": "NumBlocks",
                            "type": "int64",
                            "value": 64,
                        }
                    ],
                    "summaries": [],
                    "is_skipped": False,
                },
                {
                    "name": "Device=0 NumBlocks=default",
                    "axis_values": [
                        {
                            "name": "NumBlocks",
                            "type": "int64",
                            "value": None,
                        }
                    ],
                    "summaries": [],
                    "is_skipped": False,
                },
                {
                    "name": "Device=0 NumBlocks=64",
                    "axis_values": [
                        {
                            "name": "NumBlocks",
                            "type": "int64",
                            "input_string": "64",
                        }
                    ],
                    "summaries": [],
                    "is_skipped": False,
                },
            ],
        },
        "",
    )

    assert result.states[0].point == {"NumBlocks": "64"}
    assert result.states[1].point == {"NumBlocks": "default"}
    assert result.states[2].point == {"NumBlocks": "64"}


def test_benchmark_result_preserves_skipped_state_with_no_summaries():
    result = results.SubBenchmarkResult(
        {
            "name": "copy_sweep_grid_shape",
            "axes": [block_size_axis(6, 8)],
            "states": [
                {
                    "name": "Device=0 BlockSize=2^8",
                    "axis_values": [
                        {
                            "name": "BlockSize",
                            "type": "int64",
                            "value": "256",
                        }
                    ],
                    "summaries": None,
                    "is_skipped": True,
                },
                {
                    "name": "Device=0 BlockSize=2^6",
                    "axis_values": [
                        {
                            "name": "BlockSize",
                            "type": "int64",
                            "value": "64",
                        }
                    ],
                    "summaries": [],
                    "is_skipped": False,
                },
            ],
        },
        "",
    )

    assert len(result.states) == 2
    assert result.states[0].name() == "BlockSize[pow2]=8"
    assert result.states[0].is_skipped is True
    assert result.states[0].summaries == {}
    assert result.states[0].samples is None
    assert result.states[0].frequencies is None
    assert result.states[1].name() == "BlockSize[pow2]=6"
    assert result.states[1].is_skipped is False


def test_benchmark_result_uses_empty_summaries_when_field_is_missing():
    result = results.SubBenchmarkResult(
        {
            "name": "copy_sweep_grid_shape",
            "axes": [block_size_axis(8)],
            "states": [
                {
                    "name": "Device=0 BlockSize=2^8",
                    "axis_values": [
                        {
                            "name": "BlockSize",
                            "type": "int64",
                            "value": "256",
                        }
                    ],
                    "is_skipped": False,
                },
            ],
        },
        "",
    )

    state = result.states[0]
    assert state.name() == "BlockSize[pow2]=8"
    assert state.summaries == {}
    assert state.samples is None
    assert state.frequencies is None
    assert state.bw is None


@pytest.mark.parametrize(
    "field_name,bad_type,expected_type",
    [
        ("filename", "int64", "string"),
        ("size", "string", "int64"),
    ],
)
def test_benchmark_result_validates_binary_summary_field_types(
    field_name, bad_type, expected_type
):
    summary = sample_times_summary("result.json-bin/0.bin", 3)
    for value_data in summary["data"]:
        if value_data["name"] == field_name:
            value_data["type"] = bad_type
            if field_name == "filename":
                value_data["value"] = "123"

    with pytest.raises(
        ValueError,
        match=rf"field '{field_name}' has type '{bad_type}'; expected '{expected_type}'",
    ):
        results.SubBenchmarkResult(
            {
                "name": "copy",
                "axes": [],
                "states": [
                    {
                        "name": "Device=0",
                        "axis_values": [],
                        "summaries": [summary],
                        "is_skipped": False,
                    }
                ],
            },
            "",
        )


def test_benchmark_result_uses_none_for_unavailable_samples(tmp_path):
    json_fn = tmp_path / "result.json"
    write_json(
        json_fn,
        {
            "benchmarks": [
                {
                    "name": "copy",
                    "axes": [block_size_axis(8, 9)],
                    "states": [
                        {
                            "name": "Device=0 BlockSize=2^8",
                            "axis_values": [
                                {
                                    "name": "BlockSize",
                                    "type": "int64",
                                    "value": "256",
                                }
                            ],
                            "summaries": [],
                            "is_skipped": False,
                        },
                        {
                            "name": "Device=0 BlockSize=2^9",
                            "axis_values": [
                                {
                                    "name": "BlockSize",
                                    "type": "int64",
                                    "value": "512",
                                }
                            ],
                            "summaries": [
                                sample_times_summary(
                                    "result.json-bin/missing.bin",
                                    3,
                                ),
                                sample_frequencies_summary(
                                    "result.json-freqs-bin/missing.bin",
                                    3,
                                ),
                            ],
                            "is_skipped": False,
                        },
                    ],
                }
            ]
        },
    )

    result = results.BenchmarkResult.from_json(json_fn)

    states = result.subbenches["copy"].states
    assert states[0].samples is None
    assert states[1].samples is None
    assert states[0].frequencies is None
    assert states[1].frequencies is None
    assert result.centers(lambda samples: pytest.fail("estimator should not run")) == {
        "copy": {
            "BlockSize[pow2]=8": None,
            "BlockSize[pow2]=9": None,
        }
    }
    assert result.centers_with_frequencies(
        lambda samples, frequencies: pytest.fail("estimator should not run")
    ) == {
        "copy": {
            "BlockSize[pow2]=8": None,
            "BlockSize[pow2]=9": None,
        }
    }


def test_benchmark_result_rejects_mismatched_sample_and_frequency_counts(tmp_path):
    bin_dir = tmp_path / "result.json-bin"
    bin_dir.mkdir()
    (bin_dir / "0.bin").write_bytes(struct.pack("<3f", 1.0, 2.0, 4.0))
    freq_bin_dir = tmp_path / "result.json-freqs-bin"
    freq_bin_dir.mkdir()
    (freq_bin_dir / "0.bin").write_bytes(struct.pack("<2f", 100.0, 200.0))

    json_fn = tmp_path / "result.json"
    write_json(
        json_fn,
        {
            "benchmarks": [
                {
                    "name": "copy",
                    "axes": [block_size_axis(8)],
                    "states": [
                        {
                            "name": "Device=0 BlockSize=2^8",
                            "axis_values": [
                                {
                                    "name": "BlockSize",
                                    "type": "int64",
                                    "value": "256",
                                }
                            ],
                            "summaries": [
                                sample_times_summary("result.json-bin/0.bin", 3),
                                sample_frequencies_summary(
                                    "result.json-freqs-bin/0.bin",
                                    2,
                                ),
                            ],
                            "is_skipped": False,
                        }
                    ],
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="sample count .* frequency count"):
        results.BenchmarkResult.from_json(json_fn)
