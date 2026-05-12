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

import json
import struct
from dataclasses import dataclass

import cuda.bench
import cuda.bench.results as results
import pytest


def test_benchmark_result_reads_jsonbin_relative_to_json_path(tmp_path):
    bin_dir = tmp_path / "result.json-bin"
    bin_dir.mkdir()
    (bin_dir / "0.bin").write_bytes(struct.pack("<3f", 1.0, 2.0, 4.0))
    freq_bin_dir = tmp_path / "result.json-freqs-bin"
    freq_bin_dir.mkdir()
    (freq_bin_dir / "0.bin").write_bytes(struct.pack("<3f", 100.0, 200.0, 400.0))

    json_fn = tmp_path / "result.json"
    json_fn.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "name": "copy",
                        "axes": [
                            {
                                "name": "BlockSize",
                                "type": "int64",
                                "flags": "pow2",
                                "values": [
                                    {
                                        "input_string": "8",
                                        "description": "2^8 = 256",
                                        "value": 256,
                                    }
                                ],
                            }
                        ],
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
                                    {
                                        "tag": "nv/json/bin:nv/cold/sample_times",
                                        "data": [
                                            {
                                                "name": "filename",
                                                "type": "string",
                                                "value": "result.json-bin/0.bin",
                                            },
                                            {
                                                "name": "size",
                                                "type": "int64",
                                                "value": "3",
                                            },
                                        ],
                                    },
                                    {
                                        "tag": "nv/cold/bw/global/utilization",
                                        "name": "BWUtil",
                                        "hint": "percentage",
                                        "description": "Global memory utilization",
                                        "data": [
                                            {
                                                "name": "value",
                                                "type": "float64",
                                                "value": "0.75",
                                            }
                                        ],
                                    },
                                    {
                                        "tag": "nv/json/freqs-bin:nv/cold/sample_freqs",
                                        "data": [
                                            {
                                                "name": "filename",
                                                "type": "string",
                                                "value": "result.json-freqs-bin/0.bin",
                                            },
                                            {
                                                "name": "size",
                                                "type": "int64",
                                                "value": "3",
                                            },
                                        ],
                                    },
                                ],
                                "is_skipped": False,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    metadata = {"returncode": 0, "elapsed_seconds": 0.25}
    default_result = results.BenchmarkResult.from_json(json_fn)
    result = results.BenchmarkResult.from_json(json_fn, metadata=metadata)

    assert results.BenchmarkResult.__module__ == results.__name__
    assert results.BenchmarkResultSummary.__module__ == results.__name__
    assert not hasattr(cuda.bench, "BenchmarkResult")
    assert default_result.metadata is None
    assert result.metadata is metadata
    subbench = result["copy"]
    assert len(result) == 1
    assert list(result) == ["copy"]
    assert list(result.keys()) == ["copy"]
    assert list(result.values()) == [subbench]
    assert list(result.items()) == [("copy", subbench)]
    assert "copy" in result
    assert "missing" not in result
    state = subbench[0]
    assert len(subbench) == 1
    assert subbench[-1] is state
    assert subbench[:] == subbench.states
    assert list(subbench) == subbench.states
    with pytest.raises(IndexError):
        subbench[1]
    assert state.name() == "BlockSize[pow2]=8"
    assert state.bw == 0.75
    bw_summary = state.summaries["nv/cold/bw/global/utilization"]
    assert bw_summary.tag == "nv/cold/bw/global/utilization"
    assert bw_summary.name == "BWUtil"
    assert bw_summary.hint == "percentage"
    assert bw_summary.hide is None
    assert bw_summary.description == "Global memory utilization"
    assert bw_summary.value == pytest.approx(0.75)
    assert bw_summary["value"] == pytest.approx(0.75)
    assert state.summaries["nv/json/bin:nv/cold/sample_times"].data == {
        "filename": "result.json-bin/0.bin",
        "size": 3,
    }
    assert state.summaries["nv/json/freqs-bin:nv/cold/sample_freqs"].data == {
        "filename": "result.json-freqs-bin/0.bin",
        "size": 3,
    }
    assert state.samples is not None
    assert list(state.samples) == pytest.approx([1.0, 2.0, 4.0])
    assert state.frequencies is not None
    assert list(state.frequencies) == pytest.approx([100.0, 200.0, 400.0])
    centers = result.centers(lambda samples: sum(samples) / len(samples))
    assert set(centers) == {"copy"}
    assert set(centers["copy"]) == {"BlockSize[pow2]=8"}
    assert centers["copy"]["BlockSize[pow2]=8"] == pytest.approx(7.0 / 3.0)

    def weighted_mean(samples, frequencies):
        return sum(
            sample * frequency for sample, frequency in zip(samples, frequencies)
        ) / sum(frequencies)

    weighted_centers = result.centers_with_frequencies(weighted_mean)
    assert set(weighted_centers) == {"copy"}
    assert set(weighted_centers["copy"]) == {"BlockSize[pow2]=8"}
    assert weighted_centers["copy"]["BlockSize[pow2]=8"] == pytest.approx(3.0)
    assert subbench is result.subbenches["copy"]
    assert subbench.centers_with_frequencies(weighted_mean) == weighted_centers["copy"]
    with pytest.raises(KeyError):
        result["missing"]


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
    json_fn.write_text(
        json.dumps(
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
                                    {
                                        "tag": "nv/json/bin:nv/cold/sample_times",
                                        "data": [
                                            {
                                                "name": "filename",
                                                "type": "string",
                                                "value": "temp_data/axes_run1.json-bin/0.bin",
                                            },
                                            {
                                                "name": "size",
                                                "type": "int64",
                                                "value": "2",
                                            },
                                        ],
                                    },
                                    {
                                        "tag": "nv/json/freqs-bin:nv/cold/sample_freqs",
                                        "data": [
                                            {
                                                "name": "filename",
                                                "type": "string",
                                                "value": "temp_data/axes_run1.json-freqs-bin/0.bin",
                                            },
                                            {
                                                "name": "size",
                                                "type": "int64",
                                                "value": "2",
                                            },
                                        ],
                                    },
                                ],
                                "is_skipped": False,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
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


def test_benchmark_result_ignores_skipped_state_with_no_summaries():
    result = results.SubBenchmarkResult(
        {
            "name": "copy_sweep_grid_shape",
            "axes": [
                {
                    "name": "BlockSize",
                    "type": "int64",
                    "flags": "pow2",
                    "values": [
                        {
                            "input_string": "6",
                            "description": "2^6 = 64",
                            "value": 64,
                        },
                        {
                            "input_string": "8",
                            "description": "2^8 = 256",
                            "value": 256,
                        },
                    ],
                }
            ],
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

    assert len(result.states) == 1
    assert result.states[0].name() == "BlockSize[pow2]=6"


def test_benchmark_result_uses_none_for_unavailable_samples(tmp_path):
    json_fn = tmp_path / "result.json"
    json_fn.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "name": "copy",
                        "axes": [
                            {
                                "name": "BlockSize",
                                "type": "int64",
                                "flags": "pow2",
                                "values": [
                                    {
                                        "input_string": "8",
                                        "description": "2^8 = 256",
                                        "value": 256,
                                    },
                                    {
                                        "input_string": "9",
                                        "description": "2^9 = 512",
                                        "value": 512,
                                    },
                                ],
                            }
                        ],
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
                                    {
                                        "tag": "nv/json/bin:nv/cold/sample_times",
                                        "data": [
                                            {
                                                "name": "filename",
                                                "type": "string",
                                                "value": "result.json-bin/missing.bin",
                                            },
                                            {
                                                "name": "size",
                                                "type": "int64",
                                                "value": "3",
                                            },
                                        ],
                                    },
                                    {
                                        "tag": "nv/json/freqs-bin:nv/cold/sample_freqs",
                                        "data": [
                                            {
                                                "name": "filename",
                                                "type": "string",
                                                "value": "result.json-freqs-bin/missing.bin",
                                            },
                                            {
                                                "name": "size",
                                                "type": "int64",
                                                "value": "3",
                                            },
                                        ],
                                    },
                                ],
                                "is_skipped": False,
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
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
    json_fn.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "name": "copy",
                        "axes": [
                            {
                                "name": "BlockSize",
                                "type": "int64",
                                "flags": "pow2",
                                "values": [
                                    {
                                        "input_string": "8",
                                        "description": "2^8 = 256",
                                        "value": 256,
                                    }
                                ],
                            }
                        ],
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
                                    {
                                        "tag": "nv/json/bin:nv/cold/sample_times",
                                        "data": [
                                            {
                                                "name": "filename",
                                                "type": "string",
                                                "value": "result.json-bin/0.bin",
                                            },
                                            {
                                                "name": "size",
                                                "type": "int64",
                                                "value": "3",
                                            },
                                        ],
                                    },
                                    {
                                        "tag": "nv/json/freqs-bin:nv/cold/sample_freqs",
                                        "data": [
                                            {
                                                "name": "filename",
                                                "type": "string",
                                                "value": "result.json-freqs-bin/0.bin",
                                            },
                                            {
                                                "name": "size",
                                                "type": "int64",
                                                "value": "2",
                                            },
                                        ],
                                    },
                                ],
                                "is_skipped": False,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample count .* frequency count"):
        results.BenchmarkResult.from_json(json_fn)
