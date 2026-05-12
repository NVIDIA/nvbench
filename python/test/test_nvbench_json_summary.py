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

import importlib.util
import json
from pathlib import Path


def load_nvbench_json_summary():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "nvbench_json_summary.py"
    )
    spec = importlib.util.spec_from_file_location("nvbench_json_summary", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nvbench_json_summary = load_nvbench_json_summary()


def write_result_json(path):
    path.write_text(
        json.dumps(
            {
                "devices": [
                    {
                        "id": 0,
                        "name": "Test GPU",
                    }
                ],
                "benchmarks": [
                    {
                        "name": "copy",
                        "devices": [0],
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
                                "device": 0,
                                "type_config_index": 0,
                                "axis_values": [
                                    {
                                        "name": "BlockSize",
                                        "type": "int64",
                                        "value": "256",
                                    }
                                ],
                                "summaries": [
                                    {
                                        "tag": "nv/cold/time/gpu/sample_size",
                                        "name": "Samples",
                                        "hint": "sample_size",
                                        "data": [
                                            {
                                                "name": "value",
                                                "type": "int64",
                                                "value": "12",
                                            }
                                        ],
                                    },
                                    {
                                        "tag": "nv/cold/time/gpu/mean",
                                        "name": "GPU Time",
                                        "hint": "duration",
                                        "data": [
                                            {
                                                "name": "value",
                                                "type": "float64",
                                                "value": "1.25e-6",
                                            }
                                        ],
                                    },
                                    {
                                        "tag": "nv/cold/time/gpu/stdev/relative",
                                        "name": "Noise",
                                        "hint": "percentage",
                                        "data": [
                                            {
                                                "name": "value",
                                                "type": "float64",
                                                "value": "0.015",
                                            }
                                        ],
                                    },
                                    {
                                        "tag": "nv/cold/bw/global/bytes_per_second",
                                        "name": "GlobalMem BW",
                                        "hint": "byte_rate",
                                        "data": [
                                            {
                                                "name": "value",
                                                "type": "float64",
                                                "value": "2.5e9",
                                            }
                                        ],
                                    },
                                    {
                                        "tag": "nv/cold/bw/global/utilization",
                                        "name": "BWUtil",
                                        "hint": "percentage",
                                        "data": [
                                            {
                                                "name": "value",
                                                "type": "float64",
                                                "value": "0.625",
                                            }
                                        ],
                                    },
                                    {
                                        "tag": "nv/cold/time/gpu/min",
                                        "name": "Min GPU Time",
                                        "hint": "duration",
                                        "hide": "Hidden by default.",
                                        "data": [
                                            {
                                                "name": "value",
                                                "type": "float64",
                                                "value": "1.0e-6",
                                            }
                                        ],
                                    },
                                ],
                                "is_skipped": False,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_json_summary_formats_nvbench_style_markdown(tmp_path):
    json_path = tmp_path / "result.json"
    write_result_json(json_path)

    result = nvbench_json_summary.BenchmarkResult.from_json(json_path)
    report = nvbench_json_summary.format_result(result)

    assert "# Benchmark Results" in report
    assert "## copy" in report
    assert "### [0] Test GPU" in report
    assert (
        "| BlockSize | Samples | GPU Time | Noise | GlobalMem BW | BWUtil |" in report
    )
    assert (
        "| 2^8 = 256 |     12x | 1.250 us | 1.50% |   2.500 GB/s | 62.50% |" in report
    )
    assert "Min GPU Time" not in report


def test_json_summary_formats_axis_values_like_markdown_printer():
    axes_by_name = {
        "BlockSize": {
            "name": "BlockSize",
            "type": "int64",
            "flags": "pow2",
        },
        "NumBlocks": {
            "name": "NumBlocks",
            "type": "int64",
            "flags": "",
        },
        "Duration": {
            "name": "Duration",
            "type": "float64",
            "flags": "",
        },
    }

    assert nvbench_json_summary.format_axis_value(
        {"name": "BlockSize", "type": "int64", "value": "256"}, axes_by_name
    ) == ("BlockSize", "2^8 = 256")
    assert nvbench_json_summary.format_axis_value(
        {"name": "NumBlocks", "type": "int64", "value": "64"}, axes_by_name
    ) == ("NumBlocks", "64")
    assert nvbench_json_summary.format_axis_value(
        {"name": "Duration", "type": "float64", "value": "0.123456789"},
        axes_by_name,
    ) == ("Duration", "0.12346")


def test_json_summary_cli_writes_output_file(tmp_path):
    json_path = tmp_path / "result.json"
    output_path = tmp_path / "summary.md"
    write_result_json(json_path)

    rc = nvbench_json_summary.main([str(json_path), "--output", str(output_path)])

    assert rc == 0
    assert "GlobalMem BW" in output_path.read_text(encoding="utf-8")
