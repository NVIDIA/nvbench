import json
import struct

import cuda.bench as bench
import pytest


def test_bench_result_reads_jsonbin_relative_to_json_path(tmp_path):
    bin_dir = tmp_path / "result.json-bin"
    bin_dir.mkdir()
    (bin_dir / "0.bin").write_bytes(struct.pack("<3f", 1.0, 2.0, 4.0))

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
                                        "data": [
                                            {
                                                "name": "value",
                                                "type": "float64",
                                                "value": "0.75",
                                            }
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

    default_result = bench.BenchResult(str(json_fn))
    result = bench.BenchResult(str(json_fn), elapsed=0.25)

    assert bench.BenchResult.__module__ == bench.__name__
    assert default_result.code == 0
    assert default_result.elapsed == 0.0
    assert result.code == 0
    assert result.elapsed == 0.25
    state = result.subbenches["copy"].states[0]
    assert state.name() == "BlockSize[pow2]=8"
    assert state.bw == 0.75
    assert list(state.samples) == pytest.approx([1.0, 2.0, 4.0])
    centers = result.centers(lambda samples: sum(samples) / len(samples))
    assert set(centers) == {"copy"}
    assert set(centers["copy"]) == {"BlockSize[pow2]=8"}
    assert centers["copy"]["BlockSize[pow2]=8"] == pytest.approx(7.0 / 3.0)


def test_bench_result_code_and_elapsed_are_keyword_only():
    with pytest.raises(TypeError):
        bench.BenchResult("", 0, 0.0)
